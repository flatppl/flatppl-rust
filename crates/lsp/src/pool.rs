//! A fixed-size worker pool for off-main-thread LSP request handling.
//!
//! Each job is a `FnOnce` that runs a salsa query on a cloned [`Database`]
//! handle and pushes its `Message::Response` onto the results channel. Salsa
//! cancellation (a concurrent input write on the main thread) unwinds an
//! in-flight job with `salsa::Cancelled`, which the job body catches and
//! answers with `ContentModified` — JSON-RPC 2.0 §5 requires a response to
//! every request, and LSP 3.17 defines that code for a revision an edit
//! invalidated.
//!
//! The queue is **bounded**. Every queued job holds a `Database` clone taken
//! before enqueue, and salsa's `cancel_others` (run by an input write) blocks
//! until every outstanding clone drops — so an unbounded queue makes the next
//! edit's latency proportional to the client's backlog. The bound caps the
//! outstanding clones, and the main thread drains the queue on an input write
//! rather than waiting for it (see `server::run`).
//!
//! [`Database`]: crate::db::Database

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use lsp_server::RequestId;
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Queued requests allowed before the pool refuses. A well-behaved client keeps
/// a handful of requests outstanding; this is a safety valve on memory and on
/// the number of live salsa handles, not a throughput knob.
pub const QUEUE_CAPACITY: usize = 256;

/// A handle to a fixed pool of worker threads, each pulling [`Job`]s off a
/// shared bounded channel and running them to completion.
///
/// Dropping the `Pool` drops `job_tx`; once the last sender is gone the workers'
/// `for job in job_rx` loops terminate and the threads exit. The `JoinHandle`s
/// are retained only to keep the threads owned for the lifetime of the pool.
pub struct Pool {
    job_tx: Sender<(RequestId, Job)>,
    /// Retained so the main thread can [`drain_queued`](Self::drain_queued).
    job_rx: Receiver<(RequestId, Job)>,
    _workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Spawn `threads` worker threads (at least one) over a queue holding at
    /// most `capacity` jobs. Each worker pulls jobs off the shared channel and
    /// runs them; a job is responsible for sending its own response on the
    /// results channel it captures in its closure.
    pub fn new(threads: usize, capacity: usize) -> Self {
        let (job_tx, job_rx) = bounded::<(RequestId, Job)>(capacity.max(1));
        let workers = (0..threads.max(1))
            .map(|_| {
                let job_rx = job_rx.clone();
                std::thread::spawn(move || {
                    for (_id, job) in job_rx {
                        job();
                    }
                })
            })
            .collect();
        Pool {
            job_tx,
            job_rx,
            _workers: workers,
        }
    }

    /// Enqueue `job` for request `id`.
    ///
    /// `Err(id)` when the queue is full: the caller must answer that request
    /// itself, since JSON-RPC requires a response to every request. A
    /// disconnected channel (every worker exited, i.e. shutdown) is reported
    /// the same way; the caller's answer is then simply never delivered.
    pub fn try_spawn(
        &self,
        id: RequestId,
        job: impl FnOnce() + Send + 'static,
    ) -> Result<(), RequestId> {
        match self.job_tx.try_send((id, Box::new(job))) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full((id, _))) | Err(TrySendError::Disconnected((id, _))) => Err(id),
        }
    }

    /// Take every job still waiting in the queue and return its request id,
    /// dropping the job unrun.
    ///
    /// Dropping a job releases the `Database` clone it captured, which is what
    /// lets a pending input write proceed without waiting for the backlog. The
    /// caller owes each returned id a response. Jobs a worker has already
    /// picked up are not affected — they run to completion or unwind with
    /// `salsa::Cancelled`.
    pub fn drain_queued(&self) -> Vec<RequestId> {
        self.job_rx.try_iter().map(|(id, _job)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    /// Occupy the pool's single worker and return the barrier that releases it.
    /// The worker signals on `started` before parking, so the test never races
    /// the pull: everything enqueued after this is still in the queue.
    fn occupy_the_worker(pool: &Pool) -> Arc<Barrier> {
        let gate = Arc::new(Barrier::new(2));
        let held = gate.clone();
        let (started_tx, started_rx) = crossbeam_channel::bounded::<()>(1);
        pool.try_spawn(RequestId::from(1i32), move || {
            let _ = started_tx.send(());
            held.wait();
        })
        .expect("the first job goes straight to the idle worker");
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the worker must pick up the first job");
        gate
    }

    #[test]
    fn a_full_queue_hands_the_request_id_back() {
        // One worker, parked, and a one-slot queue: the second request fills
        // the slot and the third has nowhere to go.
        let pool = Pool::new(1, 1);
        let gate = occupy_the_worker(&pool);
        assert!(
            pool.try_spawn(RequestId::from(2i32), || {}).is_ok(),
            "the single queue slot is free"
        );
        assert_eq!(
            pool.try_spawn(RequestId::from(3i32), || {}),
            Err(RequestId::from(3i32)),
            "a full queue refuses and returns the id, so the caller can answer it"
        );
        gate.wait();
    }

    #[test]
    fn draining_returns_the_queued_ids_and_does_not_run_them() {
        let ran = Arc::new(AtomicUsize::new(0));
        let pool = Pool::new(1, 16);
        let gate = occupy_the_worker(&pool);
        for i in 2..6i32 {
            let ran = ran.clone();
            pool.try_spawn(RequestId::from(i), move || {
                ran.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        }
        let mut drained = pool.drain_queued();
        drained.sort_by_key(|id| format!("{id}"));
        assert_eq!(
            drained,
            (2..6i32).map(RequestId::from).collect::<Vec<_>>(),
            "every queued job comes back"
        );
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "a drained job must not have run"
        );
        assert!(
            pool.drain_queued().is_empty(),
            "a second drain finds nothing"
        );
        gate.wait();
    }
}
