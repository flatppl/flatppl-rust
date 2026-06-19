//! A fixed-size worker pool for off-main-thread LSP request handling.
//!
//! Each job is a `FnOnce` that runs a salsa query on a cloned [`Database`]
//! handle and pushes its `Message::Response` onto the results channel. Salsa
//! cancellation (a concurrent input write on the main thread) unwinds an
//! in-flight job with `salsa::Cancelled`, which the job body catches and drops
//! — the stale request is silently abandoned.
//!
//! [`Database`]: crate::db::Database

use crossbeam_channel::{Receiver, Sender, unbounded};
use lsp_server::Message;
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A handle to a fixed pool of worker threads, each pulling [`Job`]s off a
/// shared channel and running them to completion.
///
/// Dropping the `Pool` drops `job_tx`; once the last sender is gone the workers'
/// `for job in job_rx` loops terminate and the threads exit. The `JoinHandle`s
/// are retained only to keep the threads owned for the lifetime of the pool.
pub struct Pool {
    job_tx: Sender<Job>,
    _workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Spawn `threads` worker threads (at least one). Each pulls jobs off the
    /// shared channel and runs them; a job is responsible for sending its own
    /// response on the results channel it captures in its closure.
    ///
    /// `_results` is accepted to document the data flow (jobs send onto a clone
    /// of this sender) but is not stored — the pool itself never produces
    /// messages; only the jobs do, via the senders they close over.
    pub fn new(threads: usize, _results: Sender<Message>) -> Self {
        let (job_tx, job_rx): (Sender<Job>, Receiver<Job>) = unbounded();
        let workers = (0..threads.max(1))
            .map(|_| {
                let job_rx = job_rx.clone();
                std::thread::spawn(move || {
                    for job in job_rx {
                        job();
                    }
                })
            })
            .collect();
        Pool {
            job_tx,
            _workers: workers,
        }
    }

    /// Enqueue a job to run on a worker thread.
    ///
    /// If every worker has already exited (channel disconnected) the send fails
    /// and the job is dropped; this is benign — it only happens during shutdown.
    pub fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        let _ = self.job_tx.send(Box::new(job));
    }
}
