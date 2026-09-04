//! The write half of the client connection.
//!
//! `lsp_server::Response` carries an owned `serde_json::Value`, and a `Value`
//! costs roughly 38x the JSON text it stands for: every `DocumentSymbol`
//! becomes seven nested `Map`s, and serde_json's `Map` is a `BTreeMap`, so each
//! one allocates a node however few keys it holds. On a 4,000-binding file that
//! turns a 0.77 MiB answer into about 19 MiB, and building it costs 3.5 ms of
//! CPU plus 1.4 ms to serialise back out — per request, for an answer that is
//! identical for every request against one revision.
//!
//! There is no way to hand `lsp_server`'s writer a payload that is already
//! JSON: serde_json's `RawValue` does not survive `to_value` either, because
//! its emitter re-parses the text into a full `Value`. So this module owns the
//! writer instead, and [`Outgoing::RawResult`] carries the response body as
//! shared text that is memoized per revision and written straight through.
//!
//! Exactly one writer may hold the output stream: framing is a header write
//! followed by a body write, so two writers on one descriptor would interleave
//! mid-message. When [`Outbound::to_writer`] owns stdout, nothing else in the
//! process may write to it — including `lsp_server`'s own writer thread, whose
//! sender must be dropped before this one starts.

use crossbeam_channel::Sender;
use lsp_server::{Message, RequestId};
use std::io::{self, Write};
use std::sync::Arc;
use std::thread::JoinHandle;

/// One outgoing JSON-RPC message.
pub enum Outgoing {
    /// A typed message, serialised at write time.
    Msg(Message),
    /// A successful response whose `result` is already JSON text.
    ///
    /// The text is shared: every request answered from one revision holds the
    /// same allocation, so a queued answer costs a refcount rather than a copy.
    RawResult { id: RequestId, result: Arc<str> },
}

impl From<Message> for Outgoing {
    fn from(msg: Message) -> Self {
        Outgoing::Msg(msg)
    }
}

/// A handle for sending to the client, plus the depth of the writer's backlog.
#[derive(Clone)]
pub struct Outbound {
    tx: Sender<Outgoing>,
}

/// The writer thread, joined at shutdown.
pub type WriterThread = JoinHandle<io::Result<()>>;

impl Outbound {
    /// Frame every message onto `w` from a dedicated thread.
    ///
    /// The thread ends when every [`Outbound`] clone is dropped, and returns
    /// the first write error it hit.
    pub fn to_writer<W: Write + Send + 'static>(mut w: W) -> (Outbound, WriterThread) {
        let (tx, rx) = crossbeam_channel::unbounded::<Outgoing>();
        let thread = std::thread::spawn(move || {
            for out in rx {
                write_framed(&mut w, &out)?;
            }
            w.flush()
        });
        (Outbound { tx }, thread)
    }

    /// Forward every message to `tx` as an `lsp_server::Message`.
    ///
    /// For a host that drives the server over a channel rather than a stream.
    /// A [`Outgoing::RawResult`] has to be parsed back into a `Value` here,
    /// which is the cost this module exists to avoid — so this backend is for
    /// in-process hosts and tests, not for the stdio server.
    pub fn to_messages(tx: Sender<Message>) -> (Outbound, JoinHandle<()>) {
        let (out_tx, rx) = crossbeam_channel::unbounded::<Outgoing>();
        let thread = std::thread::spawn(move || {
            for out in rx {
                let msg = match out {
                    Outgoing::Msg(msg) => msg,
                    Outgoing::RawResult { id, result } => {
                        let value =
                            serde_json::from_str(&result).unwrap_or(serde_json::Value::Null);
                        Message::Response(lsp_server::Response {
                            id,
                            result: Some(value),
                            error: None,
                        })
                    }
                };
                if tx.send(msg).is_err() {
                    return; // the host hung up
                }
            }
        });
        (Outbound { tx: out_tx }, thread)
    }

    /// Queue `out` for the client.
    pub fn send(
        &self,
        out: impl Into<Outgoing>,
    ) -> Result<(), crossbeam_channel::SendError<Outgoing>> {
        self.tx.send(out.into())
    }

    /// Messages queued and not yet written.
    ///
    /// The main loop throttles on this: a queued answer that still holds a
    /// `serde_json::Value` is resident memory, so it stops producing more
    /// while the client is behind.
    pub fn backlog(&self) -> usize {
        self.tx.len()
    }
}

/// Write one message with its `Content-Length` header, then flush.
///
/// A typed message goes through `Message::write`, which is what adds the
/// `jsonrpc` member — `serde_json::to_string` on a `Message` alone omits it.
/// The raw body spells the same envelope out by hand.
fn write_framed<W: Write>(w: &mut W, out: &Outgoing) -> io::Result<()> {
    match out {
        Outgoing::Msg(msg) => msg.write(w),
        Outgoing::RawResult { id, result } => {
            let id = serde_json::to_string(id)?;
            let mut body = String::with_capacity(result.len() + id.len() + 40);
            body.push_str("{\"jsonrpc\":\"2.0\",\"id\":");
            body.push_str(&id);
            body.push_str(",\"result\":");
            body.push_str(result);
            body.push('}');
            write!(w, "Content-Length: {}\r\n\r\n", body.len())?;
            w.write_all(body.as_bytes())?;
            w.flush()
        }
    }
}

/// A `Receiver<Outgoing>` is not part of the public surface; this exists so the
/// framing can be unit-tested against a buffer without a thread.
#[cfg(test)]
fn drain_to_vec(rx: &crossbeam_channel::Receiver<Outgoing>) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    for out in rx.try_iter() {
        write_framed(&mut buf, &out)?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The channel is only read by the writer thread, so a test that wants the
    /// bytes builds its own pair.
    fn pair() -> (Outbound, crossbeam_channel::Receiver<Outgoing>) {
        let (tx, rx) = crossbeam_channel::unbounded::<Outgoing>();
        (Outbound { tx }, rx)
    }

    #[test]
    fn a_raw_result_frames_as_a_jsonrpc_response() {
        let (out, rx) = pair();
        out.send(Outgoing::RawResult {
            id: RequestId::from(7i32),
            result: Arc::from(r#"[{"name":"x"}]"#),
        })
        .unwrap();
        let bytes = drain_to_vec(&rx).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let (header, body) = text.split_once("\r\n\r\n").expect("framed message");
        assert_eq!(header, format!("Content-Length: {}", body.len()));
        assert_eq!(
            body, r#"{"jsonrpc":"2.0","id":7,"result":[{"name":"x"}]}"#,
            "the raw text goes on the wire untouched"
        );
    }

    #[test]
    fn a_string_request_id_stays_a_json_string() {
        let (out, rx) = pair();
        out.send(Outgoing::RawResult {
            id: RequestId::from("abc".to_owned()),
            result: Arc::from("[]"),
        })
        .unwrap();
        let text = String::from_utf8(drain_to_vec(&rx).unwrap()).unwrap();
        let (_, body) = text.split_once("\r\n\r\n").unwrap();
        assert_eq!(body, r#"{"jsonrpc":"2.0","id":"abc","result":[]}"#);
    }

    #[test]
    fn a_raw_result_and_a_typed_response_frame_the_same_way() {
        // The wire shape must not depend on which variant carried the answer,
        // or a client would see two different response encodings.
        let (out, rx) = pair();
        let value: serde_json::Value = serde_json::from_str(r#"[{"name":"x"}]"#).unwrap();
        out.send(Message::Response(lsp_server::Response {
            id: RequestId::from(7i32),
            result: Some(value),
            error: None,
        }))
        .unwrap();
        let typed = String::from_utf8(drain_to_vec(&rx).unwrap()).unwrap();
        out.send(Outgoing::RawResult {
            id: RequestId::from(7i32),
            result: Arc::from(r#"[{"name":"x"}]"#),
        })
        .unwrap();
        let raw = String::from_utf8(drain_to_vec(&rx).unwrap()).unwrap();
        assert_eq!(typed, raw);
    }

    #[test]
    fn the_writer_thread_ends_when_the_handle_drops() {
        let (out, thread) = Outbound::to_writer(Vec::new());
        out.send(Message::Notification(lsp_server::Notification {
            method: "x".to_owned(),
            params: serde_json::Value::Null,
        }))
        .unwrap();
        drop(out);
        thread
            .join()
            .expect("writer thread joins")
            .expect("no io error");
    }

    #[test]
    fn the_message_backend_materializes_a_raw_result() {
        let (tx, rx) = crossbeam_channel::unbounded::<Message>();
        let (out, thread) = Outbound::to_messages(tx);
        out.send(Outgoing::RawResult {
            id: RequestId::from(3i32),
            result: Arc::from(r#"[{"name":"x"}]"#),
        })
        .unwrap();
        let msg = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the response arrives as a Message");
        let Message::Response(resp) = msg else {
            panic!("expected a response");
        };
        assert_eq!(resp.id, RequestId::from(3i32));
        assert_eq!(
            resp.result.unwrap(),
            serde_json::json!([{ "name": "x" }]),
            "a channel host sees the same result a stream client would parse"
        );
        drop(out);
        thread.join().unwrap();
    }
}
