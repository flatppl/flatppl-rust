//! The stdio server end to end, as a real subprocess.
//!
//! The in-process tests drive [`flatppl_lsp::server::run`], which forwards over
//! a channel. The shipped binary instead answers the handshake itself and owns
//! the writer so a `documentSymbol` body can go out as pre-serialised JSON, so
//! that path needs its own coverage: the `jsonrpc` member, the
//! `Content-Length` framing, and the shutdown/exit exchange are all only
//! exercised here.

use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

/// A scratch directory unique to this test, removed on drop.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "flatppl-lsp-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// How long any single wait may block. Long enough for a debounce and a
/// contended machine, short enough that a dropped response fails the test
/// instead of hanging it.
const WAIT: Duration = Duration::from_secs(60);

struct Server {
    child: Child,
    stdin: ChildStdin,
    /// Parsed messages in arrival order, from the reader thread.
    inbox: Receiver<serde_json::Value>,
    /// Responses that arrived before the test asked for them.
    ///
    /// The worker pool answers concurrent requests in either order, so a test
    /// waiting on the earlier id may see the later one first. Holding it here
    /// is what keeps that from being a lost message.
    held: HashMap<i64, serde_json::Value>,
    next_id: i32,
}

impl Server {
    fn start(root: &std::path::Path) -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_flatppl-lsp"))
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("the server binary starts");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let (tx, inbox) = std::sync::mpsc::channel();
        // A reader thread, so a wait is bounded: a blocking `read_exact` on a
        // dropped response would hang the test with no output.
        std::thread::spawn(move || read_messages(stdout, tx));
        Server {
            child,
            stdin,
            inbox,
            held: HashMap::new(),
            next_id: 0,
        }
    }

    fn send(&mut self, msg: serde_json::Value) {
        let body = serde_json::to_string(&msg).unwrap();
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len()).unwrap();
        self.stdin.write_all(body.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: serde_json::Value) -> i32 {
        self.next_id += 1;
        self.send(serde_json::json!({
            "jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params
        }));
        self.next_id
    }

    fn notify(&mut self, method: &str, params: serde_json::Value) {
        self.send(serde_json::json!({
            "jsonrpc": "2.0", "method": method, "params": params
        }));
    }

    /// The response to `id`, holding on to any other response that arrives
    /// first and skipping notifications.
    fn wait(&mut self, id: i32) -> serde_json::Value {
        let id = id as i64;
        if let Some(msg) = self.held.remove(&id) {
            return msg;
        }
        let deadline = std::time::Instant::now() + WAIT;
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            let msg = self
                .inbox
                .recv_timeout(left)
                .unwrap_or_else(|e| panic!("no response for request {id}: {e}"));
            match msg.get("id").and_then(|v| v.as_i64()) {
                Some(got) if got == id => return msg,
                Some(got) => {
                    self.held.insert(got, msg);
                }
                None => {} // a notification
            }
        }
    }
}

/// Frame-decode the server's stdout, sending each message on `tx`.
///
/// Ends at end of stream or on the first malformed frame; the test then fails
/// on its next wait rather than hanging.
fn read_messages(mut stdout: BufReader<ChildStdout>, tx: Sender<serde_json::Value>) {
    loop {
        let mut header = Vec::new();
        while !header.ends_with(b"\r\n\r\n") {
            let mut byte = [0u8; 1];
            if stdout.read_exact(&mut byte).is_err() {
                return; // the server closed the stream
            }
            header.push(byte[0]);
        }
        let Ok(header) = String::from_utf8(header) else {
            return;
        };
        let Some(len) = header
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: "))
            .and_then(|l| l.trim().parse::<usize>().ok())
        else {
            return; // no Content-Length: the framing is broken
        };
        let mut body = vec![0u8; len];
        if stdout.read_exact(&mut body).is_err() {
            return;
        }
        let Ok(msg) = serde_json::from_slice(&body) else {
            return; // the body is not JSON, or is shorter than the header said
        };
        if tx.send(msg).is_err() {
            return; // the test finished
        }
    }
}

fn initialize(server: &mut Server, root: &std::path::Path) {
    let uri = format!("file://{}", root.display());
    let id = server.request(
        "initialize",
        serde_json::json!({
            "processId": null,
            "rootUri": uri,
            "capabilities": {},
        }),
    );
    let resp = server.wait(id);
    assert_eq!(
        resp["jsonrpc"], "2.0",
        "the handshake reply is JSON-RPC 2.0"
    );
    assert!(
        resp["result"]["capabilities"]["documentSymbolProvider"].is_boolean(),
        "the handshake advertises documentSymbol: {resp}"
    );
    server.notify("initialized", serde_json::json!({}));
}

#[test]
fn document_symbols_over_stdio_carry_the_jsonrpc_member() {
    let dir = Scratch::new("stdio");
    let path = dir.path().join("m.flatppl");
    std::fs::write(&path, "x = 1.5\ny = 2.5\n").unwrap();
    let uri = format!("file://{}", path.display());

    let mut server = Server::start(dir.path());
    initialize(&mut server, dir.path());
    server.notify(
        "textDocument/didOpen",
        serde_json::json!({
            "textDocument": {
                "uri": uri, "languageId": "flatppl", "version": 1,
                "text": "x = 1.5\ny = 2.5\n",
            }
        }),
    );

    let id = server.request(
        "textDocument/documentSymbol",
        serde_json::json!({ "textDocument": { "uri": uri } }),
    );
    let resp = server.wait(id);
    assert_eq!(
        resp["jsonrpc"], "2.0",
        "a pre-serialised body still carries the envelope: {resp}"
    );
    assert!(resp.get("error").is_none(), "no error: {resp}");
    let syms = resp["result"].as_array().expect("an array of symbols");
    let names: Vec<&str> = syms.iter().map(|s| s["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["x", "y"], "both bindings, in order");
    assert!(
        syms[0]["range"]["start"]["line"].is_number()
            && syms[0]["selectionRange"]["end"]["character"].is_number(),
        "the ranges survive the raw path: {}",
        syms[0]
    );
    assert_eq!(
        syms[0]["kind"], 13,
        "SymbolKind::VARIABLE is 13 in LSP 3.17"
    );

    // Two requests against one revision are answered from one shared payload;
    // both must still be complete responses with their own ids.
    let a = server.request(
        "textDocument/documentSymbol",
        serde_json::json!({ "textDocument": { "uri": uri } }),
    );
    let b = server.request(
        "textDocument/documentSymbol",
        serde_json::json!({ "textDocument": { "uri": uri } }),
    );
    let ra = server.wait(a);
    let rb = server.wait(b);
    assert_eq!(ra["result"], rb["result"], "one revision, one answer");
    assert_ne!(ra["id"], rb["id"], "each request keeps its own id");

    let id = server.request("shutdown", serde_json::json!(null));
    let resp = server.wait(id);
    assert!(resp["result"].is_null(), "shutdown answers null: {resp}");
    server.notify("exit", serde_json::json!(null));
    let status = server.child.wait().expect("the server exits");
    assert!(status.success(), "a clean exit, got {status:?}");
}

#[test]
fn an_edit_over_stdio_changes_the_symbol_payload() {
    // The payload is memoized per revision, so an edit must invalidate it.
    let dir = Scratch::new("stdio");
    let path = dir.path().join("m.flatppl");
    std::fs::write(&path, "x = 1.5\n").unwrap();
    let uri = format!("file://{}", path.display());

    let mut server = Server::start(dir.path());
    initialize(&mut server, dir.path());
    server.notify(
        "textDocument/didOpen",
        serde_json::json!({
            "textDocument": {
                "uri": uri, "languageId": "flatppl", "version": 1, "text": "x = 1.5\n",
            }
        }),
    );
    let id = server.request(
        "textDocument/documentSymbol",
        serde_json::json!({ "textDocument": { "uri": uri } }),
    );
    let before = server.wait(id);
    assert_eq!(before["result"].as_array().unwrap().len(), 1);

    server.notify(
        "textDocument/didChange",
        serde_json::json!({
            "textDocument": { "uri": uri, "version": 2 },
            "contentChanges": [{ "text": "x = 1.5\nzz = 2.5\n" }],
        }),
    );
    // Retry: an edit can answer an in-flight request `ContentModified`.
    for attempt in 0..10 {
        let id = server.request(
            "textDocument/documentSymbol",
            serde_json::json!({ "textDocument": { "uri": uri } }),
        );
        let resp = server.wait(id);
        if let Some(syms) = resp["result"].as_array() {
            let names: Vec<&str> = syms.iter().map(|s| s["name"].as_str().unwrap()).collect();
            assert_eq!(names, vec!["x", "zz"], "the new revision is served");
            return;
        }
        assert!(attempt < 9, "never got a result: {resp}");
    }
}

#[test]
fn a_request_before_initialize_is_refused_not_dropped() {
    let dir = Scratch::new("stdio-preinit");
    let mut server = Server::start(dir.path());
    let id = server.request(
        "textDocument/documentSymbol",
        serde_json::json!({ "textDocument": { "uri": "file:///nope.flatppl" } }),
    );
    let resp = server.wait(id);
    assert_eq!(
        resp["error"]["code"], -32002,
        "ServerNotInitialized, not silence: {resp}"
    );
    let _ = server.child.kill();
}
