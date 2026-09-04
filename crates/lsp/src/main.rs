use flatppl_lsp::outbound::Outbound;
use flatppl_lsp::server::{Wire, handshake, run_on};

fn main() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let (connection, io_threads) = lsp_server::Connection::stdio();
    // This process writes to stdout itself, so `lsp_server`'s writer thread
    // must never get a message: framing is a header write then a body write,
    // and two writers on one descriptor interleave mid-message. Dropping the
    // sender also lets that thread exit, so `io_threads.join()` returns.
    let lsp_server::Connection { sender, receiver } = connection;
    drop(sender);
    let (out, writer) = Outbound::to_writer(std::io::stdout());
    let wire = Wire { receiver, out };

    let server_caps = serde_json::to_value(flatppl_lsp::server::server_capabilities())?;
    let init_params = handshake(&wire, server_caps)?;

    run_on(wire, init_params)?;

    // `run_on` has dropped the last `Outbound`, so the writer drains and ends.
    writer.join().map_err(|_| "writer thread panicked")??;
    io_threads.join()?;
    Ok(())
}
