fn main() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let (connection, io_threads) = lsp_server::Connection::stdio();

    let server_caps = serde_json::to_value(flatppl_lsp::server::server_capabilities())?;
    let init_params = connection.initialize(server_caps)?;

    flatppl_lsp::server::run(connection, init_params)?;

    io_threads.join()?;
    Ok(())
}
