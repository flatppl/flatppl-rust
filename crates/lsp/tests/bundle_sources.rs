use lsp_server::{Connection, Message, Notification, Request, Response};
use serde_json::{Value, json};
use std::time::Duration;

fn notify(client: &Connection, method: &str, params: Value) {
    client
        .sender
        .send(Message::Notification(Notification::new(
            method.to_owned(),
            params,
        )))
        .unwrap();
}

fn request(client: &Connection, method: &str, params: Value) -> Response {
    let id = lsp_server::RequestId::from(1);
    client
        .sender
        .send(Message::Request(Request::new(
            id.clone(),
            method.to_owned(),
            params,
        )))
        .unwrap();
    loop {
        if let Message::Response(response) = client
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            && response.id == id
        {
            return response;
        }
    }
}

fn position(uri: &str, line: u32, character: u32) -> Value {
    json!({ "textDocument": { "uri": uri }, "position": { "line": line, "character": character } })
}

fn hover(client: &Connection, uri: &str, line: u32, character: u32) -> String {
    request(client, "textDocument/hover", position(uri, line, character))
        .response_result
        .unwrap()
        .to_string()
}

#[test]
fn mounted_archives_preserve_identity_navigation_and_readonly_lifecycle() {
    let (client, server) = Connection::memory();
    let thread = std::thread::spawn(move || {
        flatppl_lsp::server::run(server, json!({ "capabilities": {} })).unwrap();
    });
    let root = "/workspace/model.flatppl";
    let first = "flatppl-zip:one.zip!/main.flatppl";
    let second = "flatppl-zip:two.zip!/main.flatppl";
    let nested = "flatppl-zip:flatppl-zip%3Aone.zip%21%2Fnested.zip!/main.flatppl";
    let root_uri = "file:///workspace/model.flatppl";
    let first_uri = "flatppl-bundle:/one/main.flatppl";
    let second_uri = "flatppl-bundle:/two/main.flatppl";
    let nested_uri = "flatppl-bundle:/one/nested/main.flatppl";
    let mut graph = json!({
        "sources": {
            (root): "a = load_module(\"one.zip\")\nb = load_module(\"two.zip\")\nx = a.value\ny = b.value\n",
            (first): "path = \"nested.zip\"\nh = load_module(path)\nvalue = h.theta\n",
            (second): "value = elementof(integers)\n",
            (nested): "theta = elementof(reals)\n"
        },
        "resolutions": {
            (root): { "modules": { "one.zip": first, "two.zip": second } },
            (first): { "modules": { "nested.zip": nested }, "data": { "data.json": "opaque-data-id" } }
        },
        "uris": { (root): root_uri, (first): first_uri, (second): second_uri, (nested): nested_uri }
    });
    notify(&client, "flatppl/bundleSources", graph.clone());
    for _ in 0..4 {
        let Message::Notification(note) = client
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
        else {
            panic!("expected diagnostics")
        };
        assert_eq!(note.method, "textDocument/publishDiagnostics");
        assert_eq!(note.params["diagnostics"], json!([]));
    }
    let real_hover = hover(&client, root_uri, 2, 6);
    assert!(real_hover.to_lowercase().contains("real"), "{real_hover}");
    assert!(
        hover(&client, root_uri, 3, 6)
            .to_lowercase()
            .contains("int")
    );
    let definition = request(
        &client,
        "textDocument/definition",
        position(first_uri, 2, 10),
    );
    assert_eq!(definition.response_result.unwrap()["uri"], nested_uri);

    let mut references = position(nested_uri, 0, 0);
    references["context"] = json!({ "includeDeclaration": true });
    let references = request(&client, "textDocument/references", references)
        .response_result
        .unwrap();
    let uris: Vec<&str> = references
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["uri"].as_str().unwrap())
        .collect();
    assert!(uris.contains(&first_uri));
    assert!(uris.contains(&nested_uri));
    let symbols = request(&client, "workspace/symbol", json!({"query": "theta"}))
        .response_result
        .unwrap();
    assert_eq!(symbols[0]["location"]["uri"], nested_uri);

    assert_eq!(
        request(
            &client,
            "textDocument/prepareRename",
            position(nested_uri, 0, 0)
        )
        .response_result
        .ok(),
        Some(Value::Null)
    );
    assert_eq!(
        request(
            &client,
            "textDocument/prepareRename",
            position(root_uri, 2, 6)
        )
        .response_result
        .ok(),
        Some(Value::Null)
    );
    let mut rename = position(root_uri, 2, 6);
    rename["newName"] = json!("renamed");
    assert!(
        request(&client, "textDocument/rename", rename)
            .response_result
            .is_err()
    );
    let mut rename = position(root_uri, 0, 0);
    rename["newName"] = json!("renamed");
    let renamed = request(&client, "textDocument/rename", rename)
        .response_result
        .unwrap();
    assert_eq!(renamed["changes"].as_object().unwrap().len(), 1);
    assert!(renamed["changes"][root_uri].is_array());

    notify(
        &client,
        "textDocument/didOpen",
        json!({ "textDocument": {
        "uri": nested_uri, "languageId": "flatppl", "version": 1, "text": "theta = elementof(reals)\n"
    } }),
    );
    notify(
        &client,
        "textDocument/didClose",
        json!({ "textDocument": { "uri": nested_uri } }),
    );
    assert!(
        hover(&client, root_uri, 2, 6)
            .to_lowercase()
            .contains("real")
    );
    graph["sources"][nested] = json!("theta = elementof(booleans)\n");
    notify(&client, "flatppl/bundleSources", graph.clone());
    assert!(
        hover(&client, root_uri, 2, 6)
            .to_lowercase()
            .contains("bool")
    );
    notify(
        &client,
        "textDocument/didChange",
        json!({
            "textDocument": { "uri": nested_uri, "version": 2 },
            "contentChanges": [{ "text": "theta = broken(" }]
        }),
    );
    assert!(
        hover(&client, root_uri, 2, 6)
            .to_lowercase()
            .contains("bool")
    );

    graph["resolutions"][root]["modules"]["one.zip"] = json!(second);
    notify(&client, "flatppl/bundleSources", graph.clone());
    assert!(
        hover(&client, root_uri, 2, 6)
            .to_lowercase()
            .contains("int")
    );

    notify(
        &client,
        "textDocument/didOpen",
        json!({ "textDocument": {
        "uri": root_uri, "languageId": "flatppl", "version": 1, "text": "x = elementof(booleans)\n"
    } }),
    );
    notify(&client, "flatppl/bundleSources", graph);
    assert!(
        hover(&client, root_uri, 0, 4)
            .to_lowercase()
            .contains("bool")
    );

    request(&client, "shutdown", Value::Null);
    notify(&client, "exit", Value::Null);
    thread.join().unwrap();
}

#[test]
fn archive_sources_do_not_resolve_missing_edges_outside_the_mount() {
    let (client, server) = Connection::memory();
    let thread = std::thread::spawn(move || {
        flatppl_lsp::server::run(server, json!({ "capabilities": {} })).unwrap();
    });
    let archive = "flatppl-zip:model.zip!/main.flatppl";
    let uri = "flatppl-bundle:/model/main.flatppl";
    notify(
        &client,
        "flatppl/bundleSources",
        json!({
            "sources": {
                (archive): "h = load_module(\"helper.flatppl\")\nx = h.value\n",
                "helper.flatppl": "value = elementof(reals)\n"
            },
            "uris": { (archive): uri, "helper.flatppl": "file:///workspace/helper.flatppl" },
            "resolutions": {}
        }),
    );
    loop {
        let Message::Notification(note) = client
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
        else {
            continue;
        };
        if note.method == "textDocument/publishDiagnostics" && note.params["uri"] == uri {
            assert!(
                note.params["diagnostics"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|d| d["severity"] == 1)
            );
            break;
        }
    }
    assert_eq!(
        request(&client, "textDocument/definition", position(uri, 1, 6))
            .response_result
            .ok(),
        Some(Value::Null)
    );
    request(&client, "shutdown", Value::Null);
    notify(&client, "exit", Value::Null);
    thread.join().unwrap();
}
