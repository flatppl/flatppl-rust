use flatppl_host::{Context, EmitOptions};
use serde_json::json;

#[test]
fn sliced_arrays_retain_source_axes_in_query_outputs() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        x = elementof(cartpow(reals, [4, 3]))
        inputs = x
        rows = x[[1, 3], all]
        trailing = x[[1, 3]]
        column = get0(x, all, 1)
        nested = [[1.0, 2.0], [3.0, 4.0]][[2, 1]]
        reduced = aggregate(sum, [.col], rows[.row, .col])
        outputs = (rows, column, reduced)
        "#,
        None,
        None,
    )?;
    let bindings = query.bindings();
    assert_eq!(
        bindings
            .iter()
            .find(|b| b.name == "trailing")
            .unwrap()
            .value_type
            .as_deref(),
        Some("real[2, 3]")
    );
    assert_eq!(
        bindings
            .iter()
            .find(|b| b.name == "nested")
            .unwrap()
            .value_type
            .as_deref(),
        Some("real[2][2]")
    );
    let exported = serde_json::to_value(query.compile(&EmitOptions::default())?)?;
    let shapes: Vec<_> = exported["output"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["shape"].clone())
        .collect();
    assert_eq!(shapes, vec![json!([2, 3]), json!([4]), json!([3])]);
    Ok(())
}
