use flatppl_host::{Context, EmitOptions};
use serde_json::json;

#[test]
fn equivalent_selections_share_emitted_values() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        x = elementof(cartpow(reals, 4))
        a = x[[3, 1, 3]]
        b = [x[3], x[1], x[3]]
        c = x[[1, 2, 1, 2]]
        d = [x[1], x[2], x[1], x[2]]
        inputs = x
        outputs = (sum(a .* [2.0, -1.0, 4.0]) + sum(b .* [3.0, 5.0, -2.0]),
                   sum(c .* [1.0, 2.0, 3.0, 4.0]) + sum(d))
        "#,
        None,
        None,
    )?;
    let ir = query.compile(&EmitOptions::default())?.stablehlo;
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 1, "{ir}");
    assert_eq!(ir.matches("stablehlo.slice %arg0 [0:2]").count(), 1, "{ir}");
    Ok(())
}

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
