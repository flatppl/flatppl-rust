use flatppl_host::Context;
use serde_json::{Value, json};

const MODEL: &str = "# ünïcode comment
mu = elementof(interval(0.0, inf))
n = external(posintegers)
theta ~ Normal(mu = mu, sigma = 1.0)
v = elementof(cartpow(reals, [2, 3]))
p = elementof(cartprod(a = reals, b = stdsimplex(3)))
z = elementof(cartpow(reals, n))
_t = 2.0 * mu
y = _t + 1.0
inputs = (mu, v)
outputs = y + sum(v)
";

fn bindings(source: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let module = Context::default().parse(source, None, None)?;
    let rows = serde_json::to_value(module.bindings())?;
    Ok(Value::Object(
        rows.as_array()
            .ok_or("bindings are a list")?
            .iter()
            .map(|row| {
                (
                    row["name"].as_str().unwrap_or_default().to_string(),
                    row.clone(),
                )
            })
            .collect(),
    ))
}

#[test]
fn bindings_report_kind_domain_shape_size_and_position() -> Result<(), Box<dyn std::error::Error>> {
    let rows = bindings(MODEL)?;
    let pick = |name: &str| {
        let row = &rows[name];
        json!([
            row["kind"],
            row["domain"],
            row["shape"],
            row["size"],
            row["source"],
            row["line"],
            row["column"]
        ])
    };
    assert_eq!(
        pick("mu"),
        json!([
            "parameter", {"set": "interval", "lo": 0.0, "hi": "inf"}, [], 1,
            "elementof(interval(0.0, inf))", 2, 6
        ])
    );
    assert_eq!(
        pick("n"),
        json!(["observation", {"set": "posintegers"}, [], 1, "external(posintegers)", 3, 5])
    );
    assert_eq!(rows["theta"]["kind"], "draw");
    assert_eq!(
        pick("v"),
        json!([
            "parameter", {"set": "cartpow", "of": {"set": "reals"}, "shape": [2, 3]}, [2, 3], 6,
            "elementof(cartpow(reals, [2, 3]))", 5, 5
        ])
    );
    assert_eq!(
        rows["p"]["domain"],
        json!({"set": "record", "fields": [
            {"name": "a", "set": {"set": "reals"}},
            {"name": "b", "set": {"set": "stdsimplex", "n": 3}}
        ]})
    );
    assert_eq!(rows["p"]["shape"], Value::Null);
    assert_eq!(rows["p"]["size"], 4);
    // A size bound later by `set` is a dynamic extent until then.
    assert_eq!(rows["z"]["domain"]["shape"], json!([null]));
    assert_eq!(rows["z"]["size"], Value::Null);
    assert_eq!(rows["y"]["kind"], "derived");
    assert!(rows.get("_t").is_none());
    Ok(())
}

#[test]
fn dependencies_follow_private_bindings_and_module_aliases()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse("m = 1.0\nscale = 2.0", None, None)?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        "lib = load_module(\"model.flatppl\")\nx = elementof(reals)\n_h = x * lib.scale\ny = _h + x\nunrelated = 3.0",
        None,
        None,
    )?;
    let edges: Vec<_> = query
        .bindings()
        .into_iter()
        .map(|b| (b.name, b.dependencies))
        .collect();
    assert_eq!(
        edges,
        [
            ("lib".to_string(), vec![]),
            ("x".into(), vec![]),
            ("y".into(), vec!["lib".to_string(), "x".into()]),
            ("unrelated".into(), vec![]),
        ]
    );
    Ok(())
}

#[test]
fn bound_values_keep_their_binding_without_a_position() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let module = context.parse(
        "n = external(posintegers)\nz = elementof(cartpow(reals, n))",
        None,
        None,
    )?;
    let bound = context.set(&module, &[("n".into(), flatppl_host::Constant::Integer(3))])?;
    let rows = serde_json::to_value(bound.bindings())?;
    assert_eq!(rows[0]["kind"], "derived");
    assert_eq!(rows[0]["source"], Value::Null);
    assert_eq!(rows[0]["line"], Value::Null);
    assert_eq!(rows[1]["size"], 3);
    Ok(())
}
