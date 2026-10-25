use flatppl_host::{Context, EmitOptions, MathFormat};
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
fn ordered_and_zerosum_domains_report_their_length() -> Result<(), Box<dyn std::error::Error>> {
    let rows = bindings("o = elementof(ordered(3))\nz = elementof(zerosum(4))\n")?;
    assert_eq!(rows["o"]["domain"], json!({"set": "ordered", "n": 3}));
    assert_eq!(rows["o"]["shape"], json!([3]));
    assert_eq!(rows["z"]["domain"], json!({"set": "zerosum", "n": 4}));
    // Four stored values, three free coordinates: the size is the stored length.
    assert_eq!(rows["z"]["size"], 4);
    Ok(())
}

#[test]
fn renamed_kernel_inputs_report_the_bound_values_support() -> Result<(), Box<dyn std::error::Error>>
{
    let rows = bindings(
        "s ~ Exponential(rate = 1.0)
y ~ Normal(mu = 0.0, sigma = s)
K = kernelof(record(y = y), a = s)
L = likelihoodof(K, record(y = 0.5))
",
    )?;
    let inputs = json!([{"name": "a", "domain": {"set": "nonnegreals"}}]);
    assert_eq!(rows["K"]["inputs"], inputs);
    assert_eq!(rows["L"]["inputs"], inputs);
    assert_eq!(rows["s"]["inputs"], Value::Null);
    Ok(())
}

#[test]
fn record_kernel_inputs_report_the_record_domain() -> Result<(), Box<dyn std::error::Error>> {
    let rows = bindings(
        "p = elementof(cartprod(a = reals, b = posreals))
K = functionof(Normal(mu = p.a, sigma = p.b), p)
",
    )?;
    assert_eq!(
        rows["K"]["inputs"],
        json!([{"name": "p", "domain": {"set": "record", "fields": [
            {"name": "a", "set": {"set": "reals"}},
            {"name": "b", "set": {"set": "posreals"}},
        ]}}])
    );
    Ok(())
}

#[test]
fn pyhf_likelihood_inputs_are_its_traced_parameters() -> Result<(), Box<dyn std::error::Error>> {
    let source = serde_json::json!({
        "channels": [{"name": "c", "samples": [
            {"name": "s", "data": [5.0, 10.0],
             "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]},
            {"name": "b", "data": [50.0, 60.0],
             "modifiers": [{"name": "gamma", "type": "shapesys", "data": [5.0, 6.0]}]}]}],
        "observations": [{"name": "c", "data": [52.0, 63.0]}],
        "measurements": [{"name": "m", "config": {"poi": "mu"}}],
        "version": "1.0.0"
    })
    .to_string();
    let mut context = Context::default();
    let model = context.import_pyhf(&source, &flatppl_host::PyhfOptions::default(), None, None)?;
    let rows = serde_json::to_value(model.bindings())?;
    let likelihood = rows
        .as_array()
        .ok_or("bindings are a list")?
        .iter()
        .find(|row| row["name"] == "likelihood")
        .ok_or("no likelihood")?;
    let names: Vec<_> = likelihood["inputs"]
        .as_array()
        .ok_or("inputs are a list")?
        .iter()
        .map(|input| input["name"].as_str().unwrap_or_default())
        .collect();
    // The likelihood's own interface order, as its type shows it.
    assert_eq!(
        likelihood["value_type"],
        "likelihood(gamma, mu) over real[4]"
    );
    assert_eq!(names, ["gamma", "mu"]);
    assert_eq!(
        likelihood["inputs"][0]["domain"],
        json!({"set": "cartpow", "of": {"set": "posreals"}, "shape": [2]})
    );
    assert_eq!(likelihood["inputs"][1]["domain"], json!({"set": "reals"}));
    Ok(())
}

#[test]
fn measure_domains_report_their_support() -> Result<(), Box<dyn std::error::Error>> {
    let rows = bindings(
        "theta ~ Normal(mu = 0.0, sigma = 1.0)
s ~ Exponential(rate = 1.0)
y ~ Normal(mu = theta, sigma = s)
prior = lawof(record(theta = theta, s = s))
L = likelihoodof(kernelof(record(y = y), theta = theta, s = s), record(y = 1.0))
posterior = bayesupdate(L, prior)
",
    )?;
    let support = json!({"set": "record", "fields": [
        {"name": "theta", "set": {"set": "reals"}},
        {"name": "s", "set": {"set": "nonnegreals"}},
    ]});
    assert_eq!(rows["s"]["domain"], json!({"set": "nonnegreals"}));
    assert_eq!(rows["prior"]["domain"], support);
    assert_eq!(rows["posterior"]["domain"], support);
    assert_eq!(rows["L"]["domain"], json!({"set": "unknown"}));
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

#[test]
fn slices_keep_what_the_outputs_reach_and_their_imports() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    let model = context.parse("scale = 2.0", None, None)?;
    context.register("model.flatppl", &model)?;
    let module = context.parse(
        "lib = load_module(\"model.flatppl\")\nn = external(posintegers)\nx = elementof(reals)\n_h = x * lib.scale\nunused = elementof(reals)\ninputs = x\noutputs = _h + 1.0",
        None,
        None,
    )?;
    let bound = context.set(&module, &[("n".into(), flatppl_host::Constant::Integer(2))])?;
    let slice = context.slice(&bound, &["inputs", "outputs"])?;
    let names: Vec<_> = slice.bindings().into_iter().map(|b| b.name).collect();
    assert_eq!(names, ["lib", "x", "inputs", "outputs"]);
    // Positions index the printed slice.
    let rows = slice.bindings();
    let (start, end) = rows[1].span.ok_or("x has a span")?;
    assert_eq!(
        &slice.source()[start as usize..end as usize],
        "elementof(reals)"
    );
    let full = module.compile(&EmitOptions::default())?;
    let sliced = slice.compile(&EmitOptions::default())?;
    assert_eq!(full.stablehlo, sliced.stablehlo);
    assert!(context.slice(&module, &["absent"]).is_err());
    Ok(())
}

#[test]
fn math_renders_each_binding_in_the_requested_notation() -> Result<(), Box<dyn std::error::Error>> {
    let module = Context::default().parse(
        "theta ~ Normal(mu = 0.0, sigma = 1.0)\ny = 2.0 * theta",
        None,
        None,
    )?;
    let tex = module.math(MathFormat::Tex);
    let rows: Vec<_> = tex.iter().map(|r| (r.name.as_str(), r.kind)).collect();
    assert_eq!(rows, [("theta", "draw"), ("y", "value")]);
    assert!(tex[0].text.contains(r"\sim \mathcal{N}"), "{}", tex[0].text);
    assert!(module.math(MathFormat::MathMl)[0].text.starts_with("<math"));
    assert!(module.math(MathFormat::Typst)[0].text.contains("∼ cal(N)"));
    Ok(())
}

#[test]
fn diagnostics_locate_parse_infer_and_emission_errors() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let mut located = |source: &str| -> Result<Value, Box<dyn std::error::Error>> {
        let error = match context.parse(source, None, None) {
            Err(error) => error,
            Ok(module) => module
                .compile(&EmitOptions::default())
                .err()
                .ok_or("the source should fail")?,
        };
        let error = serde_json::to_value(&error)?;
        Ok(json!([
            error["stage"],
            error["line"],
            error["column"],
            error["excerpt"]
        ]))
    };
    // Columns count characters, so the two-byte `é` counts once.
    assert_eq!(
        located("a = 1.0\nx = \"é\" = 1")?,
        json!(["parse", 2, 9, "x = \"é\" = 1"])
    );
    assert_eq!(
        located("a = 1.0\nb = a + missing")?,
        json!(["infer", 2, 9, "b = a + missing"])
    );
    assert_eq!(
        located("a = elementof(reals)\nx = 2.0 * a\ninputs = x\noutputs = x")?,
        json!(["stablehlo", 2, 5, "x = 2.0 * a"])
    );
    Ok(())
}

#[test]
fn import_errors_locate_in_the_imported_text() -> Result<(), Box<dyn std::error::Error>> {
    let directory =
        std::env::temp_dir().join(format!("flatppl-host-locate-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    std::fs::write(directory.join("child.flatppl"), "a = 1.0\nb = = 2")?;
    let host = directory.join("host.py");
    let error = Context::default()
        .parse(
            "m = load_module(\"child.flatppl\")",
            None,
            Some(host.to_str().ok_or("utf-8 path")?),
        )
        .err()
        .ok_or("the import should fail")?;
    std::fs::remove_dir_all(&directory)?;
    assert_eq!(error.stage, "parse");
    let position = error.position.ok_or("the error is located")?;
    assert_eq!((position.line, position.column), (2, 5));
    assert_eq!(&*position.excerpt, "b = = 2");
    assert_eq!(error.import_chain.len(), 1);
    Ok(())
}
