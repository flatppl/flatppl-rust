//! Record values (spec §03 "Records" and "Tables"). A record output becomes one
//! result per field, depth-first in declared field order, so each case compares
//! a record spelling against the same fields listed one by one.

use flatppl_stablehlo::{EmitOptions, Mode};

fn emit_mode(source: &str, mode: Mode) -> Result<String, String> {
    let mut model = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("must determinize");
    flatppl_stablehlo::emit(
        &model,
        mode,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .map_err(|error| error.msg)
}

fn emit(source: &str) -> String {
    emit_mode(source, Mode::LogDensity).expect("must emit")
}

fn signature(out: &str) -> &str {
    out.lines()
        .find(|line| line.contains("func.func"))
        .expect("a function")
        .trim()
}

#[test]
fn a_nested_record_output_is_one_result_per_leaf_in_declared_order() {
    let model = "a = elementof(reals)\nb = elementof(cartpow(reals, 3))\n\
                 r = record(u = a * 2.0, v = record(p = b, q = a + 1.0))\ninputs = (a, b)\n";
    let out = emit(&format!("{model}outputs = r\n"));
    assert_eq!(
        out,
        emit(&format!("{model}outputs = (r.u, r.v.p, r.v.q)\n"))
    );
    assert_eq!(
        signature(&out),
        "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> \
         (tensor<f32>, tensor<3xf32>, tensor<f32>) {"
    );
}

#[test]
fn a_field_projection_lowers_no_sibling_field() {
    let model = "a = elementof(reals)\nr = record(u = exp(a), v = record(p = log(a), q = a))\n\
                 inputs = (a)\n";
    let out = emit(&format!("{model}outputs = r.v.p\n"));
    assert!(out.contains("stablehlo.log"), "{out}");
    assert!(!out.contains("stablehlo.exponential"), "{out}");
}

#[test]
fn a_table_is_the_record_of_its_columns() {
    let model = "a = elementof(cartpow(reals, 3))\nt = table(x = a, y = a .* 2.0)\ninputs = (a)\n";
    assert_eq!(
        emit(&format!("{model}outputs = t\n")),
        emit(&format!("{model}outputs = (t.x, t.y)\n"))
    );
    // §03 "Tables": `record(t)` and `table(r)` convert without changing a value.
    assert_eq!(
        emit(&format!("{model}outputs = table(record(t))\n")),
        emit(&format!("{model}outputs = (t.x, t.y)\n"))
    );
}

#[test]
fn a_table_row_is_a_record_of_column_elements() {
    let model = "data = elementof(cartpow(cartprod(x = reals, y = cartpow(reals, 2)), 4))\n";
    assert_eq!(
        emit(&format!("{model}inputs = (data)\noutputs = data[2]\n")),
        emit(&format!(
            "{model}inputs = (data)\noutputs = (data.x[2], data.y[2])\n"
        ))
    );
    // A runtime row index leaves the row's own type unresolved; its fields
    // still come from the table.
    let model = format!("{model}i = elementof(posintegers)\ninputs = (data, i)\n");
    assert_eq!(
        emit(&format!("{model}outputs = data[i]\n")),
        emit(&format!("{model}outputs = (data.x[i], data.y[i])\n"))
    );
}

#[test]
fn ifelse_selects_each_field() {
    let model = "a = elementof(reals)\nr1 = record(u = a, v = a * 2.0)\n\
                 r2 = record(u = a + 1.0, v = 0.0)\nc = a > 0.0\ninputs = (a)\n";
    assert_eq!(
        emit(&format!("{model}outputs = ifelse(c, r1, r2)\n")),
        emit(&format!(
            "{model}outputs = (ifelse(c, r1.u, r2.u), ifelse(c, r1.v, r2.v))\n"
        ))
    );
}

/// §03 "Tables": broadcasting a function that returns a record gives a table.
#[test]
fn a_record_valued_broadcast_is_a_table() {
    let model = "xs = elementof(cartpow(reals, 3))\nf = x -> record(a = x, b = x * 2.0)\n\
                 t = f.(xs)\ninputs = (xs)\n";
    let out = emit(&format!("{model}outputs = (t, sum(t))\n"));
    assert_eq!(
        signature(&out),
        "func.func @logdensity(%arg0: tensor<3xf32>) -> \
         (tensor<3xf32>, tensor<3xf32>, tensor<f32>, tensor<f32>) {"
    );
    assert_eq!(out.matches("stablehlo.multiply").count(), 1, "{out}");
}

#[test]
fn a_nested_record_input_flattens_depth_first() {
    let out = emit(
        "p = elementof(cartprod(x = reals, inner = cartprod(u = reals, v = cartpow(reals, 2))))\n\
         inputs = (p)\noutputs = (p.inner.v, p.x)\n",
    );
    assert_eq!(
        signature(&out),
        "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<2xf32>) -> \
         (tensor<2xf32>, tensor<f32>) {"
    );
    assert!(out.contains("return %arg2, %arg0"), "{out}");
}

/// A record output in `@sample` that is not a literal `record(...)` flattens the
/// same way: here a table reduction over sampled values.
#[test]
fn a_computed_record_sample_output_is_one_result_per_field() {
    let model = "mu ~ Normal(0, 1)\ny ~ Normal(mu, 1)\n\
                 value, next = rand(lawof(record(mu = mu, y = y)), rnginit(1))\n\
                 tab = table(m = [value.mu, value.y], n = [value.y, 2.0])\n";
    let record = emit_mode(
        &format!("{model}outputs = (sum(tab), next)\n"),
        Mode::Sample,
    )
    .unwrap();
    let fields = emit_mode(
        &format!("{model}outputs = (sum(tab.m), sum(tab.n), next)\n"),
        Mode::Sample,
    )
    .unwrap();
    assert_eq!(record, fields);
}

#[test]
fn a_record_in_tensor_position_refuses_with_the_reason() {
    let err = emit_mode(
        "data = elementof(cartpow(cartprod(x = reals), 4))\ninputs = (data)\n\
         outputs = get(sum(data), [\"x\"])\n",
        Mode::LogDensity,
    )
    .unwrap_err();
    assert!(
        err.contains("a table reduction is a record, which has no single tensor form"),
        "{err}"
    );
}

/// A scan state may hold a nested record: its leaves are carried and collected
/// one by one.
#[test]
fn a_nested_record_scan_state_carries_each_leaf() {
    let model = "xs = elementof(cartpow(reals, 3))\n\
                 step = (st, x) -> record(a = st.a + x, b = record(c = st.b.c * x))\n\
                 res = scan(step, record(a = 0.0, b = record(c = 1.0)), xs)\ninputs = (xs)\n";
    let out = emit(&format!("{model}outputs = res\n"));
    assert_eq!(out, emit(&format!("{model}outputs = (res.a, res.b.c)\n")));
    assert_eq!(out.matches("stablehlo.while").count(), 1, "{out}");
    let model = "xs = elementof(cartpow(reals, 3))\n\
                 step = (st, r) -> record(a = st.a + r.a, b = record(c = st.b.c * r.b.c))\n\
                 res = reduce(step, table(a = xs, b = table(c = xs .+ 1.0)))\ninputs = (xs)\n";
    assert_eq!(
        signature(&emit(&format!("{model}outputs = res\n"))),
        "func.func @logdensity(%arg0: tensor<3xf32>) -> (tensor<f32>, tensor<f32>) {"
    );
}
