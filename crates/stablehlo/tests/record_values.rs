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
    // A runtime row index selects one row in the same way.
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

fn determinize(source: &str) -> flatppl_core::Module {
    let mut model = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("must determinize")
}

/// §03 "Tables" lets a column be a table, and §07 "Table reductions" then
/// reduces that column column-wise: `sum(data).z` is `sum(data.z)`. `infer`
/// leaves `sum(data)` `%deferred` over a table column, so this types the call
/// as §07 does and checks the emitter.
#[test]
fn a_table_reduction_recurses_into_a_table_column() {
    use flatppl_core::{CallHead, Idx, Node, NodeId, Type};
    let model = "data = elementof(cartpow(cartprod(x = reals, z = cartprod(u = reals, v = reals)), 3))\n\
                 inputs = (data)\n";
    let mut module = determinize(&format!("{model}outputs = sum(data)\n"));
    let reductions = (0..module.node_count())
        .map(NodeId::from_usize)
        .filter(|&id| {
            matches!(module.node(id), Node::Call(c)
                if matches!(c.head, CallHead::Builtin(h) if module.resolve(h) == "sum"))
        })
        .collect::<Vec<_>>();
    assert_eq!(reductions.len(), 1);
    let Node::Call(call) = module.node(reductions[0]) else {
        unreachable!()
    };
    let Some(Type::Table { columns, .. }) = module.type_of(call.args[0]).cloned() else {
        panic!("`sum` must reduce the table")
    };
    assert!(matches!(
        module.type_of(reductions[0]),
        Some(Type::Deferred)
    ));
    module.set_type(reductions[0], Type::Record(columns));
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        ..Default::default()
    };
    let out = flatppl_stablehlo::emit(&module, Mode::LogDensity, &options).expect("must emit");
    assert_eq!(
        out,
        emit(&format!(
            "{model}outputs = (sum(data.x), sum(data.z.u), sum(data.z.v))\n"
        ))
    );
}

fn emit_err(module: &flatppl_core::Module) -> String {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        ..Default::default()
    };
    flatppl_stablehlo::emit(module, Mode::LogDensity, &options)
        .expect_err("must refuse")
        .msg
}

/// Replace the `outputs` binding with `node`, typed `ty`.
fn set_output(module: &mut flatppl_core::Module, node: flatppl_core::Node, ty: flatppl_core::Type) {
    let outputs = module.intern("outputs");
    let binding = module.binding_by_name(outputs).expect("an outputs binding");
    let id = module.alloc(node);
    module.set_type(id, ty);
    module.set_binding_rhs(binding, id);
}

fn self_ref(module: &mut flatppl_core::Module, name: &str) -> flatppl_core::NodeId {
    let name = module.intern(name);
    module.alloc(flatppl_core::Node::Ref(flatppl_core::Ref {
        ns: flatppl_core::RefNs::SelfMod,
        name,
    }))
}

fn builtin(
    module: &mut flatppl_core::Module,
    head: &str,
    args: Vec<flatppl_core::NodeId>,
) -> flatppl_core::Node {
    flatppl_core::Node::Call(flatppl_core::Call {
        head: flatppl_core::CallHead::Builtin(module.intern(head)),
        args: args.into_boxed_slice(),
        named: Box::new([]),
        inputs: None,
    })
}

fn scalar() -> flatppl_core::Type {
    flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real)
}

/// A scan is not among §07's column-wise table heads, and its domain is
/// "vectors", so it refuses over a table even when `infer`'s verdict is
/// bypassed.
#[test]
fn a_scan_over_a_table_refuses_on_the_vector_domain() {
    use flatppl_core::{Idx, NodeId, Type};
    for head in ["cumsum", "cumprod", "cummax", "cummin"] {
        let mut module = flatppl_syntax::parse(&format!(
            "data = elementof(cartpow(cartprod(x = reals), 4))\ninputs = (data)\noutputs = {head}(data)\n"
        ))
        .expect("parse");
        assert!(!flatppl_infer::infer(&mut module).is_empty());
        for id in (0..module.node_count()).map(NodeId::from_usize) {
            if matches!(module.type_of(id), Some(Type::Failed(_))) {
                module.set_type(id, Type::Deferred);
            }
        }
        let err = emit_err(&module);
        assert!(err.contains("and a table is not a vector"), "{head}: {err}");
    }
}

#[test]
fn a_missing_field_refuses() {
    let mut module =
        determinize("a = elementof(reals)\nr = record(u = a)\ninputs = (a)\noutputs = r\n");
    let r = self_ref(&mut module, "r");
    let selector = module.alloc(flatppl_core::Node::Lit(flatppl_core::Scalar::Str(
        "w".into(),
    )));
    let get = builtin(&mut module, "get", vec![r, selector]);
    set_output(&mut module, get, scalar());
    assert!(emit_err(&module).contains("the record has no field `w`"));
}

#[test]
fn a_record_typed_producer_without_a_record_form_refuses() {
    let mut module = determinize("a = elementof(reals)\ninputs = (a)\noutputs = a\n");
    let a = self_ref(&mut module, "a");
    let exp = builtin(&mut module, "exp", vec![a]);
    let p = module.intern("p");
    set_output(
        &mut module,
        exp,
        flatppl_core::Type::Record(Box::new([(p, scalar())])),
    );
    let err = emit_err(&module);
    assert!(
        err.contains("field `p` of a record-valued `exp` has no lowering"),
        "{err}"
    );
}

#[test]
fn ifelse_over_records_with_different_fields_refuses() {
    let mut module = determinize(
        "a = elementof(reals)\nr1 = record(u = record(p = a))\nr2 = record(u = record(q = a))\n\
         c = a > 0.0\ninputs = (a)\noutputs = (r1, r2, c)\n",
    );
    let (c, r1, r2) = (
        self_ref(&mut module, "c"),
        self_ref(&mut module, "r1"),
        self_ref(&mut module, "r2"),
    );
    let ifelse = builtin(&mut module, "ifelse", vec![c, r1, r2]);
    let (u, p) = (module.intern("u"), module.intern("p"));
    let inner = flatppl_core::Type::Record(Box::new([(p, scalar())]));
    set_output(
        &mut module,
        ifelse,
        flatppl_core::Type::Record(Box::new([(u, inner)])),
    );
    assert!(emit_err(&module).contains("the two branches are records with different fields"));
}

/// A batched numerical integral captures each batched leaf of a nested record
/// field, as it captures a flat one.
#[test]
fn a_batched_integral_captures_nested_record_leaves() {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        integration: Some(flatppl_stablehlo::IntegrationOptions::default()),
        ..Default::default()
    };
    let emit = |source: &str| {
        let mut model = flatppl_syntax::parse(source).expect("parse");
        assert!(flatppl_infer::infer(&mut model).is_empty());
        let roots = [model.intern("inputs"), model.intern("outputs")];
        let model = flatppl_determinizer::determinize_with_options(
            &model,
            &flatppl_infer::ModuleBundle::new(),
            Some(&roots),
            &options.lowering_options(),
        )
        .expect("must determinize");
        flatppl_stablehlo::emit(&model, Mode::LogDensity, &options).expect("must emit")
    };
    let query = |row: &str, scale: &str| {
        format!(
            "data = elementof(cartpow(cartprod(x = posreals, {row}), 3))\n\
             f = r -> logdensityof(normalize(truncate(weighted(x -> 1.0/x, \
             Pareto(3.0, {scale})), interval(0.0, 10.0))), r.x)\n\
             inputs = (data)\noutputs = f.(data)\n"
        )
    };
    assert_eq!(
        emit(&query("z = cartprod(s = posreals)", "r.z.s")),
        emit(&query("s = posreals", "r.s"))
    );
}
