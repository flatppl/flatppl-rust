//! Smoke test: the Task-1 stub `emit` accepts a determinized (FlatPDL) module
//! and returns a minimal valid StableHLO module string. Later tasks replace
//! this with real golden output comparisons.
//!
//! Task 2 adds `mlir_type_of`: the `Type`/`Dim` → MLIR `tensor<…>` mapping.
//! These tests build tiny `Module`s by hand (`alloc` a placeholder node,
//! `set_type` the type under test) rather than parsing + inferring source,
//! since only the type side-table matters for this mapping.

use flatppl_core::{
    Binding, Call, CallHead, Dim, Mass, Module, Node, NodeId, Ref, RefNs, Scalar, ScalarType, Type,
};
use flatppl_stablehlo::{Dtype, Emitter, MlirTy, Value, mlir_type_of};

/// Every physical `{`/`(`/`[` in `s` has a matching close, and vice versa —
/// a cheap structural well-formedness check for hand-assembled MLIR text
/// (it does not parse the text, just counts bracket nesting).
fn is_delimiter_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0
}

#[test]
fn emit_stub_on_flatpdl_returns_module() {
    // A bare fixed-data binding with no density term (e.g. `x = 1.0`) is
    // exactly the shape `emit_logdensity`'s query-output guard now refuses
    // (see `crates/stablehlo/src/modes.rs`) — this smoke test needs a real
    // `logdensityof` so it still exercises the success path.
    let src = "flatppl_compat = \"0.1\"\na = draw(Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n";
    let m = flatppl_syntax::parse(src).unwrap();
    let d = flatppl_determinizer::determinize(&m).unwrap();
    let out = flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .unwrap();
    assert!(out.contains("module {"));
}

/// A placeholder node to hang a `set_type` on — its `Node` payload is
/// irrelevant to `mlir_type_of`, which only reads the type side-table.
fn placeholder(m: &mut Module, ty: Type) -> flatppl_core::NodeId {
    let id = m.alloc(Node::Lit(Scalar::Real(0.0)));
    m.set_type(id, ty);
    id
}

#[test]
fn mlir_type_of_scalar_renders_rank0_tensor() {
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::Scalar(ScalarType::Real));
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Scalar);
    assert_eq!(ty.render(Dtype::F32), "tensor<f32>");
}

#[test]
fn mlir_type_of_flat_array_renders_ranked_tensor() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Array {
            shape: Box::new([Dim::Static(2), Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<2x3xf32>");
}

#[test]
fn mlir_type_of_nested_array_flattens_to_one_tensor_shape() {
    // Array{shape:[2], elem:Array{shape:[3], elem:real}} — vec-of-vec (spec
    // §03 nesting) — must flatten to the SAME single tensor shape as the flat
    // rank-2 array above, not a nested/tuple MLIR type.
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Array {
            shape: Box::new([Dim::Static(2)]),
            elem: Box::new(Type::Array {
                shape: Box::new([Dim::Static(3)]),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            }),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<2x3xf32>");
}

#[test]
fn mlir_type_of_dynamic_dim_renders_question_mark() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Array {
            shape: Box::new([Dim::Dynamic, Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![None, Some(3)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<?x3xf32>");
}

#[test]
fn mlir_type_of_tvector_renders_ranked_tensor() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::TVector {
            len: Dim::Static(4),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(4)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<4xf32>");
}

#[test]
fn mlir_type_of_dtype_is_configurable_not_hardcoded() {
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::Scalar(ScalarType::Real));
    let ty = mlir_type_of(&m, id, Dtype::F64).unwrap();
    assert_eq!(ty.render(Dtype::F64), "tensor<f64>");
}

#[test]
fn mlir_type_of_refuses_aggregate_types() {
    let mut m = Module::new();
    let field = m.intern("x");
    let id = placeholder(
        &mut m,
        Type::Record(Box::new([(field, Type::Scalar(ScalarType::Real))])),
    );
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("aggregate"));
    assert!(err.msg.contains("destructured"));
    assert_eq!(err.node, Some(id));
}

#[test]
fn mlir_type_of_refuses_residual_measure_layer_types() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Measure {
            domain: Box::new(Type::Scalar(ScalarType::Real)),
            mass: Mass::Normalized,
        },
    );
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("residual measure-layer type in FlatPDL"));
    assert_eq!(err.node, Some(id));
}

#[test]
fn mlir_type_of_refuses_other_types_naming_the_type() {
    // `RngState` hits the catch-all arm (neither aggregate nor
    // measure-layer) — the refusal must name the offending type via its
    // `Debug` form, not just say "no MLIR tensor form" with no detail.
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::RngState);
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("type has no MLIR tensor form"));
    assert!(err.msg.contains("RngState"));
    assert_eq!(err.node, Some(id));
}

// ---- Task 3: Emitter core -------------------------------------------------

/// The brief's Step-2 example, verbatim: a hand-built two-scalar-add graph,
/// `finish`ed with no arguments.
#[test]
fn emitter_scalar_add_produces_well_formed_module() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(2.0);
    let b = e.scalar(3.0);
    let c = e.add(&a, &b);
    let out = e.finish("logdensity", &[], &c);

    assert!(out.contains("stablehlo.add"));
    assert!(out.contains("func.func @logdensity"));
    assert!(out.contains("return"));
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emitter_finish_wraps_args_and_return_type() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F64);
    let arg = flatppl_stablehlo::Value {
        ssa: "%arg0".to_string(),
        ty: MlirTy::Scalar,
    };
    let doubled = e.add(&arg, &arg);
    let out = e.finish("f", &[("%arg0".to_string(), MlirTy::Scalar)], &doubled);
    assert!(out.starts_with("module {\n"));
    assert!(out.contains("func.func @f(%arg0: tensor<f64>) -> tensor<f64> {"));
    assert!(out.trim_end().ends_with('}'));
    assert!(is_delimiter_balanced(&out));
}

/// Every named elementary wrapper dispatches through `unary`/`binary` to the
/// StableHLO op its doc comment promises, and preserves the operand's shape.
#[test]
fn emitter_elementary_wrappers_emit_expected_ops() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let b = e.scalar(2.0);

    let cases: Vec<(flatppl_stablehlo::Value, &str)> = vec![
        (e.sub(&a, &b), "stablehlo.subtract"),
        (e.mul(&a, &b), "stablehlo.multiply"),
        (e.div(&a, &b), "stablehlo.divide"),
        (e.pow(&a, &b), "stablehlo.power"),
        (e.neg(&a), "stablehlo.negate"),
        (e.log(&a), "stablehlo.log"),
        (e.exp(&a), "stablehlo.exponential"),
        (e.sqrt(&a), "stablehlo.sqrt"),
        (e.abs(&a), "stablehlo.abs"),
        (e.cos(&a), "stablehlo.cosine"),
    ];
    let out = e.finish("f", &[], &cases[0].0);
    for (_, op) in &cases {
        assert!(out.contains(op), "missing {op} in:\n{out}");
    }
    assert!(is_delimiter_balanced(&out));
}

/// `chlo.lgamma` is a function-type op (`in_ty -> out_ty`), not the plain
/// `: ty` form the `stablehlo.*` elementary unaries use — the real
/// StableHLO+CHLO MLIR parser rejects `chlo.lgamma %a : ty` ("expected
/// '->'"). Pin both the op name and the `->` so a regression back to the
/// single-type form is caught.
#[test]
fn emitter_lgamma_emits_function_type_form() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let r = e.lgamma(&a);
    let out = e.finish("f", &[], &r);

    assert!(
        out.contains("chlo.lgamma %"),
        "missing chlo.lgamma in:\n{out}"
    );
    let lgamma_line = out.lines().find(|l| l.contains("chlo.lgamma")).unwrap();
    assert!(
        lgamma_line.contains(" -> "),
        "chlo.lgamma line missing '->' function-type arrow:\n{lgamma_line}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emitter_compare_and_select_type_check() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let b = e.scalar(2.0);
    let pred = e.compare("LT", &a, &b);
    let picked = e.select(&pred, &a, &b);
    let out = e.finish("f", &[], &picked);

    assert!(out.contains("stablehlo.compare LT"));
    assert!(out.contains("tensor<i1>"));
    assert!(out.contains("stablehlo.select"));
    assert!(is_delimiter_balanced(&out));
}

/// `reduce_sum`/`reduce_max` must emit `stablehlo.reduce`'s *pretty* form
/// (`stablehlo.reduce(%in init: %init) applies stablehlo.OP across
/// dimensions = [D] : (...) -> ...`) — the real parser rejects the generic
/// `"stablehlo.reduce"(...) <{dimensions=...}> ({region})` form this crate
/// used to emit. `reduce_max`'s identity must be real dtype-exact negative
/// infinity (`0xFF800000` for f32), not a finite `-1e30` stand-in that is
/// silently wrong for inputs at or below it.
#[test]
fn emitter_reduce_sum_and_max_reduce_to_scalar() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let v = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));
    let s = e.reduce_sum(&v);
    assert_eq!(s.ty, MlirTy::Scalar);

    let mx = e.reduce_max(&v);
    assert_eq!(mx.ty, MlirTy::Scalar);

    let out = e.finish("f", &[], &mx);
    assert!(out.contains("stablehlo.reduce("));
    assert!(
        out.contains("applies stablehlo.add across dimensions"),
        "missing pretty-form add reduce in:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum across dimensions"),
        "missing pretty-form maximum reduce in:\n{out}"
    );
    assert!(
        out.contains("dense<0xFF800000>"),
        "missing dtype-exact -inf reduce_max identity in:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.return"),
        "no region form expected:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Same dtype-exact `-inf` identity check, pinned for `f64` too — the bit
/// pattern (not just the presence of *a* hex literal) is dtype-dependent.
#[test]
fn emitter_reduce_max_f64_identity_is_dtype_exact_neg_inf() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F64);
    let v = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));
    let mx = e.reduce_max(&v);
    let out = e.finish("f", &[], &mx);
    assert!(
        out.contains("dense<0xFFF0000000000000>"),
        "missing f64 -inf identity in:\n{out}"
    );
}

#[test]
fn emitter_reduce_sum_on_scalar_is_a_noop() {
    // A rank-0 operand has no axes to reduce: `reduce_sum` should hand back
    // the same value without emitting a spurious reduce op.
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let s = e.scalar(1.0);
    let summed = e.reduce_sum(&s);
    assert_eq!(summed.ssa, s.ssa);
    assert_eq!(summed.ty, MlirTy::Scalar);
}

#[test]
fn emitter_matrix_helpers_emit_expected_ops() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let mat = e.constant(1.0, MlirTy::Ranked(vec![Some(3), Some(3)]));
    let vec3 = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));

    let l = e.cholesky(&mat);
    assert_eq!(l.ty, mat.ty);
    let d = e.diag(&l);
    assert_eq!(d.ty, MlirTy::Ranked(vec![Some(3)]));
    let mv = e.matvec(&l, &vec3);
    assert_eq!(mv.ty, vec3.ty);
    let y = e.tri_solve(&l, &vec3);
    assert_eq!(y.ty, vec3.ty);

    let out = e.finish("f", &[], &y);
    assert!(out.contains("stablehlo.cholesky"));
    assert!(out.contains("stablehlo.iota"));
    assert!(out.contains("stablehlo.dot_general"));
    assert!(
        out.contains("contracting_dims = [1] x [0]"),
        "missing dot_general pretty-form contracting_dims in:\n{out}"
    );
    assert!(out.contains("precision = [DEFAULT, DEFAULT]"));
    assert!(
        out.contains("\"stablehlo.triangular_solve\"("),
        "missing triangular_solve generic-form head in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// `matvec`'s result type must be `a`'s leading dimension, not `b`'s type —
/// those only coincide in the square case the other test exercises. A
/// rectangular `[m, n] @ [n]` product must produce `[m]`.
#[test]
fn emitter_matvec_result_type_is_lhs_leading_dim() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let mat = e.constant(1.0, MlirTy::Ranked(vec![Some(5), Some(3)]));
    let vec3 = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));

    let mv = e.matvec(&mat, &vec3);
    assert_eq!(mv.ty, MlirTy::Ranked(vec![Some(5)]));
    assert_ne!(mv.ty, vec3.ty);
}

/// `matvec` panics rather than mis-lowering when the shapes don't line up
/// (mirrors the panic-on-bad-shape discipline `diag`/`reduce_axis` already
/// follow) — a mismatched trailing dim is an internal invariant violation
/// upstream type-checking should have already ruled out.
#[test]
#[should_panic(expected = "does not match rhs length")]
fn emitter_matvec_panics_on_shape_mismatch() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let mat = e.constant(1.0, MlirTy::Ranked(vec![Some(5), Some(3)]));
    let vec4 = e.constant(1.0, MlirTy::Ranked(vec![Some(4)]));
    e.matvec(&mat, &vec4);
}

#[test]
fn emitter_fresh_ssa_names_never_repeat() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let b = e.scalar(2.0);
    let c = e.add(&a, &b);
    // Three distinct ops so far (two scalars + one add) must have three
    // distinct SSA names.
    let mut names = vec![a.ssa.clone(), b.ssa.clone(), c.ssa.clone()];
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 3);
}

// ---- Task 4: node dispatch + deterministic op map -------------------------
//
// All of these build tiny FlatPDL fragments by hand (no parse/infer pass —
// `Emitter::lower_node`'s dispatch never consults the type side-table, only
// node structure and already-lowered operand shapes) mirroring Task 2/3's
// hand-built-`Module` test style.

fn top_level(m: &mut Module, name: &str, rhs: NodeId) {
    let sym = m.intern(name);
    m.add_binding(Binding {
        name: sym,
        rhs,
        doc: None,
        public: true,
        synthetic: false,
    });
}

fn call(m: &mut Module, head: &str, args: &[NodeId]) -> NodeId {
    let sym = m.intern(head);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: args.to_vec().into(),
        named: Vec::new().into(),
        inputs: None,
    }))
}

fn self_ref(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name: sym,
    }))
}

fn local_ref(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Ref(Ref {
        ns: RefNs::Local,
        name: sym,
    }))
}

fn const_node(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Const(sym))
}

fn real(m: &mut Module, x: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(x)))
}

fn int(m: &mut Module, i: i64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Int(i)))
}

/// The brief's Step-1 fragment, verbatim: `add(mul(x, 2.0), 1.0)` must emit
/// one `stablehlo.multiply` before one `stablehlo.add` (`x` a top-level
/// binding, reached via `Ref`).
#[test]
fn lower_node_add_mul_emits_multiply_before_add() {
    let mut m = Module::new();
    let x = real(&mut m, 3.0);
    top_level(&mut m, "x", x);
    let x_ref = self_ref(&mut m, "x");
    let two = real(&mut m, 2.0);
    let one = real(&mut m, 1.0);
    let mul_node = call(&mut m, "mul", &[x_ref, two]);
    let add_node = call(&mut m, "add", &[mul_node, one]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(add_node).unwrap();
    let out = e.finish("logdensity", &[], &result);

    let mul_pos = out.find("stablehlo.multiply").expect("missing multiply");
    let add_pos = out.find("stablehlo.add").expect("missing add");
    assert!(mul_pos < add_pos, "expected multiply before add in:\n{out}");
    assert!(is_delimiter_balanced(&out));
}

/// Every named head in the Step-2 map dispatches through `lower_builtin` to
/// the StableHLO op its `Emitter` helper emits.
#[test]
fn lower_builtin_head_map_dispatches_expected_ops() {
    let cases: &[(&str, &str, usize)] = &[
        ("add", "stablehlo.add", 2),
        ("sub", "stablehlo.subtract", 2),
        ("mul", "stablehlo.multiply", 2),
        ("div", "stablehlo.divide", 2),
        ("pow", "stablehlo.power", 2),
        ("neg", "stablehlo.negate", 1),
        ("log", "stablehlo.log", 1),
        ("exp", "stablehlo.exponential", 1),
        ("sqrt", "stablehlo.sqrt", 1),
        ("abs", "stablehlo.abs", 1),
        ("cos", "stablehlo.cosine", 1),
    ];
    for &(head, op, arity) in cases {
        let mut m = Module::new();
        let a = real(&mut m, 2.0);
        let b = real(&mut m, 3.0);
        let args: Vec<NodeId> = if arity == 1 { vec![a] } else { vec![a, b] };
        let node = call(&mut m, head, &args);

        let mut e = Emitter::new(&m, Dtype::F32);
        let result = e.lower_node(node).unwrap();
        let out = e.finish("f", &[], &result);
        assert!(out.contains(op), "head '{head}': missing {op} in:\n{out}");
        assert!(is_delimiter_balanced(&out));
    }
}

/// `ifelse(in(v, interval(lo, hi)), a, neg(inf))` — the exact shape the
/// determiniser's `truncate` lowering builds — must lower to a single
/// `compare` feeding a `select`, and `inf` must use the dtype-exact `+inf`
/// bit pattern (a decimal `f64::INFINITY` literal does not parse as an MLIR
/// float attribute).
#[test]
fn lower_ifelse_of_in_interval_selects_via_stablehlo_select() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let lo = real(&mut m, 0.0);
    let hi = real(&mut m, 1.0);
    let interval = call(&mut m, "interval", &[lo, hi]);
    let cond = call(&mut m, "in", &[v, interval]);
    let a = real(&mut m, 2.0);
    let inf_node = const_node(&mut m, "inf");
    let neg_inf = call(&mut m, "neg", &[inf_node]);
    let ifelse_node = call(&mut m, "ifelse", &[cond, a, neg_inf]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Scalar,
        },
    );
    let result = e.lower_node(ifelse_node).unwrap();
    let out = e.finish(
        "logdensity",
        &[("%arg0".to_string(), MlirTy::Scalar)],
        &result,
    );

    assert!(
        out.contains("stablehlo.compare GE"),
        "missing compare in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.select"),
        "missing select in:\n{out}"
    );
    assert!(
        out.contains("dense<0x7F800000>"),
        "missing dtype-exact +inf in:\n{out}"
    );
    let compare_pos = out.find("stablehlo.compare").unwrap();
    let select_pos = out.find("stablehlo.select").unwrap();
    assert!(compare_pos < select_pos);
    assert!(is_delimiter_balanced(&out));
}

/// `logsumexp(v)` must emit the numerically-stable shift-by-max formula in
/// order: `max` reduce, broadcast the max back up to `v`'s shape, subtract,
/// `exp`, `sum` reduce, `log`, then the final `+ max`.
#[test]
fn lower_logsumexp_emits_stable_shift_by_max_formula_in_order() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let node = call(&mut m, "logsumexp", &[v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(3)]))],
        &result,
    );

    let max_pos = out
        .find("applies stablehlo.maximum across dimensions")
        .expect("missing max reduce");
    let bc_pos = out
        .find("stablehlo.broadcast_in_dim")
        .expect("missing broadcast");
    let sub_pos = out.find("stablehlo.subtract").expect("missing subtract");
    let exp_pos = out
        .find("stablehlo.exponential")
        .expect("missing exponential");
    let sum_pos = out
        .find("applies stablehlo.add across dimensions")
        .expect("missing sum reduce");
    let log_pos = out.find("stablehlo.log").expect("missing log");
    // The outer `log_sum + m` add, distinguished from the sum-reduce's
    // "applies stablehlo.add across ..." text by the `%`-operand form.
    let final_add_pos = out.find("stablehlo.add %").expect("missing final add");

    assert!(max_pos < bc_pos, "max before broadcast, in:\n{out}");
    assert!(bc_pos < sub_pos, "broadcast before subtract, in:\n{out}");
    assert!(sub_pos < exp_pos, "subtract before exp, in:\n{out}");
    assert!(exp_pos < sum_pos, "exp before sum reduce, in:\n{out}");
    assert!(sum_pos < log_pos, "sum reduce before log, in:\n{out}");
    assert!(log_pos < final_add_pos, "log before final add, in:\n{out}");
    assert!(is_delimiter_balanced(&out));
}

/// The REAL shape the determiniser emits (superpose/discrete-marginal):
/// `logsumexp(vector(t1, t2))`, built as an actual `vector(...)` call node —
/// not a pre-`bind`ed synthetic tensor. Must emit `stablehlo.concatenate`
/// (packing the two scalar elements into a length-2 tensor) before the
/// stable logsumexp formula.
#[test]
fn lower_logsumexp_of_vector_emits_concatenate_then_stable_formula() {
    let mut m = Module::new();
    let t1 = local_ref(&mut m, "t1");
    let t2 = local_ref(&mut m, "t2");
    let vec_node = call(&mut m, "vector", &[t1, t2]);
    let node = call(&mut m, "logsumexp", &[vec_node]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        t1,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Scalar,
        },
    );
    e.bind(
        t2,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Scalar,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[
            ("%arg0".to_string(), MlirTy::Scalar),
            ("%arg1".to_string(), MlirTy::Scalar),
        ],
        &result,
    );

    let concat_pos = out
        .find("stablehlo.concatenate")
        .expect("missing concatenate");
    let max_pos = out
        .find("applies stablehlo.maximum across dimensions")
        .expect("missing max reduce");
    let sub_pos = out.find("stablehlo.subtract").expect("missing subtract");
    let exp_pos = out
        .find("stablehlo.exponential")
        .expect("missing exponential");
    let sum_pos = out
        .find("applies stablehlo.add across dimensions")
        .expect("missing sum reduce");
    let log_pos = out.find("stablehlo.log").expect("missing log");

    assert!(concat_pos < max_pos, "concatenate before max, in:\n{out}");
    assert!(max_pos < sub_pos, "max before subtract, in:\n{out}");
    assert!(sub_pos < exp_pos, "subtract before exp, in:\n{out}");
    assert!(exp_pos < sum_pos, "exp before sum reduce, in:\n{out}");
    assert!(sum_pos < log_pos, "sum reduce before log, in:\n{out}");
    assert!(
        out.contains("dim = 0"),
        "missing concatenate dim attr in:\n{out}"
    );
    assert!(
        out.contains("-> tensor<2xf32>"),
        "missing concatenate result shape in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// `sum(a)` (histfactory's `sum(broadcast(builtin_logdensityof, …))`) is a
/// full reduction to a scalar, identical to `Emitter::reduce_sum`.
#[test]
fn lower_sum_reduces_to_scalar_via_reduce_sum() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let node = call(&mut m, "sum", &[v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(3)]))],
        &result,
    );
    assert!(
        out.contains("applies stablehlo.add across dimensions"),
        "missing pretty-form add reduce in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// `ifelse(true, a, b)` — a bare bool-literal condition, not an `in`/
/// `compare` predicate — must refuse rather than let `select` render an
/// ill-typed `i1` predicate operand against a `Lit(Bool)`'s actual
/// `tensor<f32>` lowering.
#[test]
fn lower_ifelse_refuses_non_predicate_condition() {
    let mut m = Module::new();
    let cond = m.alloc(Node::Lit(Scalar::Bool(true)));
    let a = real(&mut m, 1.0);
    let b = real(&mut m, 2.0);
    let node = call(&mut m, "ifelse", &[cond, a, b]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("boolean predicate"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(cond));
}

/// `in(v, interval(lo, hi))` with a scalar `v` (matching lo/hi's shape)
/// reduces to a single `compare` — two `subtract`s and one `multiply`, no
/// `broadcast_in_dim` (shapes already match).
#[test]
fn lower_in_interval_reduces_to_one_compare() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let lo = real(&mut m, 0.0);
    let hi = real(&mut m, 1.0);
    let interval = call(&mut m, "interval", &[lo, hi]);
    let node = call(&mut m, "in", &[v, interval]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Scalar,
        },
    );
    let result = e.lower_node(node).unwrap();
    let out = e.finish("f", &[("%arg0".to_string(), MlirTy::Scalar)], &result);

    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        2,
        "expected v-lo and hi-v subtracts, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.multiply").count(), 1);
    assert!(out.contains("stablehlo.compare GE"));
    assert!(
        !out.contains("broadcast_in_dim"),
        "scalar operands need no broadcast, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn lower_in_refuses_non_interval_set() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let reals = const_node(&mut m, "reals");
    let node = call(&mut m, "in", &[v, reals]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Scalar,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("interval"),
        "unexpected message: {}",
        err.msg
    );
}

/// `get0(v, 2)` on a length-5 rank-1 `v` slices out element 2 then reshapes
/// the length-1 result down to a `Scalar`.
#[test]
fn lower_get0_slices_and_reshapes_to_scalar() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let idx = int(&mut m, 2);
    let node = call(&mut m, "get0", &[v, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(5)]),
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(5)]))],
        &result,
    );

    assert!(
        out.contains("stablehlo.slice %arg0 [2:3]"),
        "unexpected slice bounds in:\n{out}"
    );
    assert!(out.contains("stablehlo.reshape"));
    let slice_pos = out.find("stablehlo.slice").unwrap();
    let reshape_pos = out.find("stablehlo.reshape").unwrap();
    assert!(slice_pos < reshape_pos);
    assert!(is_delimiter_balanced(&out));
}

/// `get(v, 1)` (1-based) must slice the *same* element as `get0(v, 0)`.
#[test]
fn lower_get_is_one_based() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let idx = int(&mut m, 1);
    let node = call(&mut m, "get", &[v, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(5)]),
        },
    );
    let result = e.lower_node(node).unwrap();
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(5)]))],
        &result,
    );
    assert!(
        out.contains("stablehlo.slice %arg0 [0:1]"),
        "expected 1-based get(v, 1) to slice index 0, in:\n{out}"
    );
}

#[test]
fn lower_get0_refuses_non_rank1_container() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let idx = int(&mut m, 0);
    let node = call(&mut m, "get0", &[v, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Scalar,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("rank-1"),
        "unexpected message: {}",
        err.msg
    );
}

#[test]
fn lower_get0_refuses_non_literal_index() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let idx = local_ref(&mut m, "i");
    let node = call(&mut m, "get0", &[v, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(5)]),
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("literal integer"),
        "unexpected message: {}",
        err.msg
    );
}

#[test]
fn lower_get0_refuses_out_of_range_index() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let idx = int(&mut m, 5);
    let node = call(&mut m, "get0", &[v, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("out of range"),
        "unexpected message: {}",
        err.msg
    );
}

/// A `Ref`ed top-level binding used from two call sites must be lowered
/// once: the shared ancestor's op(s) appear exactly once in the output, and
/// both use sites reuse the same SSA name.
#[test]
fn lower_node_memoizes_shared_ancestor() {
    let mut m = Module::new();
    let x = real(&mut m, 5.0);
    top_level(&mut m, "x", x);
    let x_ref1 = self_ref(&mut m, "x");
    let x_ref2 = self_ref(&mut m, "x");
    let two = real(&mut m, 2.0);
    let doubled = call(&mut m, "mul", &[x_ref2, two]);
    let node = call(&mut m, "add", &[x_ref1, doubled]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(node).unwrap();
    let out = e.finish("f", &[], &result);

    assert_eq!(
        out.matches("dense<5").count(),
        1,
        "x re-emitted instead of reused, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn lower_builtin_refuses_unknown_head() {
    let mut m = Module::new();
    let a = real(&mut m, 1.0);
    let node = call(&mut m, "frobnicate", &[a]);
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("unsupported builtin head 'frobnicate'"),
        "{}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

#[test]
fn lower_builtin_refuses_wrong_arity() {
    let mut m = Module::new();
    let a = real(&mut m, 1.0);
    let node = call(&mut m, "add", &[a]);
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("expected 2 argument"),
        "unexpected message: {}",
        err.msg
    );
}

#[test]
fn lower_builtin_refuses_record_in_tensor_position() {
    let mut m = Module::new();
    let node = call(&mut m, "record", &[]);
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("record has no tensor form"));
}

#[test]
fn lower_node_refuses_user_callable_application() {
    let mut m = Module::new();
    let callee = self_ref(&mut m, "f");
    let arg = real(&mut m, 1.0);
    let node = m.alloc(Node::Call(Call {
        head: CallHead::User(callee),
        args: vec![arg].into(),
        named: Vec::new().into(),
        inputs: None,
    }));
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("user-callable"));
}

#[test]
fn lower_node_refuses_unresolved_self_reference() {
    let mut m = Module::new();
    let node = self_ref(&mut m, "nope");
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("unresolved reference"));
}

#[test]
fn lower_node_refuses_unbound_local_reference() {
    let mut m = Module::new();
    let node = local_ref(&mut m, "theta");
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("%local"));
}

#[test]
fn lower_node_refuses_module_member_reference() {
    let mut m = Module::new();
    let alias = m.intern("hepphys");
    let name = m.intern("foo");
    let node = m.alloc(Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }));
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("module-member"));
}

#[test]
fn lower_node_refuses_bare_hole() {
    let mut m = Module::new();
    let node = m.alloc(Node::Hole);
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("hole"));
}

#[test]
fn lower_node_refuses_axis_label() {
    let mut m = Module::new();
    let name = m.intern("i");
    let node = m.alloc(Node::Axis(flatppl_core::Axis {
        name,
        variance: None,
    }));
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("axis"));
}

#[test]
fn lower_node_refuses_string_literal() {
    let mut m = Module::new();
    let node = m.alloc(Node::Lit(Scalar::Str("hi".into())));
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("string literal"));
}

#[test]
fn lower_node_lowers_int_and_bool_literals_as_scalars() {
    let mut m = Module::new();
    let i = int(&mut m, 7);
    let b = m.alloc(Node::Lit(Scalar::Bool(true)));
    let mut e = Emitter::new(&m, Dtype::F32);
    let iv = e.lower_node(i).unwrap();
    let bv = e.lower_node(b).unwrap();
    assert_eq!(iv.ty, MlirTy::Scalar);
    assert_eq!(bv.ty, MlirTy::Scalar);
    let out = e.finish("f", &[], &bv);
    assert!(out.contains("dense<7"));
    assert!(out.contains("dense<1"));
}

/// A bare `Const` symbol (`inf`) is dispatched through the same builtin-head
/// map as a zero-arg call, and must use the dtype-exact `+inf` bit pattern.
#[test]
fn lower_const_inf_emits_dtype_exact_positive_infinity() {
    let mut m = Module::new();
    let node = const_node(&mut m, "inf");
    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish("f", &[], &result);
    assert!(
        out.contains("dense<0x7F800000>"),
        "missing dtype-exact +inf in:\n{out}"
    );
}

// ---- Task 5: distribution registry + Normal `@logdensity` -------------------
//
// The registry framework (`registry.rs`: ctor-name-keyed `DistLowering`
// table + `Params`), the §08 Normal `logpdf` builder, and the `emit_logdensity`
// mode builder (`modes.rs`) — the first fully emitted StableHLO module (the
// density vertical slice). `Emitter::lower_node`'s `builtin_logdensityof`
// head now dispatches through the registry instead of falling through to
// `ops::lower_builtin`'s catch-all refusal.

/// The Task-5 anchor fixture: a scalar Normal with free (`elementof`-declared)
/// `mu`/`sigma`, scored at a pinned observation via
/// `logdensityof(lawof(record(...)), record(...))` — the same record-of-draws
/// shape every `flatppl-determinizer` density golden uses
/// (`crates/determinizer/tests/density_golden.rs`), just with `elementof`
/// parameters (not literals) so `mu`/`sigma` survive determinize as free
/// parameters rather than being folded away.
const NORMAL_DENSITY_SRC: &str = "\
mu = elementof(reals)
sigma = elementof(posreals)
a = draw(Normal(mu = mu, sigma = sigma))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

/// Parse, infer, and determinize `src`, panicking (with the diagnostics/
/// refusal) if any step fails — the shared setup for every Task-5 test below.
fn determinize_src(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    flatppl_determinizer::determinize(&m).expect("must determinize, not refuse")
}

fn emit_logdensity(m: &Module) -> String {
    flatppl_stablehlo::emit(
        m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

/// The brief's Step-1 structural test: `mu`/`sigma` become `func.func` args
/// (free parameters), the pinned observation is walked to a
/// `stablehlo.constant` (no special-casing needed — `Lit` dispatch already
/// handles it), and the Normal formula's op counts are exact. Normal needs no
/// `chlo.*` special function.
#[test]
fn emit_logdensity_normal_has_expected_structure() {
    let d = determinize_src(NORMAL_DENSITY_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_logdensity(&d);

    assert!(
        out.contains("func.func @logdensity"),
        "missing func.func @logdensity in:\n{out}"
    );
    assert!(
        out.contains("-> tensor<f32>"),
        "must return tensor<f32> in:\n{out}"
    );
    assert!(
        out.contains("%arg0: tensor<f32>") && out.contains("%arg1: tensor<f32>"),
        "mu/sigma must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        1,
        "expected exactly one log, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        1,
        "expected exactly one negate, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        1,
        "expected exactly one subtract, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.divide").count(),
        1,
        "expected exactly one divide, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        2,
        "expected exactly two multiplies, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        2,
        "expected exactly two adds, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Normal needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_normal_matches_frozen_golden() {
    let d = determinize_src(NORMAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/normal_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/normal_logdensity.mlir)"
    );
}

/// Build a `record(%field name1 = value1, ...)` call node — a hand-built
/// `kernel_input` for the `builtin_logdensityof` refuse tests below (mirrors
/// `flatppl_determinizer::density::build_record`, which this crate cannot
/// depend on directly).
fn record_node(m: &mut Module, fields: &[(&str, NodeId)]) -> NodeId {
    let head = m.intern("record");
    let named: Vec<flatppl_core::NamedArg> = fields
        .iter()
        .map(|&(name, value)| flatppl_core::NamedArg {
            kind: flatppl_core::NamedKind::Field,
            name: m.intern(name),
            value,
        })
        .collect();
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Vec::<NodeId>::new().into(),
        named: named.into(),
        inputs: None,
    }))
}

/// `builtin_logdensityof(Cauchy, ..., v)` — a distribution with no registry
/// entry — must refuse precisely (refuse-don't-mislower), not panic or guess
/// a lowering.
#[test]
fn builtin_logdensityof_refuses_unregistered_ctor() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Cauchy");
    let field_val = real(&mut m, 0.0);
    let kernel_input = record_node(&mut m, &[("x0", field_val)]);
    let v = real(&mut m, 1.0);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no lowering for distribution 'Cauchy'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `builtin_logdensityof`'s kernel must be a bare `Const` distribution
/// constructor (never a general expression) — a `Ref` in that position
/// refuses rather than being silently mis-resolved.
#[test]
fn builtin_logdensityof_refuses_non_const_kernel() {
    let mut m = Module::new();
    let kernel = local_ref(&mut m, "k");
    let kernel_input = call(&mut m, "record", &[]);
    let v = real(&mut m, 1.0);
    let node = call(&mut m, "builtin_logdensityof", &[kernel, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("bare distribution constructor"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(kernel));
}

/// A `builtin_logdensityof(Normal, kernel_input, v)` whose `kernel_input`
/// record is missing a parameter the registry entry needs (`sigma`) must
/// refuse, naming the missing field, rather than panicking on the `None`.
#[test]
fn normal_logpdf_refuses_missing_kernel_input_field() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let mu_val = real(&mut m, 0.0);
    let kernel_input = record_node(&mut m, &[("mu", mu_val)]);
    let v = real(&mut m, 1.0);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(err.msg.contains("sigma"), "unexpected message: {}", err.msg);
}

/// A trailing public binding *after* the density expression (e.g. a
/// diagnostic/auxiliary value) must not be silently lowered as the query
/// output just because it happens to be the last public binding in source
/// order — `Module`'s own doc disclaims that binding order carries spec
/// meaning. `emit_logdensity` must refuse (precisely, naming the missing
/// density term) rather than mis-lower it.
#[test]
fn emit_logdensity_refuses_trailing_binding_with_no_density_term() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let mu = real(&mut m, 0.0);
    let sigma = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("sigma", sigma)]);
    let v = real(&mut m, 0.5);
    let density = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);
    top_level(&mut m, "lp", density);

    // A diagnostic/auxiliary binding that happens to land after `lp` in
    // source order — no density term anywhere in its subtree.
    let diag = real(&mut m, 42.0);
    top_level(&mut m, "diag", diag);

    let err = flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("contains no density term (builtin_logdensityof)"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(diag));
}

// ---- Task 6: Normal `@sample` + `emit_sample` (sampling vertical slice) ----
//
// `Emitter::rng` (`stablehlo.rng`), Normal's `@sample` builder (§08's
// `mu + sigma * Z` transform), and the `emit_sample` mode builder
// (`modes.rs`), wired up as `emit`'s `Mode::Sample` route.

/// The Task-6 anchor fixture: a fixed-hyperparameter scalar Normal forward
/// model, sampled via the value-terminal `rand(rng, lawof(x))` convention
/// (`flatppl_determinizer::sample`). Verified (via a throwaway determinize +
/// `flatppl_flatpir::write` dump) to determinize to `draws`'s RHS being
/// exactly `get0(builtin_sample(s, Normal, record(mu=0.0, sigma=1.0)), 0)` —
/// no wrapping `record(...)` around the single draw, unlike
/// `crates/determinizer/tests/sample_golden.rs`'s `record(x = x)` fixtures
/// (this is `lawof(x)` directly, not `lawof(record(x = x))`, so
/// `lower_measure_sample` dispatches straight to `lower_draw`, never
/// `lower_record_of_draws_sample`). Fixed (not `elementof`) hyperparameters,
/// so `emit_sample` should produce a `func.func @sample()` with no args.
const NORMAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(x))
";

fn emit_sample(m: &Module) -> String {
    flatppl_stablehlo::emit(
        m,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @sample")
}

/// The brief's Step-1 structural test: `func.func @sample` with no args (a
/// fixed-hyperparameter prior), exactly one `stablehlo.rng` with
/// `distribution = NORMAL`, returning the drawn `tensor<f32>` variate.
#[test]
fn emit_sample_normal_has_expected_structure() {
    let d = determinize_src(NORMAL_SAMPLE_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample()"),
        "missing func.func @sample() (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> tensor<f32>"),
        "must return tensor<f32> in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng").count(),
        1,
        "expected exactly one stablehlo.rng, in:\n{out}"
    );
    assert!(
        out.contains("distribution = NORMAL"),
        "missing NORMAL distribution attr, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_sample_normal_matches_frozen_golden() {
    let d = determinize_src(NORMAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/normal_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/normal_sample.mlir)"
    );
}

/// The `emit_sample` analogue of
/// `emit_logdensity_refuses_trailing_binding_with_no_density_term`: a
/// trailing public binding with no `builtin_sample` anywhere in its subtree
/// must refuse rather than be silently lowered just because it is the last
/// public binding in source order.
#[test]
fn emit_sample_refuses_trailing_binding_with_no_sample_term() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let mu = real(&mut m, 0.0);
    let sigma = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("sigma", sigma)]);
    let rng = real(&mut m, 0.0); // stand-in rng-state arg (never lowered)
    let sample = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);
    let zero_idx = int(&mut m, 0);
    let draws = call(&mut m, "get0", &[sample, zero_idx]);
    top_level(&mut m, "draws", draws);

    // A diagnostic/auxiliary binding that happens to land after `draws` in
    // source order — no sample term anywhere in its subtree.
    let diag = real(&mut m, 42.0);
    top_level(&mut m, "diag", diag);

    let err = flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("contains no sample term (builtin_sample)"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(diag));
}

// ---- Task 6 review fix: `contains_sample_call` ref-following (Finding 1) --
//
// `contains_sample_call`'s guard used to walk the query subtree via
// `Node::for_each_child` alone, which does not descend through `Node::Ref` —
// so a record/hierarchical `@sample` forward model, whose query is
// `record(mu = (%ref self mu), y = (%ref self y))` with the rewritten
// `builtin_sample` sitting one or more binding-hops away on each ref's
// resolved RHS (`flatppl_determinizer::sample::lower_shared_record_sample`),
// refused at the guard ("no sample term") before a real lowering attempt
// ever ran. `modes.rs`'s `contains_sample_call` now follows `(%ref self x)`
// leaves to `x`'s bound RHS TRANSITIVELY (a `HashSet` visited-set guards
// against a cycle).

/// Isolates just the ref-chasing fix, independent of the separate
/// record-output limitation documented below: `query` refs `a` refs `b`
/// refs `builtin_sample(...)`, TWO hops deep with no intervening `Call`
/// wrapper — the old one-`for_each_child`-hop walk (and even a
/// single-ref-hop resolution, mirroring `Emitter::resolves_to_builtin_sample`'s
/// deliberately-one-hop rule) would not reach it. This must both pass the
/// guard AND fully emit, since the query itself is a plain scalar sample
/// (no record involved).
#[test]
fn emit_sample_query_reaches_sample_via_chained_self_refs() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let mu = real(&mut m, 0.0);
    let sigma = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("sigma", sigma)]);
    let rng = real(&mut m, 0.0); // stand-in rng-state arg (never lowered)
    let sample = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);
    let zero_idx = int(&mut m, 0);
    let value = call(&mut m, "get0", &[sample, zero_idx]);
    top_level(&mut m, "b", value);

    let b_ref = self_ref(&mut m, "b");
    top_level(&mut m, "a", b_ref);

    let query = self_ref(&mut m, "a");
    top_level(&mut m, "query", query);

    let out = flatppl_stablehlo::emit(&m, flatppl_stablehlo::Mode::Sample, &Default::default())
        .expect("must emit @sample: query reaches builtin_sample via a 2-hop self-ref chain");
    assert_eq!(
        out.matches("stablehlo.rng").count(),
        1,
        "expected exactly one stablehlo.rng, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// The review's canonical fixture: a genuinely hierarchical, record-output
/// forward model (`y`'s Normal mean is `mu`, itself drawn; both are read
/// back out via the output record) — `flatppl_determinizer::sample`'s
/// `lower_shared_record_sample` path (verified via a throwaway determinize +
/// `flatppl_flatpir::write` dump: `mu`'s and `y`'s draw-bindings are each
/// rewritten in place to `get0((%ref self __sample_*), 0)`, and `draws`'s
/// RHS is `record(mu = (%ref self mu), y = (%ref self y))`).
const HIERARCHICAL_SAMPLE_SRC: &str = "\
mu = draw(Normal(mu = 0.0, sigma = 1.0))
y  = draw(Normal(mu = mu, sigma = 1.0))
s  = rnginit(0)
draws = rand(s, lawof(record(mu = mu, y = y)))
";

/// This fixture's query (`draws`'s RHS) now correctly PASSES
/// `contains_sample_call`'s guard (it no longer refuses with "no sample
/// term" — the false-negative Finding 1 reported). Emission then refuses
/// for a DIFFERENT, genuine reason: the query is `record(...)`-typed, and
/// `ops::lower_builtin`'s `"record"` arm has no tensor form for it — the
/// mode builder has no structural decomposition for a record-SHAPED
/// `@sample` OUTPUT (only for a record-shaped free-parameter *input*, via
/// the `elementof` loop). Deciding that output ABI (multiple `func.func`
/// results? a `stablehlo.tuple`? field order convention?) is a new-capability
/// design decision outside a Task 6 review-findings fix, not forced here —
/// see the fix-pass report for the concern writeup. This test locks in that
/// the GUARD itself is fixed without overclaiming the record-output case
/// fully emits.
#[test]
fn emit_sample_hierarchical_record_passes_guard_refuses_on_record_output() {
    let d = determinize_src(HIERARCHICAL_SAMPLE_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        !err.msg.contains("no sample term"),
        "the query-output guard must no longer refuse a record/hierarchical \
         query that DOES contain a builtin_sample term (reached only via a \
         chain of self-refs): {}",
        err.msg
    );
    assert!(
        err.msg.contains("record has no tensor form"),
        "expected the record-output limitation, not a different refusal: {}",
        err.msg
    );
}

// ---- Task 6 review fix: `lower_sample` refuse tests (Finding 2) -----------
//
// Task 6 shipped `registry::lower_sample`'s three refuse-don't-mislower
// guards (arity, non-const ctor, unregistered ctor) with no direct test
// coverage — `lower_logdensityof`'s equivalent guards
// (`builtin_logdensityof_refuses_unregistered_ctor` /
// `_refuses_non_const_kernel`, Task 5 above) are the precedent these mirror.

/// `builtin_sample(rng, Cauchy, kernel_input)` — a ctor with no registry
/// entry at all (only `Normal` is registered, spec §08) — must refuse
/// precisely, not panic or guess a lowering. This exercises the same
/// `registry::lookup` miss `builtin_logdensityof_refuses_unregistered_ctor`
/// does (shared code, not sample-specific text): `lower_sample`'s OWN
/// sample-specific message, `"no @sample lowering for '{ctor}'"`
/// (`dist.sample.ok_or_else`, for a ctor registered for `@logdensity` with
/// no `@sample` builder yet), is currently unreachable by any real ctor
/// name — every registered entry (just `Normal`) has both builders today —
/// so it cannot be exercised without adding a fake registry entry purely
/// for this test.
#[test]
fn builtin_sample_refuses_unregistered_ctor() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0); // stand-in rng-state arg (never lowered)
    let ctor = const_node(&mut m, "Cauchy");
    let mu_val = real(&mut m, 0.0);
    let sigma_val = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu_val), ("sigma", sigma_val)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no lowering for distribution 'Cauchy'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `builtin_sample`'s ctor must be a bare `Const` distribution constructor
/// (never a general expression) — mirrors
/// `builtin_logdensityof_refuses_non_const_kernel`.
#[test]
fn builtin_sample_refuses_non_const_ctor() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = local_ref(&mut m, "k");
    let kernel_input = call(&mut m, "record", &[]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("bare distribution constructor"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(ctor));
}

/// `builtin_sample` with the wrong number of arguments must refuse (naming
/// the exact expected/actual count), not panic on the
/// `<[NodeId; 3]>::try_from`.
#[test]
fn builtin_sample_refuses_wrong_arity() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "Normal");
    let node = call(&mut m, "builtin_sample", &[rng, ctor]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("builtin_sample: expected 3 arguments, got 2"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

// ---- Task 6 review fix: `Emitter::rng` defensive assert (Finding 3) -------

/// `stablehlo.rng` requires rank-0 `a`/`b` bounds operands — a non-`Scalar`
/// `a` must panic (an internal invariant violation caught before emitting
/// ill-typed StableHLO), mirroring `diag`/`matvec`'s panic-on-bad-shape
/// discipline in the same file.
#[test]
#[should_panic(expected = "rng expects a rank-0 (scalar) `a` operand")]
fn emitter_rng_panics_on_non_scalar_a() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = flatppl_stablehlo::Value {
        ssa: "%a".to_string(),
        ty: MlirTy::Ranked(vec![Some(3)]),
    };
    let b = e.scalar(1.0);
    e.rng("NORMAL", &a, &b, &MlirTy::Scalar);
}

/// The `b`-operand mirror of [`emitter_rng_panics_on_non_scalar_a`].
#[test]
#[should_panic(expected = "rng expects a rank-0 (scalar) `b` operand")]
fn emitter_rng_panics_on_non_scalar_b() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(0.0);
    let b = flatppl_stablehlo::Value {
        ssa: "%b".to_string(),
        ty: MlirTy::Ranked(vec![Some(3)]),
    };
    e.rng("NORMAL", &a, &b, &MlirTy::Scalar);
}
