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
    let src = "flatppl_compat = \"0.1\"\nx = 1.0\n";
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
