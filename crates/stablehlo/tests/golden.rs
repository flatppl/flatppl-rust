//! Golden tests for `flatppl-stablehlo`'s `emit`: parse + infer + determinize
//! a FlatPPL source, emit textual StableHLO for both `Mode::LogDensity` and
//! `Mode::Sample`, and check the emitted MLIR against expectations (including
//! per-distribution logpdf/sample coverage in `registry.rs`).
//!
//! The `mlir_type_of` tests (the `Type`/`Dim` → MLIR `tensor<…>` mapping)
//! build tiny `Module`s by hand (`alloc` a placeholder node, `set_type` the
//! type under test) rather than parsing + inferring source, since only the
//! type side-table matters for this mapping.

use flatppl_core::{
    Binding, Call, CallHead, Dim, Mass, Module, Node, NodeId, Ref, RefNs, Scalar, ScalarType, Type,
};
use flatppl_stablehlo::{Dtype, ElemKind, Emitter, MlirTy, Value, mlir_type_of};

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

// A broadcast/iid likelihood (`mu = a .+ x`, `y ~ Normal.(mu, 0.5)`) determinizes
// to `sum(broadcast(builtin_logdensityof, Normal, broadcast(record, mu = <vec>,
// sigma = 0.5), y_obs))` — the batched-density path (§04 sec:broadcasting). The
// emitter's `broadcast` lowering must run the rank-agnostic Normal logpdf over
// the rank-1 (`3x`) batch, scalars auto-broadcasting, then reduce. Verified
// out-of-tree to IREE-execute == scipy oracle (linear_regression/partial_pooling);
// this golden pins the emit + the batched shape. Buffy #303 (broadcast gap).
#[test]
fn broadcast_iid_likelihood_lowers_to_batched_density() {
    let src = "flatppl_compat = \"0.1\"\n\
x = [1.0, 2.0, 3.0]\n\
y_obs = [1.1, 2.2, 2.9]\n\
a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
mu = a .+ x\n\
y = draw(Normal.(mu, 0.5))\n\
L = likelihoodof(kernelof(record(y = y), a = a), record(y = y_obs))\n\
post = bayesupdate(L, lawof(record(a = a)))\n\
score = logdensityof(post, record(a = 0.5))\n";
    let m = flatppl_syntax::parse(src).unwrap();
    let d = flatppl_determinizer::determinize(&m).unwrap();
    let out = flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .unwrap();
    assert!(out.contains("module {") && is_delimiter_balanced(&out));
    // The 3-observation batch survives as a rank-1 tensor, is reduced (the iid
    // sum), and scalars are broadcast to it.
    assert!(out.contains("tensor<3xf32>"), "batch dim present:\n{out}");
    assert!(
        out.contains("stablehlo.reduce"),
        "iid sum reduces the batch:\n{out}"
    );
    assert!(
        out.contains("stablehlo.broadcast"),
        "scalar auto-broadcast:\n{out}"
    );
}

// `iid(Dist, n)` axis-native density with SCALAR kernel params: the determiniser
// lifts each scalar param to a length-1 array-of-records
// (`sum(broadcast(builtin_logdensityof, K, <length-1 Array{Record} params>, obs))`),
// distinct from `broadcast_iid_likelihood_lowers_to_batched_density` above whose
// per-cell `mu` is itself already length-`n` (`a .+ x`). Here the iid kernel is
// `Normal(mu = a, sigma = 1.0)` — `mu` the sole `elementof` parameter, `sigma`
// a scalar constant — scored against the length-3 observation `y_obs`. The
// per-cell `builtin_logdensityof` therefore combines a `tensor<1xf32>` param
// with the `tensor<3xf32>` variate — a Ranked/Ranked pair StableHLO's raw
// elementwise ops cannot take directly (operand shapes must be IDENTICAL; no
// implicit broadcast). `Emitter::broadcast_pair` must reconcile the size-1 axis
// via an explicit `stablehlo.broadcast_in_dim`; before that fix the emitter
// silently produced mismatched-shape op text, which only `MLIRError: Unable to
// parse module assembly` (IREE/Enzyme-JAX execution in `flatppl-testsuite`)
// caught — never in-tree, since this crate's other goldens only
// `.contains(...)`-check substrings rather than parse-validate the MLIR.
//
// The query is designated by the `inputs`/`outputs` ABI (order-agnostic —
// FlatPPL is not order-dependent), NOT by binding position.
#[test]
fn iid_density_broadcasts_size_one_params_to_batch_shape() {
    let src = "\
y_obs = [0.5, -0.3, 1.2]
a = elementof(reals)
m = lawof(record(y = draw(iid(Normal(mu = a, sigma = 1.0), 3))))
q = logdensityof(m, record(y = y_obs))
inputs = (a)
outputs = (q)
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let mlir = emit_logdensity(&d);
    assert!(
        mlir.contains("module {") && is_delimiter_balanced(&mlir),
        "well-formed module:\n{mlir}"
    );
    // The length-1 params broadcast up to the length-3 batch via an explicit
    // `stablehlo.broadcast_in_dim` under an IDENTITY dimension map (`dims =
    // [0]`, the only axis) — NOT a raw op left on mismatched
    // `tensor<1xf32>`/`tensor<3xf32>` operands.
    assert!(
        mlir.contains("tensor<1xf32>") && mlir.contains("tensor<3xf32>"),
        "both the length-1 param shape and the length-3 batch shape appear:\n{mlir}"
    );
    assert!(
        mlir.contains("stablehlo.broadcast_in_dim") && mlir.contains("dims = [0]"),
        "size-1 axis expands via an explicit identity broadcast_in_dim:\n{mlir}"
    );
    // No ELEMENTWISE arithmetic op may mix a `tensor<1xf32>` with a
    // `tensor<3xf32>` operand (the raw-mismatch bug this test guards against) —
    // a size-1 param must be `broadcast_in_dim`-expanded to the batch shape
    // FIRST. Shape ops legitimately combine the two: `broadcast_in_dim` IS the
    // expansion, and `concatenate` stacks the three length-1 literals into the
    // length-3 observation vector.
    const ELEMENTWISE: &[&str] = &[
        "stablehlo.add",
        "stablehlo.subtract",
        "stablehlo.multiply",
        "stablehlo.divide",
        "stablehlo.power",
        "stablehlo.maximum",
        "stablehlo.minimum",
    ];
    for line in mlir.lines() {
        if !line.contains("tensor<1xf32>") || !line.contains("tensor<3xf32>") {
            continue;
        }
        assert!(
            !ELEMENTWISE.iter().any(|op| line.contains(op)),
            "an elementwise op must not mix tensor<1xf32> and tensor<3xf32> operands \
             (a size-1 param was not broadcast_in_dim-expanded first):\n{line}"
        );
    }
}

// Nested `iid(iid(Dist, k), n)` axis-native FLATTEN, rank-2: the SAME
// single-axis broadcast form as the rank-1 golden above
// (`iid_density_broadcasts_size_one_params_to_batch_shape`), but the innermost
// kernel is peeled through TWO `iid` layers (`depth = 2` in `lower_iid`'s
// nested fast path) and scored against a RANK-2 `[3, 2]` observation rather
// than rank-1 — previously only the rank-1 primitive case had an in-tree
// parse-validating golden; the rank-2 nested case rested solely on the
// external IREE gate (`flatppl-testsuite`). Verified out-of-tree to
// IREE-execute == the closed-form oracle
// (`iid(iid(Normal(0,1),2),3)@[[0.5,-0.3],[1.2,0.1],[-0.7,0.9]] ==
// -7.058631199228036`); this golden pins the emit + parse-validates the
// rank-2 MLIR in-tree.
#[test]
fn iid_density_flattens_nested_iid_to_rank2_batch_shape() {
    let src = "\
y_obs = [[0.5, -0.3], [1.2, 0.1], [-0.7, 0.9]]
a = elementof(reals)
m = lawof(record(y = draw(iid(iid(Normal(mu = a, sigma = 1.0), 2), 3))))
q = logdensityof(m, record(y = y_obs))
inputs = (a)
outputs = (q)
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    // A SINGLE `builtin_logdensityof` broadcast head lowers the WHOLE nested
    // product (the flatten from `lower_iid`'s nested fast path) — not two
    // separate per-layer broadcasts from a rank-by-rank `get0` recursion, and
    // no `get0` unroll at all (the safe fallback this fast path bypasses).
    let pir = flatppl_flatpir::write(&d);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "exactly one flattened broadcast head, not a per-layer recursion:\n{pir}"
    );
    assert!(
        !pir.contains("get0"),
        "the flatten fast path must not fall back to the get0 unroll:\n{pir}"
    );

    let mlir = emit_logdensity(&d);
    assert!(
        mlir.contains("module {") && is_delimiter_balanced(&mlir),
        "well-formed module:\n{mlir}"
    );
    // The depth-2 nested params (each a `vector(vector(param))`, shape
    // `[1, 1]`, §04 "Size-one array axes are implicitly expanded by
    // repetition") broadcast up to the rank-2 `[3, 2]` batch shape.
    assert!(
        mlir.contains("tensor<1x1xf32>") && mlir.contains("tensor<3x2xf32>"),
        "both the depth-2 size-1 param shape and the rank-2 batch shape appear:\n{mlir}"
    );
    // The size-1 axes expand via an explicit `stablehlo.broadcast_in_dim`
    // under a RANK-2 IDENTITY dimension map (`dims = [0, 1]`, both axes) — not
    // a raw op left on mismatched `tensor<1x1xf32>`/`tensor<3x2xf32>` operands.
    assert!(
        mlir.contains("stablehlo.broadcast_in_dim") && mlir.contains("dims = [0, 1]"),
        "rank-2 axes expand via an explicit identity broadcast_in_dim:\n{mlir}"
    );
    // No ELEMENTWISE arithmetic op may mix a `tensor<1x1xf32>` with a
    // `tensor<3x2xf32>` operand (mirrors the rank-1 golden's guard) — a size-1
    // param must be `broadcast_in_dim`-expanded to the batch shape FIRST.
    const ELEMENTWISE: &[&str] = &[
        "stablehlo.add",
        "stablehlo.subtract",
        "stablehlo.multiply",
        "stablehlo.divide",
        "stablehlo.power",
        "stablehlo.maximum",
        "stablehlo.minimum",
    ];
    for line in mlir.lines() {
        if !line.contains("tensor<1x1xf32>") || !line.contains("tensor<3x2xf32>") {
            continue;
        }
        assert!(
            !ELEMENTWISE.iter().any(|op| line.contains(op)),
            "an elementwise op must not mix tensor<1x1xf32> and tensor<3x2xf32> operands \
             (a size-1 axis was not broadcast_in_dim-expanded first):\n{line}"
        );
    }
}

// A `broadcast` whose callable is a USER FUNCTION (not a bare builtin): a
// Poisson GLM with a named linear predictor `predict(i,s,x)=i+s*x` broadcast
// over the covariate array and a named inverse-link `rate(eta)=exp(eta)`
// dot-called elementwise (§04 sec:broadcasting; §05 "Named functions"). The
// determiniser keeps these as `broadcast((%ref self predict), intercept=…,
// slope=…, xi=<vec>)` / `rate.(<vec>)` — a callable in higher-order position is
// NOT inlined (unlike a direct user call). The emitter must monomorphise the
// reified `functionof` elementwise: bind each input (by keyword name / position)
// and lower the body, whose arithmetic auto-broadcasts scalar↔rank-1. This is
// the ex_poisson_glm_link posterior; verified out-of-tree to Enzyme-execute ==
// scipy oracle (Δ≈2.5e-6 f32) at the corpus theta points. Buffy #328.
#[test]
fn broadcast_of_user_function_monomorphises_elementwise() {
    let src = "flatppl_compat = \"0.1\"\n\
x = [-1.0, 0.2, 0.5, 1.3, 2.1]\n\
y_obs = [0, 1, 2, 3, 8]\n\
a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
b = draw(Normal(mu = 0.0, sigma = 1.0))\n\
predict(intercept, slope, xi) = intercept + slope * xi\n\
rate(eta) = exp(eta)\n\
eta = broadcast(predict, intercept = a, slope = b, xi = x)\n\
mu = rate.(eta)\n\
y = draw(Poisson.(mu))\n\
L = likelihoodof(kernelof(record(y = y), a = a, b = b), record(y = y_obs))\n\
post = bayesupdate(L, lawof(record(a = a, b = b)))\n\
score = logdensityof(post, record(a = 0.0, b = 0.0))\n";
    let m = flatppl_syntax::parse(src).unwrap();
    let d = flatppl_determinizer::determinize(&m).unwrap();
    // Pin that this actually exercises the user-function broadcast path: the
    // determinised FlatPDL keeps the reified `functionof`s the query broadcasts.
    let pdl = flatppl_syntax::print(&d);
    assert!(
        pdl.contains("broadcast(predict") && pdl.contains("rate.("),
        "determinised query broadcasts the user functions:\n{pdl}"
    );
    let out = flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .unwrap();
    assert!(out.contains("module {") && is_delimiter_balanced(&out));
    // predict's body inlined over the length-5 covariate batch (rank-1), rate's
    // `exp` applied elementwise, then the iid Poisson log-likelihood reduced.
    assert!(out.contains("tensor<5xf32>"), "batch dim present:\n{out}");
    assert!(
        out.contains("stablehlo.multiply") && out.contains("stablehlo.add"),
        "predict body (i + s*x) inlined:\n{out}"
    );
    assert!(
        out.contains("stablehlo.exponential"),
        "rate body (exp) inlined elementwise:\n{out}"
    );
    assert!(
        out.contains("stablehlo.reduce"),
        "iid Poisson sum reduces the batch:\n{out}"
    );
}

// `invlogit(x)` (§07 logistic sigmoid, e.g. a logit-link GLM's inverse link)
// lowers to the native, numerically-stable `stablehlo.logistic` op. Verified
// out-of-tree to IREE-execute == scipy `expit` oracle (Δ≈2e-8). Buffy #303.
#[test]
fn invlogit_lowers_to_stablehlo_logistic() {
    let src = "flatppl_compat = \"0.1\"\n\
a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
p = invlogit(a)\n\
y = draw(Bernoulli(p))\n\
L = likelihoodof(kernelof(record(y = y), a = a), record(y = 1))\n\
post = bayesupdate(L, lawof(record(a = a)))\n\
score = logdensityof(post, record(a = 0.7))\n";
    let m = flatppl_syntax::parse(src).unwrap();
    let d = flatppl_determinizer::determinize(&m).unwrap();
    let out = flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .unwrap();
    assert!(out.contains("module {") && is_delimiter_balanced(&out));
    assert!(
        out.contains("stablehlo.logistic"),
        "invlogit emits the native logistic op:\n{out}"
    );
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
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Scalar);
    assert_eq!(ty.render(Dtype::F32, elem), "tensor<f32>");
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
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
    assert_eq!(ty.render(Dtype::F32, elem), "tensor<2x3xf32>");
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
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
    assert_eq!(ty.render(Dtype::F32, elem), "tensor<2x3xf32>");
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
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![None, Some(3)]));
    assert_eq!(ty.render(Dtype::F32, elem), "tensor<?x3xf32>");
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
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(4)]));
    assert_eq!(ty.render(Dtype::F32, elem), "tensor<4xf32>");
}

#[test]
fn rngstate_maps_to_key_type() {
    // The rng-state key tensor (spec §07 rng ABI) is a fixed `ui64` type,
    // independent of `Dtype` — unlike every other `Type::Scalar`/`Array`
    // mapping in this file, `MlirTy::Key`'s rendering must NOT vary with the
    // emitter's f32/f64 element dtype.
    assert_eq!(
        MlirTy::Key.render(Dtype::F32, ElemKind::Real),
        "tensor<2xui64>"
    );
    assert_eq!(
        MlirTy::Key.render(Dtype::F64, ElemKind::Real),
        "tensor<2xui64>"
    );

    let mut m = Module::new();
    let id = placeholder(&mut m, Type::RngState);
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Key);
    assert_eq!(ty.render(Dtype::F64, elem), "tensor<2xui64>");
}

#[test]
fn mlir_type_of_dtype_is_configurable_not_hardcoded() {
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::Scalar(ScalarType::Real));
    let (ty, elem) = mlir_type_of(&m, id, Dtype::F64).unwrap();
    assert_eq!(ty.render(Dtype::F64, elem), "tensor<f64>");
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
    // `Module` hits the catch-all arm (neither aggregate nor measure-layer,
    // and not `RngState` — which now maps to `MlirTy::Key`) — the refusal
    // must name the offending type via its `Debug` form, not just say "no
    // MLIR tensor form" with no detail.
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::Module);
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("type has no MLIR tensor form"));
    assert!(err.msg.contains("Module"));
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
    let out = e.finish("logdensity", &[], &[&c]);

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
        elem: ElemKind::Real,
    };
    let doubled = e.add(&arg, &arg);
    let out = e.finish("f", &[("%arg0".to_string(), MlirTy::Scalar)], &[&doubled]);
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
    let out = e.finish("f", &[], &[&cases[0].0]);
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
    let out = e.finish("f", &[], &[&r]);

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
    let out = e.finish("f", &[], &[&picked]);

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

    let out = e.finish("f", &[], &[&mx]);
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
    let out = e.finish("f", &[], &[&mx]);
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

    let out = e.finish("f", &[], &[&y]);
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
// Most of these build tiny FlatPDL fragments by hand (no parse/infer pass),
// mirroring Task 2/3's hand-built-`Module` test style — node structure and
// already-lowered operand shapes drive `Emitter::lower_node`'s dispatch.
// Since Task A2, the deterministic op map's kind-polymorphic/real-domain
// coercion (`ops::lower_builtin`) DOES consult the type side-table
// (`Emitter::node_kind`), but every fragment below never calls `set_type` on
// its call nodes, so it falls back to `ElemKind::Real` exactly as before —
// a hand-built fragment that needs a NON-`Real` result kind sets it
// explicitly (see `lower_node_mixed_int_real_add_converts_before_add`).

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
    let out = e.finish("logdensity", &[], &[&result]);

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
        ("divide", "stablehlo.divide", 2),
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
        let out = e.finish("f", &[], &[&result]);
        assert!(out.contains(op), "head '{head}': missing {op} in:\n{out}");
        assert!(is_delimiter_balanced(&out));
    }
}

/// Task A2 Step 1: `add(int_literal, real_binding)`, with the `add` node's
/// OWN inferred type set to `Real` (mixed-kind inference has already decided
/// the result, exactly as `flatppl-infer` would for `3 + mu` with `mu:
/// reals`) — must render the int literal as an `i32` tensor, insert exactly
/// one `stablehlo.convert` (the canonical `integers ⊂ reals` embedding, spec
/// §03) immediately before the `stablehlo.add` it feeds, and the add itself
/// renders `f32`. Types are set directly on the hand-built `Module`
/// (mirroring `mlir_type_of`'s `placeholder` helper) rather than run through
/// a full parse+infer pass — only the type side-table `node_kind` reads
/// matters here.
#[test]
fn lower_node_mixed_int_real_add_converts_before_add() {
    let mut m = Module::new();
    let i = int(&mut m, 3);
    m.set_type(i, Type::Scalar(ScalarType::Integer));
    let x = real(&mut m, 2.5);
    m.set_type(x, Type::Scalar(ScalarType::Real));
    let add_node = call(&mut m, "add", &[i, x]);
    m.set_type(add_node, Type::Scalar(ScalarType::Real));

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(add_node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    assert_eq!(result.elem, ElemKind::Real);
    let out = e.finish("f", &[], &[&result]);

    assert!(
        out.contains("tensor<i32>"),
        "int literal renders as an i32 scalar tensor:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.convert").count(),
        1,
        "expected exactly one stablehlo.convert (the int->real embedding):\n{out}"
    );
    let convert_pos = out.find("stablehlo.convert").expect("missing convert");
    let add_pos = out.find("stablehlo.add").expect("missing add");
    assert!(
        convert_pos < add_pos,
        "convert must precede the add it feeds:\n{out}"
    );
    let add_line = out
        .lines()
        .find(|l| l.contains("stablehlo.add"))
        .expect("missing add line");
    assert!(
        add_line.contains("tensor<f32>"),
        "add renders f32 operands:\n{add_line}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Task A2 regression: a CONTINUOUS distribution invoked with literal
/// INTEGER parameters (`Gamma(2, 1)`, fully idiomatic FlatPPL — same style as
/// `Normal(0, 1)`) must still emit well-typed StableHLO. `crate::registry`'s
/// `gamma_logpdf` calls `Emitter::log`/`Emitter::lgamma` directly on `rate`/
/// `shape` — real-only ops with no `NodeId` to coerce against an inferred
/// result kind, since registry.rs never goes through `crate::ops`'s dispatch
/// table — so every `stablehlo.log`/`chlo.lgamma` operand must be converted
/// to `f32` FIRST, never applied directly to the literals' native `i32`.
/// (Caught by running `poisson-model.flatppl`'s `Gamma(2, 1)` prior through
/// the emitter during Task A2's Step-8 numeric re-verify — no golden here
/// exercised a literal-parameter continuous distribution before.)
#[test]
fn literal_int_params_on_continuous_distribution_convert_before_real_only_ops() {
    let src = "flatppl_compat = \"0.1\"\n\
lambda = draw(Gamma(shape = 2, rate = 1))\n\
lp = logdensityof(lawof(record(lambda = lambda)), record(lambda = 2.5))\n";
    let m = flatppl_syntax::parse(src).unwrap();
    let d = flatppl_determinizer::determinize(&m).unwrap();
    let out = flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .unwrap();
    assert!(out.contains("module {") && is_delimiter_balanced(&out));
    assert!(
        out.contains("tensor<i32>"),
        "the literal shape/rate params render as i32:\n{out}"
    );
    for line in out.lines() {
        if line.contains("stablehlo.log") || line.contains("chlo.lgamma") {
            assert!(
                !line.contains("i32"),
                "a real-only op must never see an i32 operand directly \
                 (missing a convert):\n{line}\nfull module:\n{out}"
            );
        }
    }
}

/// Fix-up regression (post-A2 review): `sum` over an `Int`-typed array
/// (§07: sum-of-integers is integer) must reduce with an `Int` init constant
/// AND `Int` result — `Emitter::reduce_axis` used to hardcode `Real` for
/// both, which would emit an `Int` operand against an `f32` init/result
/// (invalid: `stablehlo.reduce`'s operand/init/result element types must all
/// agree). Hand-built (mirroring `mlir_type_of`'s `placeholder`-style tests):
/// `set_type` on the `vector(...)` call gives `ops::lower_vector` a
/// `node_kind` to convert its (already-`Int`) elements against, so this
/// exercises `reduce_axis` exactly, with no dependency on the determinizer's
/// const-fold or `modes.rs`'s free-parameter binding.
#[test]
fn sum_over_int_array_reduces_with_int_init_and_result() {
    let mut m = Module::new();
    let e1 = int(&mut m, 2);
    let e2 = int(&mut m, 3);
    let e3 = int(&mut m, 7);
    let xs = call(&mut m, "vector", &[e1, e2, e3]);
    m.set_type(
        xs,
        Type::Array {
            shape: Box::new([Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        },
    );
    let total = call(&mut m, "sum", &[xs]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(total).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    assert_eq!(result.elem, ElemKind::Int);
    let out = e.finish("f", &[], &[&result]);
    assert!(is_delimiter_balanced(&out));

    let reduce_line = out
        .lines()
        .find(|l| l.contains("stablehlo.reduce("))
        .expect("missing stablehlo.reduce line");
    assert!(
        reduce_line.contains("tensor<i32>") && !reduce_line.contains("f32"),
        "sum over an int array must reduce entirely in i32 (operand+init+result):\n{reduce_line}"
    );
    let init_line = out
        .lines()
        .find(|l| l.contains("stablehlo.constant") && l.contains("dense<0>"))
        .expect("missing the int-typed additive-identity init constant");
    assert!(
        init_line.contains("tensor<i32>"),
        "the reduce's init constant must be i32, not the old hardcoded f32:\n{init_line}"
    );
}

/// Fix-up regression (post-A2 review): an all-integer `in(k, interval(0,
/// 10))` must still emit a well-typed `stablehlo.compare`. `ops::lower_in`'s
/// `below`/`above`/`product` chain (`sub`/`mul`, kind-polymorphic) stays
/// `Int` throughout since every operand here is a literal `Int`, but
/// `lower_in`'s own `zero` constant is unconditionally `Real`
/// (`Emitter::constant(0.0, ...)`) — so `Emitter::compare` must reconcile the
/// mismatched pair (widening the `Int` product up to `Real`, per
/// `elem_rank`'s order) rather than emitting `stablehlo.compare` over a
/// declared-mismatched `(tensor<i32>, tensor<f32>)` operand pair. Every
/// operand is a plain `Lit(Int)` (not a free/bound arg), so this needs no
/// determinizer pass and cannot be const-folded (this test calls
/// `Emitter::lower_node` directly, never `flatppl_determinizer::determinize`).
#[test]
fn all_integer_in_interval_reconciles_compare_operand_kinds() {
    let mut m = Module::new();
    let k = int(&mut m, 5);
    let lo = int(&mut m, 0);
    let hi = int(&mut m, 10);
    let interval = call(&mut m, "interval", &[lo, hi]);
    let cond = call(&mut m, "in", &[k, interval]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(cond).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    assert_eq!(result.elem, ElemKind::Bool);
    let out = e.finish("f", &[], &[&result]);
    assert!(is_delimiter_balanced(&out));

    let compare_line = out
        .lines()
        .find(|l| l.contains("stablehlo.compare"))
        .expect("missing stablehlo.compare line");
    // Both operand types in the `(lhs, rhs) -> result` signature must agree
    // with each other (both widened to f32) — a `(tensor<i32>, tensor<f32>)`
    // pair would be the exact bug this test pins.
    assert!(
        !compare_line.contains("i32"),
        "compare's product operand must be widened to real (matching the zero \
         constant it's compared against), not left i32:\n{compare_line}"
    );
    // Task A2 (Bool result) must still hold.
    assert!(
        compare_line.contains("-> tensor<i1>"),
        "compare's result stays i1:\n{compare_line}"
    );
}

/// Fix-up regression (post-A2 review): an `ifelse` selecting between two
/// `Int` (literal) branches must return an `Int`-tagged `Value` whose tag
/// matches the emitted `i32` `stablehlo.select` — `Emitter::select` used to
/// hardcode its result `elem: Real` regardless of the branches' actual kind.
/// Reuses the all-`Int` `in(...)` predicate above as `ifelse`'s condition, so
/// this also exercises `lower_ifelse`/`require_predicate_head` end to end
/// over an all-integer expression.
#[test]
fn int_ifelse_select_returns_int_tagged_value() {
    let mut m = Module::new();
    let k = int(&mut m, 5);
    let lo = int(&mut m, 0);
    let hi = int(&mut m, 10);
    let interval = call(&mut m, "interval", &[lo, hi]);
    let cond = call(&mut m, "in", &[k, interval]);
    let a = int(&mut m, 3);
    let b = int(&mut m, 7);
    let ifelse_node = call(&mut m, "ifelse", &[cond, a, b]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(ifelse_node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    assert_eq!(result.elem, ElemKind::Int);
    let out = e.finish("f", &[], &[&result]);
    assert!(is_delimiter_balanced(&out));

    let select_line = out
        .lines()
        .find(|l| l.contains("stablehlo.select"))
        .expect("missing stablehlo.select line");
    assert!(
        select_line.contains("tensor<i32>") && !select_line.contains("f32"),
        "an int ifelse must emit an i32 select (on_true/on_false/result), not the \
         old hardcoded f32:\n{select_line}"
    );
}

/// Task B1 CRITICAL fix (PR-A whole-branch review, Minor #1): the
/// value-path `Emitter::compare` must append `compare_type = SIGNED` when
/// its (reconciled) operand kind is `Int` — StableHLO requires an explicit
/// signedness for an integer compare, and this emitter's integer values are
/// always signed (`i32`/`i64`). A `Real` (or `Bool`) operand pair must stay
/// byte-identical to before (no `compare_type` attribute at all) — booleans
/// must NOT carry one per the StableHLO spec, and every pre-existing caller
/// here is `Real`.
#[test]
fn compare_adds_signed_compare_type_for_int_operands_only() {
    let m = Module::new();

    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.int_value_const(3);
    let b = e.int_value_const(5);
    let cmp = e.compare("LT", &a, &b);
    let out = e.finish("f", &[], &[&cmp]);
    let line = out
        .lines()
        .find(|l| l.contains("stablehlo.compare"))
        .expect("missing compare line");
    assert!(
        line.contains(", SIGNED :"),
        "an Int compare must carry compare_type = SIGNED:\n{line}"
    );

    let mut e2 = Emitter::new(&m, Dtype::F32);
    let x = e2.scalar(1.0);
    let y = e2.scalar(2.0);
    let cmp2 = e2.compare("LT", &x, &y);
    let out2 = e2.finish("f", &[], &[&cmp2]);
    let line2 = out2
        .lines()
        .find(|l| l.contains("stablehlo.compare"))
        .expect("missing compare line");
    assert!(
        !line2.contains("SIGNED"),
        "a Real compare must stay byte-identical (no compare_type):\n{line2}"
    );
}

/// Task B1: `div(a, b) = ⌊a/b⌋` (spec §07 integer floor division) over `Int`
/// literals — mirroring the A2-fix goldens' hand-built-`Module` style, so
/// this bypasses the determiniser and cannot const-fold. Before this task
/// `div` has no arm in `ops::lower_builtin`'s dispatch and refuses via the
/// catch-all "unsupported builtin head". After: StableHLO's truncating
/// integer `divide` plus the sign-correction idiom (`compare`+`and`+
/// `select`), returning `Int` — never `stablehlo.remainder` (that's `mod`'s
/// op, not `div`'s).
#[test]
fn div_lowers_to_floor_corrected_integer_divide() {
    let mut m = Module::new();
    let a = int(&mut m, 7);
    let b = int(&mut m, 2);
    let node = call(&mut m, "div", &[a, b]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    assert_eq!(result.elem, ElemKind::Int);
    let out = e.finish("f", &[], &[&result]);
    assert!(is_delimiter_balanced(&out));

    for op in [
        "stablehlo.divide",
        "stablehlo.compare",
        "stablehlo.and",
        "stablehlo.select",
    ] {
        assert!(out.contains(op), "div: missing {op} in:\n{out}");
    }
    assert!(
        !out.contains("stablehlo.remainder"),
        "div must not use stablehlo.remainder (that's mod's op):\n{out}"
    );
}

/// Task B1: `mod(a, b) = a − b·⌊a/b⌋` (spec §07 floored modulo) over `Int`
/// literals — same hand-built-`Module` style as `div`'s golden above. Before
/// this task `mod` refuses via the same catch-all as `div`. After: StableHLO's
/// truncating integer `remainder` plus the sign-correction idiom, returning
/// `Int`.
#[test]
fn mod_lowers_to_floor_corrected_integer_remainder() {
    let mut m = Module::new();
    let a = int(&mut m, 7);
    let b = int(&mut m, 2);
    let node = call(&mut m, "mod", &[a, b]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    assert_eq!(result.elem, ElemKind::Int);
    let out = e.finish("f", &[], &[&result]);
    assert!(is_delimiter_balanced(&out));

    for op in [
        "stablehlo.remainder",
        "stablehlo.compare",
        "stablehlo.and",
        "stablehlo.select",
    ] {
        assert!(out.contains(op), "mod: missing {op} in:\n{out}");
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
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(ifelse_node).unwrap();
    let out = e.finish(
        "logdensity",
        &[("%arg0".to_string(), MlirTy::Scalar)],
        &[&result],
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
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(3)]))],
        &[&result],
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
            elem: ElemKind::Real,
        },
    );
    e.bind(
        t2,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
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
        &[&result],
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
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(3)]))],
        &[&result],
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
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(node).unwrap();
    let out = e.finish("f", &[("%arg0".to_string(), MlirTy::Scalar)], &[&result]);

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
            elem: ElemKind::Real,
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
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Scalar);
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(5)]))],
        &[&result],
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
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(node).unwrap();
    let out = e.finish(
        "f",
        &[("%arg0".to_string(), MlirTy::Ranked(vec![Some(5)]))],
        &[&result],
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
            elem: ElemKind::Real,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("rank-1"),
        "unexpected message: {}",
        err.msg
    );
}

/// A non-literal selector that ALSO isn't a runtime rank-1 `Int` tensor (here
/// a bound scalar `Real`, e.g. `get0(v, some_real_expr)`) still refuses —
/// falling through both the literal-index fast path AND the
/// `lower_get_gather` fallback (see `lower_get0_gather_lowers_runtime_index`
/// below for the case that fallback DOES accept).
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
            elem: ElemKind::Real,
        },
    );
    e.bind(
        idx,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("literal integer"),
        "unexpected message: {}",
        err.msg
    );
}

/// `get0(container, index)` with a RUNTIME rank-1 `Int` `index` (the
/// `theta[person]`-style vector-index case) lowers to `stablehlo.gather`:
/// `base = 0` (`get0`) is subtracted from `index` (a no-op numerically, but
/// still emitted — `Emitter::gather` doesn't special-case `base == 0`), the
/// result reshaped `[N] -> [N, 1]`, then gathered against the authoritative
/// JAX/XLA `dimension_numbers`/`slice_sizes` for a single-scalar-slice-per-
/// index gather along a rank-1 operand's only axis.
#[test]
fn lower_get0_gather_lowers_runtime_index() {
    let mut m = Module::new();
    let vals = local_ref(&mut m, "vals");
    let idx = local_ref(&mut m, "idx");
    let node = call(&mut m, "get0", &[vals, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        vals,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(4)]),
            elem: ElemKind::Real,
        },
    );
    e.bind(
        idx,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Int,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.ty, MlirTy::Ranked(vec![Some(3)]));
    assert_eq!(result.elem, ElemKind::Real);
    let out = e.finish(
        "f",
        &[
            ("%arg0".to_string(), MlirTy::Ranked(vec![Some(4)])),
            ("%arg1".to_string(), MlirTy::Ranked(vec![Some(3)])),
        ],
        &[&result],
    );

    assert!(
        out.contains("stablehlo.constant dense<0> : tensor<i32>"),
        "get0's base-0 constant missing in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.subtract"),
        "base subtraction missing in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.reshape") && out.contains("tensor<3x1xi32>"),
        "index reshape to Nx1 missing in:\n{out}"
    );
    assert!(
        out.contains("\"stablehlo.gather\"(%arg0,"),
        "generic-form gather over the operand missing in:\n{out}"
    );
    assert!(
        out.contains(
            "dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], \
             start_index_map = [0], index_vector_dim = 1>"
        ),
        "dimension_numbers must match the JAX/XLA reference verbatim:\n{out}"
    );
    assert!(
        out.contains("indices_are_sorted = false"),
        "missing indices_are_sorted in:\n{out}"
    );
    assert!(
        out.contains("slice_sizes = array<i64: 1>"),
        "missing slice_sizes in:\n{out}"
    );
    assert!(
        out.contains("(tensor<4xf32>, tensor<3x1xi32>) -> tensor<3xf32>"),
        "gather operand/index/result types wrong in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// `get(container, index)` (1-based) with the same runtime rank-1 `Int`
/// index subtracts `base = 1` before the gather — the dense constant, unlike
/// `get0`'s, is `1`.
#[test]
fn lower_get_gather_uses_base_one() {
    let mut m = Module::new();
    let vals = local_ref(&mut m, "vals");
    let idx = local_ref(&mut m, "idx");
    let node = call(&mut m, "get", &[vals, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        vals,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(4)]),
            elem: ElemKind::Real,
        },
    );
    e.bind(
        idx,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Int,
        },
    );
    let result = e.lower_node(node).unwrap();
    let out = e.finish(
        "f",
        &[
            ("%arg0".to_string(), MlirTy::Ranked(vec![Some(4)])),
            ("%arg1".to_string(), MlirTy::Ranked(vec![Some(3)])),
        ],
        &[&result],
    );
    assert!(
        out.contains("stablehlo.constant dense<1> : tensor<i32>"),
        "get's base-1 constant missing in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// An `Int` operand (e.g. an integer-valued array) gathers to an `Int`
/// result — the gather's result `elem` copies the OPERAND's, not the index's.
#[test]
fn lower_get_gather_preserves_int_operand_elem() {
    let mut m = Module::new();
    let vals = local_ref(&mut m, "vals");
    let idx = local_ref(&mut m, "idx");
    let node = call(&mut m, "get", &[vals, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        vals,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(4)]),
            elem: ElemKind::Int,
        },
    );
    e.bind(
        idx,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Int,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(result.elem, ElemKind::Int);
    let out = e.finish(
        "f",
        &[
            ("%arg0".to_string(), MlirTy::Ranked(vec![Some(4)])),
            ("%arg1".to_string(), MlirTy::Ranked(vec![Some(3)])),
        ],
        &[&result],
    );
    assert!(
        out.contains("(tensor<4xi32>, tensor<3x1xi32>) -> tensor<3xi32>"),
        "Int operand must gather to an Int result, in:\n{out}"
    );
}

/// A runtime index that is `Real`, not `Int`, refuses (spec §07 `get`/`get0`
/// selectors are integer-valued) — the `lower_get_gather` fallback's own
/// elem-kind check, distinct from the container-shape check
/// `lower_get0_refuses_non_rank1_container` exercises.
#[test]
fn lower_get_gather_refuses_non_int_index() {
    let mut m = Module::new();
    let vals = local_ref(&mut m, "vals");
    let idx = local_ref(&mut m, "idx");
    let node = call(&mut m, "get", &[vals, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        vals,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(4)]),
            elem: ElemKind::Real,
        },
    );
    e.bind(
        idx,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Real,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("literal integer"),
        "unexpected message: {}",
        err.msg
    );
}

/// A rank-2+ operand with a runtime `Int` index refuses rather than
/// mislowering a multi-dimensional gather (out of scope: only a rank-1
/// operand is supported).
#[test]
fn lower_get_gather_refuses_rank2_operand() {
    let mut m = Module::new();
    let vals = local_ref(&mut m, "vals");
    let idx = local_ref(&mut m, "idx");
    let node = call(&mut m, "get", &[vals, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        vals,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(4), Some(4)]),
            elem: ElemKind::Real,
        },
    );
    e.bind(
        idx,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Int,
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
            elem: ElemKind::Real,
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
    let out = e.finish("f", &[], &[&result]);

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

/// Task A2: an int/bool literal lowers as a rank-0 tensor at its OWN kind —
/// `Int`/`Bool`, not `Real` — regardless of any inferred type (`Node::Lit`
/// dispatch never consults the type side-table, only the literal's own
/// `Scalar` tag; see [`Emitter::int_value_const`]/[`Emitter::bool_value_const`]).
#[test]
fn lower_node_lowers_int_and_bool_literals_as_scalars() {
    let mut m = Module::new();
    let i = int(&mut m, 7);
    let b = m.alloc(Node::Lit(Scalar::Bool(true)));
    let mut e = Emitter::new(&m, Dtype::F32);
    let iv = e.lower_node(i).unwrap();
    let bv = e.lower_node(b).unwrap();
    assert_eq!(iv.ty, MlirTy::Scalar);
    assert_eq!(iv.elem, ElemKind::Int);
    assert_eq!(bv.ty, MlirTy::Scalar);
    assert_eq!(bv.elem, ElemKind::Bool);
    let out = e.finish("f", &[], &[&iv, &bv]);
    assert!(out.contains("dense<7> : tensor<i32>"));
    assert!(out.contains("dense<true> : tensor<i1>"));
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
    let out = e.finish("f", &[], &[&result]);
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

/// `builtin_logdensityof(Bogus, ..., v)` — a distribution with no registry
/// entry — must refuse precisely (refuse-don't-mislower), not panic or guess
/// a lowering. `Bogus` (not a real §08/§09/§12/§13 constructor name) is used
/// rather than a real not-yet-registered distribution so this test stays
/// stable as later tasks register more of them.
#[test]
fn builtin_logdensityof_refuses_unregistered_ctor() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Bogus");
    let field_val = real(&mut m, 0.0);
    let kernel_input = record_node(&mut m, &[("x0", field_val)]);
    let v = real(&mut m, 1.0);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no lowering for distribution 'Bogus'"),
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

/// With `EmitOptions::query` naming the density binding, `emit_logdensity`
/// emits THAT binding even though an inert binding (`diag`) sorts after it in
/// source order. This is the cross-module-grafting case (a `load_module` query
/// scoring a foreign `posterior`): determinization splices the foreign model's
/// data / pinned-draw residue in after the query, so the query's position is
/// not stable but its name is. The SAME module refuses without the designation
/// — see `emit_logdensity_refuses_trailing_binding_with_no_density_term`.
#[test]
fn emit_logdensity_designated_query_skips_trailing_binding() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let mu = real(&mut m, 0.0);
    let sigma = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("sigma", sigma)]);
    let v = real(&mut m, 0.5);
    let density = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);
    top_level(&mut m, "lp", density);

    // A diagnostic/auxiliary binding after `lp` in source order — exactly the
    // shape that makes the positional (`query: None`) path refuse.
    let diag = real(&mut m, 42.0);
    top_level(&mut m, "diag", diag);

    let opts = flatppl_stablehlo::EmitOptions {
        query: Some("lp".to_string()),
        ..Default::default()
    };
    let out = flatppl_stablehlo::emit(&m, flatppl_stablehlo::Mode::LogDensity, &opts)
        .expect("designated query `lp` emits despite the trailing `diag` binding");
    // The standard-normal `logpdf` normalizing constant `-0.5*ln(2π)` — the
    // structural signal that `lp`'s Normal density (not `diag = 42`) was lowered.
    assert!(
        out.contains("-0.9189385332046727"),
        "expected the Normal logpdf normalizing constant, got:\n{out}"
    );
}

/// An `EmitOptions::query` naming a binding that is not a public binding of the
/// (determinized) module refuses with a precise message rather than silently
/// falling back to a positional guess — a mis-designation by the host is a bug
/// to surface, not to paper over.
#[test]
fn emit_logdensity_refuses_unknown_designated_query() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let mu = real(&mut m, 0.0);
    let sigma = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("sigma", sigma)]);
    let v = real(&mut m, 0.5);
    let density = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);
    top_level(&mut m, "lp", density);

    let opts = flatppl_stablehlo::EmitOptions {
        query: Some("nope".to_string()),
        ..Default::default()
    };
    let err = flatppl_stablehlo::emit(&m, flatppl_stablehlo::Mode::LogDensity, &opts).unwrap_err();
    assert!(
        err.msg
            .contains("designated logdensity query binding `nope` is not a public binding"),
        "unexpected message: {}",
        err.msg
    );
}

// ---- Task 8: location-scale continuous `@logdensity` batch -----------------
//
// Cauchy/Logistic/Laplace (§08), registered alongside Normal in
// `registry.rs`'s `REGISTRY` with `sample: None` (samplers land in Task 14).
// Same anchor-fixture shape as `NORMAL_DENSITY_SRC` above: free
// (`elementof`-declared) parameters, scored at a pinned observation via
// `logdensityof(lawof(record(...)), record(...))`.

const CAUCHY_DENSITY_SRC: &str = "\
location = elementof(reals)
scale = elementof(posreals)
a = draw(Cauchy(location = location, scale = scale))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const LOGISTIC_DENSITY_SRC: &str = "\
mu = elementof(reals)
s = elementof(posreals)
a = draw(Logistic(mu = mu, s = s))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const LAPLACE_DENSITY_SRC: &str = "\
location = elementof(reals)
scale = elementof(posreals)
a = draw(Laplace(location = location, scale = scale))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

/// §08 Cauchy, verbatim: `-log(pi) - log(scale) - log(1 + ((x -
/// location)/scale)^2)`. Op counts: two `log`s (scale, and the `1 + z^2`
/// term), two `negate`s (each log's negation), one `subtract` (`x -
/// location`), one `divide` (`/scale`), one `multiply` (`z * z`), three
/// `add`s (`1 + z^2`, and the two outer sums). No `chlo.*` needed.
#[test]
fn emit_logdensity_cauchy_has_expected_structure() {
    let d = determinize_src(CAUCHY_DENSITY_SRC);
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
        "location/scale must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        2,
        "expected exactly two negates, in:\n{out}"
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
        1,
        "expected exactly one multiply, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Cauchy needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_cauchy_matches_frozen_golden() {
    let d = determinize_src(CAUCHY_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/cauchy_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/cauchy_logdensity.mlir)"
    );
}

/// §08 Logistic, verbatim: with `u = (x - mu)/s`, `-u - log(s) -
/// 2*log(1 + exp(-u))`. Op counts: one `subtract` (`x - mu`), one `divide`
/// (`/s`), three `negate`s (`-u`, `-log(s)`, the final `-2*log(...)`), two
/// `log`s (`log(s)`, `log(1 + exp(-u))`), one `exponential` (`exp(-u)`), one
/// `multiply` (`2 * log(...)`), three `add`s (`1 + exp(-u)`, and the two
/// outer sums). No `chlo.*` needed.
#[test]
fn emit_logdensity_logistic_has_expected_structure() {
    let d = determinize_src(LOGISTIC_DENSITY_SRC);
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
        "mu/s must become func args, in:\n{out}"
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
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.exponential").count(),
        1,
        "expected exactly one exponential, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        1,
        "expected exactly one multiply, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Logistic needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_logistic_matches_frozen_golden() {
    let d = determinize_src(LOGISTIC_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/logistic_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/logistic_logdensity.mlir)"
    );
}

/// §08 Laplace, verbatim: `-log(2*scale) - |x - location|/scale`. Op
/// counts: one `multiply` (`2 * scale`), one `log` (`log(2*scale)`), two
/// `negate`s (`-log(2*scale)`, the final `-|.../scale`), one `subtract` (`x -
/// location`), one `abs`, one `divide` (`/scale`), one `add` (the final
/// sum). No `chlo.*` needed.
#[test]
fn emit_logdensity_laplace_has_expected_structure() {
    let d = determinize_src(LAPLACE_DENSITY_SRC);
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
        "location/scale must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        1,
        "expected exactly one multiply, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        1,
        "expected exactly one log, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        2,
        "expected exactly two negates, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        1,
        "expected exactly one subtract, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.abs").count(),
        1,
        "expected exactly one abs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.divide").count(),
        1,
        "expected exactly one divide, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        1,
        "expected exactly one add, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Laplace needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_laplace_matches_frozen_golden() {
    let d = determinize_src(LAPLACE_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/laplace_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/laplace_logdensity.mlir)"
    );
}

// ---- Task 9: gamma-family / positive-support continuous `@logdensity` batch
//
// Exponential/Gamma/Weibull/Pareto/InverseGamma/ChiSquared/LogNormal (§08),
// registered alongside Normal/Cauchy/Logistic/Laplace in `registry.rs`'s
// `REGISTRY` with `sample: None` (samplers land in Task 14). Same
// anchor-fixture shape as `NORMAL_DENSITY_SRC`/`CAUCHY_DENSITY_SRC` above:
// free (`elementof`-declared) parameters, scored at a pinned observation via
// `logdensityof(lawof(record(...)), record(...))`. A drawn value's type is
// always `scalar(Real)` regardless of the distribution's *support* (§08's
// "Domain/Support" column lists `reals` as the domain for every one of these
// — `posreals`/`nonnegreals` is the support, a density-positivity region, not
// the type), so `record(a = 0.5)` type-checks against every one of them
// exactly as it did for Cauchy/Logistic/Laplace.

const EXPONENTIAL_DENSITY_SRC: &str = "\
rate = elementof(posreals)
a = draw(Exponential(rate = rate))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const GAMMA_DENSITY_SRC: &str = "\
shape = elementof(posreals)
rate = elementof(posreals)
a = draw(Gamma(shape = shape, rate = rate))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const WEIBULL_DENSITY_SRC: &str = "\
shape = elementof(posreals)
scale = elementof(posreals)
a = draw(Weibull(shape = shape, scale = scale))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const PARETO_DENSITY_SRC: &str = "\
shape = elementof(posreals)
scale = elementof(posreals)
a = draw(Pareto(shape = shape, scale = scale))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const INVERSE_GAMMA_DENSITY_SRC: &str = "\
shape = elementof(posreals)
scale = elementof(posreals)
a = draw(InverseGamma(shape = shape, scale = scale))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const CHI_SQUARED_DENSITY_SRC: &str = "\
k = elementof(posreals)
a = draw(ChiSquared(k = k))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const LOGNORMAL_DENSITY_SRC: &str = "\
mu = elementof(reals)
sigma = elementof(posreals)
a = draw(LogNormal(mu = mu, sigma = sigma))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

/// §08 Exponential, verbatim: `log(rate) - rate * x`. Op counts: one `log`
/// (`rate`), one `multiply` (`rate * x`), one `negate`, one `add`. No
/// `chlo.*` needed.
#[test]
fn emit_logdensity_exponential_has_expected_structure() {
    let d = determinize_src(EXPONENTIAL_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "rate must become a func arg, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        1,
        "expected exactly one log, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        1,
        "expected exactly one multiply, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        2,
        "expected exactly two negates (1 formula + 1 off-support -inf)"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        1,
        "expected exactly one add, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Exponential needs no CHLO ops, in:\n{out}"
    );
    // Constrained support (§08 support `nonnegreals`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_exponential_matches_frozen_golden() {
    let d = determinize_src(EXPONENTIAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/exponential_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/exponential_logdensity.mlir)"
    );
}

/// §08 Gamma, verbatim: `shape * log(rate) - lgamma(shape) + (shape - 1) *
/// log(x) - rate * x`. Op counts: two `log`s (`rate`, `x`), one `chlo.lgamma`
/// (`shape`), two `negate`s (`-lgamma(shape)`, `-rate*x`), one `subtract`
/// (`shape - 1`), three `multiply`s (`shape*log(rate)`, `(shape-1)*log(x)`,
/// `rate*x`), three `add`s.
#[test]
fn emit_logdensity_gamma_has_expected_structure() {
    let d = determinize_src(GAMMA_DENSITY_SRC);
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
        "shape/rate must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("chlo.lgamma").count(),
        1,
        "expected exactly one lgamma, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates (2 formula + 1 off-support -inf)"
    );
    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        1,
        "expected exactly one subtract, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        3,
        "expected exactly three multiplies, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    // Constrained support (§08 support `posreals`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_gamma_matches_frozen_golden() {
    let d = determinize_src(GAMMA_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/gamma_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/gamma_logdensity.mlir)"
    );
}

/// §08 Weibull, verbatim: with `u = x/scale`, `log(shape) - log(scale) +
/// (shape - 1) * log(u) - u^shape`. Op counts: three `log`s (`shape`,
/// `scale`, `u`), two `negate`s (`-log(scale)`, `-u^shape`), one `divide`
/// (`u`), one `subtract` (`shape - 1`), one `multiply` (`(shape-1)*log(u)`),
/// one `power` (`u^shape`), three `add`s. No `chlo.*` needed.
#[test]
fn emit_logdensity_weibull_has_expected_structure() {
    let d = determinize_src(WEIBULL_DENSITY_SRC);
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
        "shape/scale must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        3,
        "expected exactly three logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates (2 formula + 1 off-support -inf)"
    );
    assert_eq!(
        out.matches("stablehlo.divide").count(),
        1,
        "expected exactly one divide, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        1,
        "expected exactly one subtract, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        1,
        "expected exactly one multiply, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.power").count(),
        1,
        "expected exactly one power, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Weibull needs no CHLO ops, in:\n{out}"
    );
    // Constrained support (§08 support `nonnegreals`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_weibull_matches_frozen_golden() {
    let d = determinize_src(WEIBULL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/weibull_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/weibull_logdensity.mlir)"
    );
}

/// §08 Pareto, verbatim: `log(shape) + shape * log(scale) - (shape + 1) *
/// log(x)`. Op counts: three `log`s (`shape`, `scale`, `x`), one `negate`
/// (the trailing term), one `add` for `shape + 1`, two `multiply`s
/// (`shape*log(scale)`, `(shape+1)*log(x)`), three `add`s total (including
/// `shape + 1`). No `chlo.*` needed.
#[test]
fn emit_logdensity_pareto_has_expected_structure() {
    let d = determinize_src(PARETO_DENSITY_SRC);
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
        "shape/scale must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        3,
        "expected exactly three logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        2,
        "expected exactly two negates (1 formula + 1 off-support -inf)"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        2,
        "expected exactly two multiplies, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "Pareto needs no CHLO ops, in:\n{out}"
    );
    // Constrained support (§08 support `x >= scale`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_pareto_matches_frozen_golden() {
    let d = determinize_src(PARETO_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/pareto_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/pareto_logdensity.mlir)"
    );
}

/// §08 InverseGamma, verbatim: `shape * log(scale) - lgamma(shape) - (shape +
/// 1) * log(x) - scale / x`. Op counts: two `log`s (`scale`, `x`), one
/// `chlo.lgamma` (`shape`), three `negate`s (`-lgamma(shape)`,
/// `-(shape+1)*log(x)`, `-scale/x`), one `divide` (`scale/x`), two
/// `multiply`s (`shape*log(scale)`, `(shape+1)*log(x)`), four `add`s
/// (including `shape + 1`).
#[test]
fn emit_logdensity_inverse_gamma_has_expected_structure() {
    let d = determinize_src(INVERSE_GAMMA_DENSITY_SRC);
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
        "shape/scale must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("chlo.lgamma").count(),
        1,
        "expected exactly one lgamma, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        4,
        "expected exactly four negates (3 formula + 1 off-support -inf)"
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
        4,
        "expected exactly four adds, in:\n{out}"
    );
    // Constrained support (§08 support `posreals`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_inverse_gamma_matches_frozen_golden() {
    let d = determinize_src(INVERSE_GAMMA_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/inverse_gamma_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/inverse_gamma_logdensity.mlir)"
    );
}

/// §08 ChiSquared, verbatim: with `half_k = k/2`, `-half_k * log(2) -
/// lgamma(half_k) + (half_k - 1) * log(x) - x/2`. `log(2)` folds to a scalar
/// literal (no `stablehlo.log` op for it — see `chi_squared_logpdf`'s doc
/// comment). Op counts: one `log` (`x`), one `chlo.lgamma` (`half_k`), three
/// `negate`s, one `subtract` (`half_k - 1`), one `divide` (`x/2`), three
/// `multiply`s (`half_k = k*0.5`, `half_k*log(2)`, `(half_k-1)*log(x)`),
/// three `add`s.
#[test]
fn emit_logdensity_chi_squared_has_expected_structure() {
    let d = determinize_src(CHI_SQUARED_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "k must become a func arg, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        1,
        "expected exactly one log, in:\n{out}"
    );
    assert_eq!(
        out.matches("chlo.lgamma").count(),
        1,
        "expected exactly one lgamma, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        4,
        "expected exactly four negates (3 formula + 1 off-support -inf)"
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
        3,
        "expected exactly three multiplies, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    // Constrained support (§08 support `posreals`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_chi_squared_matches_frozen_golden() {
    let d = determinize_src(CHI_SQUARED_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/chi_squared_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/chi_squared_logdensity.mlir)"
    );
}

/// §08 LogNormal, verbatim: `-log(x) - log(sigma) - 1/2*log(2*pi) -
/// (log(x) - mu)^2/(2*sigma^2)`. Op counts: two `log`s (`x`, `sigma` — `x`'s
/// single `log` [`Value`] is reused for both the leading term and the
/// quadratic's `z`, see `lognormal_logpdf`'s doc comment), two `negate`s
/// (`-log(x)`, `-log(sigma)`), one `subtract` (`log(x) - mu`), one `divide`
/// (`/sigma`), two `multiply`s (`z*z`, `-0.5*z^2`), three `add`s. No
/// `chlo.*` needed.
#[test]
fn emit_logdensity_lognormal_has_expected_structure() {
    let d = determinize_src(LOGNORMAL_DENSITY_SRC);
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
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates (2 formula + 1 off-support -inf)"
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
        3,
        "expected exactly three adds, in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "LogNormal needs no CHLO ops, in:\n{out}"
    );
    // Constrained support (§08 support `posreals`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_lognormal_matches_frozen_golden() {
    let d = determinize_src(LOGNORMAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/lognormal_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/lognormal_logdensity.mlir)"
    );
}

// ---- Task 10: remaining univariate continuous `@logdensity` batch ----------
//
// Uniform/Beta/StudentT/GeneralizedNormal/VonMises (§08), registered
// alongside the rest of §08 in `registry.rs`'s `REGISTRY` with `sample: None`
// (samplers land in Task 14). Same anchor-fixture shape as the Task 8/9
// batches above, EXCEPT Uniform: its `support` is a literal `interval(lo,
// hi)` set expression (not an `elementof`-declared scalar parameter), so
// `a`'s Uniform draw has no free parameters at all — `emit_logdensity`
// produces a zero-arg `func.func @logdensity()`.

const UNIFORM_DENSITY_SRC: &str = "\
a = draw(Uniform(support = interval(-1.0, 3.0)))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const BETA_DENSITY_SRC: &str = "\
alpha = elementof(posreals)
beta = elementof(posreals)
a = draw(Beta(alpha = alpha, beta = beta))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const STUDENTT_DENSITY_SRC: &str = "\
nu = elementof(posreals)
a = draw(StudentT(nu = nu))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const GENERALIZED_NORMAL_DENSITY_SRC: &str = "\
mean = elementof(reals)
alpha = elementof(posreals)
beta = elementof(posreals)
a = draw(GeneralizedNormal(mean = mean, alpha = alpha, beta = beta))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const VON_MISES_DENSITY_SRC: &str = "\
mu = elementof(reals)
kappa = elementof(posreals)
a = draw(VonMises(mu = mu, kappa = kappa))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

/// §08 Uniform, verbatim: `-log(lambda(S))`, a compile-time constant once
/// `S = interval(-1.0, 3.0)`'s length (`4.0`) is known — `a` has no free
/// parameters at all, so `func.func @logdensity` takes NO args (a distinct
/// shape from every other Task 8/9/10 fixture, all of which have at least
/// one `elementof`-declared parameter). Exactly two `stablehlo.constant`s:
/// the pinned observation (`registry::lower_logdensityof` always lowers `v`
/// up front, even though [`registry::uniform_logpdf`] itself never reads
/// it — see its doc comment) and the folded `-log(4.0)` — no other op.
#[test]
fn emit_logdensity_uniform_has_expected_structure() {
    let d = determinize_src(UNIFORM_DENSITY_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_logdensity(&d);

    assert!(
        out.contains("func.func @logdensity()"),
        "missing func.func @logdensity() (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> tensor<f32>"),
        "must return tensor<f32> in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.constant").count(),
        2,
        "expected exactly two constants (the pinned observation, and the folded -log(4.0)), in:\n{out}"
    );
    assert!(
        out.contains("dense<-1.3862943611198906>"),
        "expected the folded -log(4.0) literal, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_uniform_matches_frozen_golden() {
    let d = determinize_src(UNIFORM_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/uniform_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/uniform_logdensity.mlir)"
    );
}

/// `Uniform(support = reals)` — an unbounded set, infinite Lebesgue
/// measure — must refuse with the exact message `registry::lebesgue_measure`
/// promises, rather than lowering a nonsensical `-log(inf)`.
#[test]
fn uniform_logpdf_refuses_unbounded_support() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Uniform");
    let support = const_node(&mut m, "reals");
    let kernel_input = record_node(&mut m, &[("support", support)]);
    let v = real(&mut m, 0.5);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    // No parse/infer pass ran over this hand-built fragment, so `support`
    // has no `valueset_of` entry at all (`None`) — exactly the same refusal
    // path as an inferred-but-`Unknown` set (e.g. non-literal bounds).
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("Uniform logpdf needs a measurable interval/box support"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(support));
}

/// §08 Beta, verbatim: `(alpha - 1) * log(x) + (beta - 1) * log(1 - x) -
/// [lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)]`.
#[test]
fn emit_logdensity_beta_has_expected_structure() {
    let d = determinize_src(BETA_DENSITY_SRC);
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
        "alpha/beta must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        3,
        "expected exactly three subtracts, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        2,
        "expected exactly two multiplies, in:\n{out}"
    );
    assert_eq!(
        out.matches("chlo.lgamma").count(),
        3,
        "expected exactly three lgammas, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates (2 formula + 1 off-support -inf)"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        5,
        "expected exactly five adds, in:\n{out}"
    );
    // Constrained support (§08 support `unitinterval`): masked to `-inf`
    // off-support — a `compare` guards the variate into the support and two
    // `select`s (guard, then off-support pick) wrap the formula, with `-inf`
    // the negated `+inf` (`0x7F800000`) bit pattern. The in-support formula
    // op counts above are unchanged (numerically identical in-support).
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "expected exactly two selects (variate guard + off-support mask), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.constant dense<0x7F800000>"),
        "off-support -inf floor (negated +inf) missing, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_beta_matches_frozen_golden() {
    let d = determinize_src(BETA_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/beta_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/beta_logdensity.mlir)"
    );
}

/// §08 StudentT, verbatim: with `half_nu_plus_one = (nu + 1) / 2`,
/// `lgamma(half_nu_plus_one) - 1/2 * log(nu * pi) - lgamma(nu / 2) -
/// half_nu_plus_one * log(1 + x^2 / nu)`.
#[test]
fn emit_logdensity_studentt_has_expected_structure() {
    let d = determinize_src(STUDENTT_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "nu must become a func arg, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        5,
        "expected exactly five adds, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        6,
        "expected exactly six multiplies, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("chlo.lgamma").count(),
        2,
        "expected exactly two lgammas, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.divide").count(),
        1,
        "expected exactly one divide, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_studentt_matches_frozen_golden() {
    let d = determinize_src(STUDENTT_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/studentt_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/studentt_logdensity.mlir)"
    );
}

/// §08 GeneralizedNormal, verbatim: `log(beta) - log(2 * alpha) -
/// lgamma(1 / beta) - (|x - mean| / alpha)^beta`.
#[test]
fn emit_logdensity_generalized_normal_has_expected_structure() {
    let d = determinize_src(GENERALIZED_NORMAL_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>")
            && out.contains("%arg1: tensor<f32>")
            && out.contains("%arg2: tensor<f32>"),
        "mean/alpha/beta must become func args, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        2,
        "expected exactly two logs, in:\n{out}"
    );
    assert_eq!(
        out.matches("chlo.lgamma").count(),
        1,
        "expected exactly one lgamma, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.negate").count(),
        3,
        "expected exactly three negates, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.multiply").count(),
        1,
        "expected exactly one multiply, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.divide").count(),
        2,
        "expected exactly two divides, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.abs").count(),
        1,
        "expected exactly one abs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.subtract").count(),
        1,
        "expected exactly one subtract, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.power").count(),
        1,
        "expected exactly one power, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.add").count(),
        3,
        "expected exactly three adds, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_generalized_normal_matches_frozen_golden() {
    let d = determinize_src(GENERALIZED_NORMAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/generalized_normal_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden \
         (tests/goldens/generalized_normal_logdensity.mlir)"
    );
}

/// §08 VonMises, verbatim: `kappa * cos(x - mu) - log(2*pi) -
/// log(I_0(kappa))`. `log(I_0(kappa))` is `registry::log_bessel_i0`'s
/// inlined Abramowitz & Stegun approximation: a `stablehlo.select` between
/// two `stablehlo.compare LT`-branches (small-`kappa`/large-`kappa`, each a
/// Horner-scheme polynomial), never a `chlo.bessel*` op (no such op exists).
#[test]
fn emit_logdensity_von_mises_has_expected_structure() {
    let d = determinize_src(VON_MISES_DENSITY_SRC);
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
        "mu/kappa must become func args, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.cosine"),
        "missing cosine, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.compare LT").count(),
        1,
        "expected exactly one small/large-kappa branch compare, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.select").count(),
        1,
        "expected exactly one select (the log-I0 branch), in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log").count(),
        3,
        "expected exactly three logs (small-branch log, large-branch log, log(kappa)), in:\n{out}"
    );
    assert!(
        !out.contains("chlo."),
        "VonMises needs no CHLO op (no chlo.bessel* exists), in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula, or the A&S polynomial coefficients themselves) must be a
/// deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_von_mises_matches_frozen_golden() {
    let d = determinize_src(VON_MISES_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/von_mises_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/von_mises_logdensity.mlir)"
    );
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
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing func.func @sample(%key: tensor<2xui64>) (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<f32>, tensor<2xui64>)"),
        "must return the (value, advanced-key) pair in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "expected exactly one threaded rng_bit_generator draw, in:\n{out}"
    );
    assert!(
        out.contains("chlo.erf_inv"),
        "NORMAL draws through the erf_inv probit, in:\n{out}"
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

/// `builtin_sample(rng, Bogus, kernel_input)` — a ctor with no registry
/// entry at all — must refuse precisely, not panic or guess a lowering.
/// This exercises the same `registry::lookup` miss
/// `builtin_logdensityof_refuses_unregistered_ctor` does (shared code, not
/// sample-specific text) — distinct from
/// `builtin_sample_refuses_registered_ctor_without_sample_builder` below,
/// which exercises a ctor that IS registered but has no `@sample` builder.
/// `Bogus` (not a real §08/§09/§12/§13 constructor name) is used rather than
/// a real not-yet-registered distribution so this test stays stable as later
/// tasks register more of them.
#[test]
fn builtin_sample_refuses_unregistered_ctor() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0); // stand-in rng-state arg (never lowered)
    let ctor = const_node(&mut m, "Bogus");
    let mu_val = real(&mut m, 0.0);
    let sigma_val = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu_val), ("sigma", sigma_val)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no lowering for distribution 'Bogus'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `builtin_sample(rng, VonMises, kernel_input)` — `VonMises` IS registered
/// (Task 10's `@logdensity` builder) but has no `@sample` builder yet
/// (`sample: None`: VonMises needs a dedicated rejection sampler, e.g. Best &
/// Fisher — Task 15's rejection batch covers Gamma/Beta/ChiSquared/StudentT/
/// InverseGamma/GeneralizedNormal + Dirichlet, but not VonMises) — must refuse
/// with `lower_sample`'s OWN sample-specific message, `"no @sample lowering
/// for '{ctor}'"` (`dist.sample.ok_or_else`), distinct from the
/// unregistered-ctor message above. This arm stays reachable via any of the
/// registry's still-`sample: None` entries (originally exercised via `Cauchy`,
/// before Task 14 gave Cauchy a sampler; then `Gamma`, before Task 15 gave the
/// rejection batch theirs).
#[test]
fn builtin_sample_refuses_registered_ctor_without_sample_builder() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0); // stand-in rng-state arg (never lowered)
    let ctor = const_node(&mut m, "VonMises");
    let mu = real(&mut m, 0.0);
    let kappa = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("kappa", kappa)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no @sample lowering for 'VonMises'"),
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
/// the exact expected/actual count), not panic on the arg-count match. The
/// scalar form takes 3 args and the fanned iid form 4 (spec §07 size dims),
/// so a 2-arg call refuses with the "3 or 4 arguments" message.
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
            .contains("builtin_sample: expected 3 or 4 arguments, got 2"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

// ---- Task 7: refuse taxonomy — closing coverage gaps -----------------------
//
// Task 7's audit (see `crates/stablehlo/src/refuse.rs`'s module doc comment
// for the full enumerated taxonomy) found every `EmitError` construction site
// already covered by a Task 2/4/5/6 test EXCEPT the ones below — each new
// test here locks exactly one previously-untested site, with no duplication
// of an existing case. `registry.rs`'s "no @sample lowering for '{ctor}'"
// site (a registered ctor with `sample: None`) was untested at Task 7 time:
// it was genuinely unreachable then (only `Normal` was registered, and it has
// `sample: Some(_)`). Task 8 registers `Cauchy`/`Logistic`/`Laplace` with
// `sample: None`, making the site reachable — see
// `builtin_sample_refuses_registered_ctor_without_sample_builder` above,
// alongside the Task 6-review `lower_sample` refuse tests it extends.

/// `mlir_type_of` on a node with no inferred type at all (never
/// `set_type`-ed) — a distinct site from the aggregate/measure-layer/catch-all
/// refusals below it in `types.rs`, all of which require a type to already be
/// present in the side table.
#[test]
fn mlir_type_of_refuses_node_with_no_inferred_type() {
    let mut m = Module::new();
    let id = m.alloc(Node::Lit(Scalar::Real(0.0))); // no `m.set_type` call
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(
        err.msg.contains("no inferred type"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(id));
}

/// `emit`'s own up-front `is_flatpdl` gate (`lib.rs`) — a module still
/// carrying a residual measure-layer type must refuse with the module-level
/// "not FlatPDL" message (`EmitError::whole`, `node: None`), distinct from
/// `mlir_type_of`'s own (node-localized) "residual measure-layer type"
/// refusal: this one is reached before the emitter ever starts walking the
/// query, on `flatppl_determinizer::is_flatpdl`'s own conformance check.
#[test]
fn emit_refuses_input_that_is_not_flatpdl() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Measure {
            domain: Box::new(Type::Scalar(ScalarType::Real)),
            mass: Mass::Normalized,
        },
    );
    top_level(&mut m, "x", id);

    let err = flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("is not FlatPDL"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(
        err.node, None,
        "module-level refusal has no localizing node"
    );
}

/// `emit_logdensity` on a module with no public binding at all (not even a
/// trailing non-density one) — distinct from
/// `emit_logdensity_refuses_trailing_binding_with_no_density_term`, which
/// exercises the query-CONTENT guard on a module that DOES have a public
/// binding.
#[test]
fn emit_logdensity_refuses_module_with_no_public_binding() {
    let m = Module::new();
    let err = flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("no public binding to emit as the logdensity query"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, None);
}

/// The `emit_sample` mirror of
/// [`emit_logdensity_refuses_module_with_no_public_binding`].
#[test]
fn emit_sample_refuses_module_with_no_public_binding() {
    let m = Module::new();
    let err = flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("no public binding to emit as the sample query"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, None);
}

/// `get0(builtin_sample(...), 1)` — projecting the ADVANCED RNG-STATE slot
/// (index 1) of a sampled `(value, new_rngstate)` pair (spec §07), as opposed
/// to the drawn-value slot (index 0, the ordinary case every other sample
/// test projects). Under the threaded-key rng ABI this slot resolves to the
/// key `stablehlo.rng_bit_generator` advanced (typed `MlirTy::Key`) — the
/// mechanism a chained `rand` threads onward — rather than refusing.
#[test]
fn lower_get_of_sampled_tuple_yields_advanced_rng_key() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0); // stand-in rng source (bound to the key below)
    let ctor = const_node(&mut m, "Normal");
    let mu = real(&mut m, 0.0);
    let sigma = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("mu", mu), ("sigma", sigma)]);
    let sample = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);
    let one_idx = int(&mut m, 1);
    let node = call(&mut m, "get0", &[sample, one_idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    // Seed the source rng arg with `%key`, as `emit_sample` would.
    e.bind(
        rng,
        Value {
            ssa: "%key".to_string(),
            ty: MlirTy::Key,
            elem: ElemKind::Real,
        },
    );
    let v = e
        .lower_node(node)
        .expect("advanced rng-state slot must project to the advanced key");
    assert_eq!(
        v.ty,
        MlirTy::Key,
        "slot 1 is the advanced rng-state key, not a tensor value"
    );
}

/// `vector()` with zero elements — `concatenate` needs at least one operand;
/// refuses rather than asserting inside `Emitter::vector`.
#[test]
fn lower_vector_refuses_empty_element_list() {
    let mut m = Module::new();
    let node = call(&mut m, "vector", &[]);
    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("vector: expected at least one element"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// A `vector(...)` whose elements are themselves rank-1 tensors (not
/// scalars) — a vector-of-vectors (spec §03: legal, distinct from a matrix,
/// since matrices only come from `array`/`rowstack`/`colstack`/`eye`) —
/// must lower to a rank-2 tensor via reshape-then-concatenate at the
/// element's own rank, not silently truncate to rank-1 by assuming a scalar
/// element (the confirmed Task-13-review `Emitter::vector` mis-lowering
/// bug: `reshape` performs no validation, so a non-scalar element used to
/// reshape down to a wrong-rank `tensor<1x…>` without ever refusing).
#[test]
fn lower_vector_of_vectors_lowers_to_rank2_tensor() {
    let mut m = Module::new();
    let t1 = local_ref(&mut m, "t1");
    let t2 = local_ref(&mut m, "t2");
    let node = call(&mut m, "vector", &[t1, t2]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        t1,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Real,
        },
    );
    e.bind(
        t2,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Real,
        },
    );
    let result = e.lower_node(node).unwrap();
    assert_eq!(
        result.ty,
        MlirTy::Ranked(vec![Some(2), Some(3)]),
        "vector-of-two-length-3-vectors must be rank-2 [2, 3], not rank-1"
    );
    let out = e.finish(
        "f",
        &[
            ("%arg0".to_string(), MlirTy::Ranked(vec![Some(3)])),
            ("%arg1".to_string(), MlirTy::Ranked(vec![Some(3)])),
        ],
        &[&result],
    );
    assert!(
        out.contains("stablehlo.concatenate"),
        "missing concatenate, in:\n{out}"
    );
    assert!(
        out.contains("-> tensor<2x3xf32>"),
        "result must be a rank-2 tensor<2x3xf32>, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// A `vector(...)` whose elements are rank-1 tensors of DIFFERENT lengths —
/// a RAGGED vector-of-vectors — has no rectangular tensor form (spec §03:
/// arrays are fixed-size/rectangular). Must refuse precisely, not mis-lower
/// via `Emitter::reshape`'s lack of shape validation.
#[test]
fn lower_vector_refuses_ragged_elements() {
    let mut m = Module::new();
    let t1 = local_ref(&mut m, "t1");
    let t2 = local_ref(&mut m, "t2");
    let node = call(&mut m, "vector", &[t1, t2]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        t1,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(2)]),
            elem: ElemKind::Real,
        },
    );
    e.bind(
        t2,
        Value {
            ssa: "%arg1".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]),
            elem: ElemKind::Real,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("ragged vector-of-vectors has no tensor form"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `in(v, interval(lo, hi))` where `lo` is itself a ranked (non-scalar) value
/// of a DIFFERENT shape than `v` — `broadcast_to` only knows how to broadcast
/// a `Scalar` up to a bigger shape, so a ranked/ranked mismatch must refuse
/// rather than emit an ill-shaped op. Distinct from
/// `lower_in_interval_reduces_to_one_compare` (matching scalar shapes, no
/// broadcast needed at all).
#[test]
fn lower_in_refuses_shape_mismatched_bound() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let e0 = real(&mut m, 0.0);
    let e1 = real(&mut m, 1.0);
    let lo = call(&mut m, "vector", &[e0, e1]); // Ranked([Some(2)])
    let hi = real(&mut m, 5.0); // Scalar
    let interval = call(&mut m, "interval", &[lo, hi]);
    let node = call(&mut m, "in", &[v, interval]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(3)]), // different length than lo's
            elem: ElemKind::Real,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("shape mismatch"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `get(v, 0)` — 1-based `get` with a selector of `0` computes a negative
/// 0-based index (`0 - 1 = -1`) BEFORE the container is ever lowered or its
/// length checked — a distinct guard from
/// `lower_get0_refuses_out_of_range_index` (which trips the separate
/// known-length check on an already-lowered container), even though both
/// report the same "index out of range" text.
#[test]
fn lower_get_refuses_selector_below_one_based_floor() {
    let mut m = Module::new();
    let v = local_ref(&mut m, "v");
    let idx = int(&mut m, 0);
    let node = call(&mut m, "get", &[v, idx]);

    let mut e = Emitter::new(&m, Dtype::F32);
    e.bind(
        v,
        Value {
            ssa: "%arg0".to_string(),
            ty: MlirTy::Ranked(vec![Some(5)]),
            elem: ElemKind::Real,
        },
    );
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("out of range"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `builtin_logdensityof` with the wrong number of arguments must refuse
/// (naming the exact expected/actual count), not panic on the
/// `<[NodeId; 3]>::try_from` — mirrors `builtin_sample_refuses_wrong_arity`.
#[test]
fn builtin_logdensityof_refuses_wrong_arity() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Normal");
    let kernel_input = call(&mut m, "record", &[]);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("builtin_logdensityof: expected 3 arguments, got 2"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

// ---- Task 11: univariate discrete `@logdensity` batch ----------------------
//
// Bernoulli/Poisson/Binomial/Geometric/NegativeBinomial/NegativeBinomial2/
// Categorical/Categorical0 (§08), registered alongside the rest of §08 in
// `registry.rs`'s `REGISTRY` with `sample: None` (no discrete `@sample`
// builder is planned). Same anchor-fixture shape as the Task 8/9/10
// batches — free (`elementof`-declared) parameters, scored at a pinned
// LITERAL-INTEGER observation (unlike the continuous batches' `record(a =
// 0.5)`: a discrete density's formula is a function of the observed COUNT,
// and Categorical/Categorical0 additionally need that literal to drive a
// static `get`/`get0` slice — see `registry::categorical_logpdf`'s doc
// comment). Categorical/Categorical0's `p` is a literal array (`[0.2, 0.3,
// 0.5]`), not an `elementof`-declared free parameter, same reasoning as
// `UNIFORM_DENSITY_SRC`'s literal `support`: `elementof(stdsimplex(N))`
// currently leaves `stdsimplex` typed `%deferred` with a `Severity::Note`
// diagnostic (`crates/infer`'s `stdsimplex` type rule is not yet
// implemented — see `crates/infer/tests/spec_coverage_shape_gaps.rs`'s
// `stdsimplex_size_from_fixed_ref` for the same gap acknowledged there),
// which trips this file's `determinize_src` helper's strict `diags.is_empty()`
// assert; a literal `p` sidesteps the gap entirely and keeps `a` a zero-arg
// `func.func @logdensity()`, mirroring `UNIFORM_DENSITY_SRC`.

const BERNOULLI_DENSITY_SRC: &str = "\
p = elementof(unitinterval)
a = draw(Bernoulli(p = p))
lp = logdensityof(lawof(record(a = a)), record(a = 1))
";

const POISSON_DENSITY_SRC: &str = "\
rate = elementof(nonnegreals)
a = draw(Poisson(rate = rate))
lp = logdensityof(lawof(record(a = a)), record(a = 3))
";

const BINOMIAL_DENSITY_SRC: &str = "\
n = elementof(posintegers)
p = elementof(unitinterval)
a = draw(Binomial(n = n, p = p))
lp = logdensityof(lawof(record(a = a)), record(a = 2))
";

const GEOMETRIC_DENSITY_SRC: &str = "\
p = elementof(unitinterval)
a = draw(Geometric(p = p))
lp = logdensityof(lawof(record(a = a)), record(a = 4))
";

const NEGATIVE_BINOMIAL_DENSITY_SRC: &str = "\
alpha = elementof(posreals)
beta = elementof(posreals)
a = draw(NegativeBinomial(alpha = alpha, beta = beta))
lp = logdensityof(lawof(record(a = a)), record(a = 2))
";

const NEGATIVE_BINOMIAL2_DENSITY_SRC: &str = "\
mu = elementof(posreals)
psi = elementof(posreals)
a = draw(NegativeBinomial2(mu = mu, psi = psi))
lp = logdensityof(lawof(record(a = a)), record(a = 2))
";

const CATEGORICAL_DENSITY_SRC: &str = "\
a = draw(Categorical(p = [0.2, 0.3, 0.5]))
lp = logdensityof(lawof(record(a = a)), record(a = 2))
";

const CATEGORICAL0_DENSITY_SRC: &str = "\
a = draw(Categorical0(p = [0.2, 0.3, 0.5]))
lp = logdensityof(lawof(record(a = a)), record(a = 1))
";

/// §06 `Dirac(value = value)` (the measure monad's unit, not a §08 catalog
/// distribution): the point mass at a free `value`, scored at the literal
/// atom `a = 3`. Mirrors the discrete-batch fixtures above (a free
/// `elementof`-declared parameter, scored at a pinned literal observation) —
/// the zero-inflated-binomial idiom's own `Dirac(0)` pins `value` to a
/// literal instead, but the registry builder (`compare`/`select`) is
/// identical either way; a free `value` here exercises it as an ordinary
/// `func.func` argument, same discipline as `BERNOULLI_DENSITY_SRC`'s free
/// `p`.
const DIRAC_DENSITY_SRC: &str = "\
value = elementof(integers)
a = draw(Dirac(value = value))
lp = logdensityof(lawof(record(a = a)), record(a = 3))
";

/// §08 Bernoulli, verbatim: `k * log(p) + (1 - k) * log(1 - p)`. Op counts:
/// two `log`s (`p`, `1-p`), two `multiply`s, two `subtract`s (`1-k`, `1-p`),
/// one `add`. No `chlo.*`.
#[test]
fn emit_logdensity_bernoulli_has_expected_structure() {
    let d = determinize_src(BERNOULLI_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "p must become a func arg, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 2);
    assert_eq!(out.matches("stablehlo.multiply").count(), 2);
    assert_eq!(out.matches("stablehlo.subtract").count(), 2);
    assert_eq!(out.matches("stablehlo.add").count(), 1);
    assert!(
        !out.contains("chlo."),
        "Bernoulli needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_bernoulli_matches_frozen_golden() {
    let d = determinize_src(BERNOULLI_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/bernoulli_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/bernoulli_logdensity.mlir)"
    );
}

/// §08 Poisson, verbatim: `k * log(rate) - rate - lgamma(k + 1)`. Op counts:
/// one `log`, one `multiply`, two `negate`s (`-rate`'s own, `-lgamma(k+1)`),
/// three `add`s, one `chlo.lgamma`.
#[test]
fn emit_logdensity_poisson_has_expected_structure() {
    let d = determinize_src(POISSON_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "rate must become a func arg, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 1);
    assert_eq!(out.matches("stablehlo.multiply").count(), 1);
    assert_eq!(out.matches("stablehlo.negate").count(), 2);
    assert_eq!(out.matches("stablehlo.add").count(), 3);
    assert_eq!(out.matches("chlo.lgamma").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_poisson_matches_frozen_golden() {
    let d = determinize_src(POISSON_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/poisson_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/poisson_logdensity.mlir)"
    );
}

/// §08 Binomial, verbatim: `logC(n, k) + k * log(p) + (n - k) * log(1 - p)`,
/// `logC(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)`. Op counts: two
/// `log`s, two `multiply`s, two `subtract`s (`n-k`, `1-p`), seven `add`s, two
/// `negate`s, three `chlo.lgamma`s.
#[test]
fn emit_logdensity_binomial_has_expected_structure() {
    let d = determinize_src(BINOMIAL_DENSITY_SRC);
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
        "n/p must become func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 2);
    assert_eq!(out.matches("stablehlo.multiply").count(), 2);
    assert_eq!(out.matches("stablehlo.subtract").count(), 2);
    assert_eq!(out.matches("stablehlo.add").count(), 7);
    assert_eq!(out.matches("stablehlo.negate").count(), 2);
    assert_eq!(out.matches("chlo.lgamma").count(), 3);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_binomial_matches_frozen_golden() {
    let d = determinize_src(BINOMIAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/binomial_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/binomial_logdensity.mlir)"
    );
}

/// §08 Geometric, verbatim: `log(p) + k * log(1 - p)`. Op counts: two `log`s,
/// one `multiply`, one `subtract` (`1-p`), one `add`. No `chlo.*`.
#[test]
fn emit_logdensity_geometric_has_expected_structure() {
    let d = determinize_src(GEOMETRIC_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "p must become a func arg, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 2);
    assert_eq!(out.matches("stablehlo.multiply").count(), 1);
    assert_eq!(out.matches("stablehlo.subtract").count(), 1);
    assert_eq!(out.matches("stablehlo.add").count(), 1);
    assert!(
        !out.contains("chlo."),
        "Geometric needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_geometric_matches_frozen_golden() {
    let d = determinize_src(GEOMETRIC_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/geometric_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/geometric_logdensity.mlir)"
    );
}

/// §08 NegativeBinomial, verbatim: `[lgamma(k+alpha) - lgamma(alpha) -
/// lgamma(k+1)] + alpha * (log(beta) - log(beta+1)) - k * log(beta+1)`. Op
/// counts: two `log`s, two `multiply`s, four `negate`s, eight `add`s, three
/// `chlo.lgamma`s, no `subtract` (every term is additive/negated, not
/// subtracted).
#[test]
fn emit_logdensity_negative_binomial_has_expected_structure() {
    let d = determinize_src(NEGATIVE_BINOMIAL_DENSITY_SRC);
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
        "alpha/beta must become func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 2);
    assert_eq!(out.matches("stablehlo.multiply").count(), 2);
    assert_eq!(out.matches("stablehlo.negate").count(), 4);
    assert_eq!(out.matches("stablehlo.add").count(), 8);
    assert_eq!(out.matches("chlo.lgamma").count(), 3);
    assert!(
        !out.contains("stablehlo.subtract"),
        "NegativeBinomial's log-form has no subtraction, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_negative_binomial_matches_frozen_golden() {
    let d = determinize_src(NEGATIVE_BINOMIAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/negative_binomial_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/negative_binomial_logdensity.mlir)"
    );
}

/// §08 NegativeBinomial2, verbatim: `[lgamma(k+psi) - lgamma(psi) -
/// lgamma(k+1)] + k * (log(mu) - log(mu+psi)) + psi * (log(psi) -
/// log(mu+psi))`. Op counts: three `log`s, two `multiply`s, three `negate`s,
/// nine `add`s, three `chlo.lgamma`s, no `subtract`.
#[test]
fn emit_logdensity_negative_binomial2_has_expected_structure() {
    let d = determinize_src(NEGATIVE_BINOMIAL2_DENSITY_SRC);
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
        "mu/psi must become func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 3);
    assert_eq!(out.matches("stablehlo.multiply").count(), 2);
    assert_eq!(out.matches("stablehlo.negate").count(), 3);
    assert_eq!(out.matches("stablehlo.add").count(), 9);
    assert_eq!(out.matches("chlo.lgamma").count(), 3);
    assert!(
        !out.contains("stablehlo.subtract"),
        "NegativeBinomial2's log-form has no subtraction, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_negative_binomial2_matches_frozen_golden() {
    let d = determinize_src(NEGATIVE_BINOMIAL2_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/negative_binomial2_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/negative_binomial2_logdensity.mlir)"
    );
}

/// §08 Categorical, verbatim: `log(p_k)`, `k` 1-based. `p`'s literal array
/// lowers via `vector(...)` (one `concatenate` of three reshaped scalars,
/// spec §07), then the 1-based selector `k=2` slices 0-based array position
/// `k-1=1` (the `[1:2]` slice bound) — a zero-arg `func.func @logdensity()`
/// (no free parameters: `p` is a literal, and the observed `k` is consumed
/// structurally, never lowered as an arithmetic operand). Exactly one
/// `slice`, one final `log`, and four `reshape`s (three packing `vector`'s
/// elements + one unpacking the sliced length-1 result to a `Scalar`) — no
/// `chlo.*`, `negate`, `subtract`, `multiply`, or `add`: the density is a
/// pure lookup.
#[test]
fn emit_logdensity_categorical_has_expected_structure() {
    let d = determinize_src(CATEGORICAL_DENSITY_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_logdensity(&d);

    assert!(
        out.contains("func.func @logdensity()"),
        "missing func.func @logdensity() (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> tensor<f32>"),
        "must return tensor<f32> in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.concatenate"),
        "missing concatenate (p's vector literal), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.slice") && out.contains("[1:2]"),
        "expected 1-based k=2 to slice 0-based index 1, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.slice").count(), 1);
    assert_eq!(out.matches("stablehlo.reshape").count(), 4);
    assert_eq!(out.matches("stablehlo.log").count(), 1);
    assert!(
        !out.contains("chlo.")
            && !out.contains("stablehlo.negate")
            && !out.contains("stablehlo.subtract")
            && !out.contains("stablehlo.multiply")
            && !out.contains("stablehlo.add"),
        "Categorical's density is a pure lookup, no arithmetic, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_categorical_matches_frozen_golden() {
    let d = determinize_src(CATEGORICAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/categorical_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/categorical_logdensity.mlir)"
    );
}

/// §08 Categorical0, verbatim: `log(p_{k+1})`, `k` 0-based — the 0-based
/// selector `k=1` slices the SAME 0-based array position (`[1:2]`) as
/// `CATEGORICAL_DENSITY_SRC`'s 1-based `k=2` (see
/// `registry::categorical0_logpdf`'s doc comment for why `p_{k+1}` and
/// `get0(p, k)` coincide), which is why both fixtures are pinned to
/// numerically identical `log(0.3)` results — cross-checked against SciPy in
/// the Task 11 report.
#[test]
fn emit_logdensity_categorical0_has_expected_structure() {
    let d = determinize_src(CATEGORICAL0_DENSITY_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_logdensity(&d);

    assert!(
        out.contains("func.func @logdensity()"),
        "missing func.func @logdensity() (no free params) in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.slice") && out.contains("[1:2]"),
        "expected 0-based k=1 to slice 0-based index 1, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.slice").count(), 1);
    assert_eq!(out.matches("stablehlo.reshape").count(), 4);
    assert_eq!(out.matches("stablehlo.log").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_categorical0_matches_frozen_golden() {
    let d = determinize_src(CATEGORICAL0_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/categorical0_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/categorical0_logdensity.mlir)"
    );
}

/// §06 Dirac, verbatim: `select(v == value, 0.0, -inf)`. Op counts: one
/// `stablehlo.compare` (`"EQ"`), one `stablehlo.negate` (`-inf`), one
/// `stablehlo.select`. No `log`/`multiply`/`add`/`chlo.*` — Dirac's density
/// is a pure indicator, not an arithmetic formula. `value` becomes an f32
/// (not i32) func arg — every free scalar parameter emits at the target
/// `Dtype` regardless of its declared FlatPPL domain (same as every other
/// discrete distribution's real-valued PARAMETERS here, e.g. Bernoulli's
/// `p`); the pinned literal `v = 3` is `stablehlo.convert`ed up to match at
/// the `compare`, via `Emitter::compare`'s elem-kind widening.
#[test]
fn emit_logdensity_dirac_has_expected_structure() {
    let d = determinize_src(DIRAC_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>"),
        "value must become a func arg, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert!(out.contains("EQ"), "must compare EQ, in:\n{out}");
    assert_eq!(out.matches("stablehlo.negate").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert_eq!(out.matches("stablehlo.log").count(), 0);
    assert_eq!(out.matches("stablehlo.multiply").count(), 0);
    assert!(
        !out.contains("chlo."),
        "Dirac needs no CHLO ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_dirac_matches_frozen_golden() {
    let d = determinize_src(DIRAC_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/dirac_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/dirac_logdensity.mlir)"
    );
}

/// §06 Dirac `@sample`: `rand(rng, Dirac(value = v))` returns the atom `v`
/// deterministically, consuming NO randomness. Fixed-hyperparameter forward
/// model (`value = 3.0`), so — like `NORMAL_SAMPLE_SRC` — `emit_sample`
/// produces a `func.func @sample(%key)` with no free params. The single draw
/// is `dirac_sample`'s identity `value`.
const DIRAC_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Dirac(value = 3.0))
draws = rand(s, lawof(x))
";

/// The deterministic-draw structural test: exactly ZERO `rng_bit_generator`
/// draws (Dirac consumes no randomness), the drawn value is the `value` atom,
/// and the `(value, advanced-key)` pair returns the seeded `%key` UNTOUCHED —
/// the RNG state threads through unchanged (spec §07 rng ABI).
#[test]
fn emit_sample_dirac_has_expected_structure() {
    let d = determinize_src(DIRAC_SAMPLE_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing func.func @sample(%key: tensor<2xui64>) (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<f32>, tensor<2xui64>)"),
        "must return the (value, advanced-key) pair in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        0,
        "Dirac consumes no randomness — expected zero rng_bit_generator draws, in:\n{out}"
    );
    assert!(
        out.contains("return %0, %key :"),
        "the seeded %key must thread through untouched as the advanced rng state, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, key-threading)
/// must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_sample_dirac_matches_frozen_golden() {
    let d = determinize_src(DIRAC_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/dirac_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/dirac_sample.mlir)"
    );
}

/// `Categorical(p)` scored at a NON-literal `k` — a `Ref` to a top-level
/// binding, not an integer literal `Node` itself — must refuse precisely
/// (refuse-don't-mislower) rather than attempt a `stablehlo.gather`-shaped
/// dynamic selector this emitter has no helper for: the task brief's
/// explicit "dynamic gather is not supported" case. `v` here still lowers
/// fine as an ordinary scalar (`lower_logdensityof` eagerly lowers `v` for
/// every registry entry, before ever reaching `categorical_logpdf` — see
/// `Params::variate`'s doc comment); what makes it "non-literal" is
/// structural, not numeric: [`literal_variate_index`]'s check (mirroring
/// `ops::literal_index`'s identical no-ref-chasing discipline for an
/// ordinary `get`/`get0` selector) only accepts a bare `Node::Lit(Scalar::
/// Int(_))`, not a `Ref` that happens to resolve to one. Hand-built (not
/// `determinize_src`): the determiniser's own discrete-marginal expansion
/// never produces this shape (every real `builtin_logdensityof(Categorical,
/// ...)` it emits scores a literal atom directly — see
/// `crates/determinizer/tests/density_golden.rs`'s
/// `kchain_discrete_categorical_latent_lowers_to_mass_weighted_logsumexp`),
/// so this exercises the registry's defensive check directly.
#[test]
fn categorical_logpdf_refuses_non_literal_selector() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Categorical");
    let e0 = real(&mut m, 0.2);
    let e1 = real(&mut m, 0.3);
    let e2 = real(&mut m, 0.5);
    let probs = call(&mut m, "vector", &[e0, e1, e2]);
    let kernel_input = record_node(&mut m, &[("p", probs)]);
    let k_val = int(&mut m, 2);
    top_level(&mut m, "k", k_val);
    let v = self_ref(&mut m, "k");
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("dynamic gather is not supported"),
        "unexpected message: {}",
        err.msg
    );
}

/// `Categorical(p)` scored at an in-range-looking but too-large literal `k`
/// (`4`, `p` only length 3) must refuse with an "out of range" message
/// naming the mismatch, not slice past `p`'s statically-known length.
#[test]
fn categorical_logpdf_refuses_out_of_range_category() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Categorical");
    let e0 = real(&mut m, 0.2);
    let e1 = real(&mut m, 0.3);
    let e2 = real(&mut m, 0.5);
    let probs = call(&mut m, "vector", &[e0, e1, e2]);
    let kernel_input = record_node(&mut m, &[("p", probs)]);
    let v = int(&mut m, 4);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("out of range"),
        "unexpected message: {}",
        err.msg
    );
}

/// `Categorical0(p)` scored at literal `k = 0` (the FLOOR of its 0-based
/// support) must slice 0-based array position 0 — the boundary opposite
/// `CATEGORICAL0_DENSITY_SRC`'s interior `k = 1` case, and distinct from
/// `Categorical`'s own `k = 1` floor (one-based, `get`'s convention already
/// covered by `lower_get_refuses_selector_below_one_based_floor`).
#[test]
fn categorical0_logpdf_at_floor_slices_first_element() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Categorical0");
    let e0 = real(&mut m, 0.2);
    let e1 = real(&mut m, 0.3);
    let e2 = real(&mut m, 0.5);
    let probs = call(&mut m, "vector", &[e0, e1, e2]);
    let kernel_input = record_node(&mut m, &[("p", probs)]);
    let v = int(&mut m, 0);
    let node = call(&mut m, "builtin_logdensityof", &[ctor, kernel_input, v]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let result = e.lower_node(node).unwrap();
    let out = e.finish("f", &[], &[&result]);
    assert!(
        out.contains("stablehlo.slice") && out.contains("[0:1]"),
        "expected k=0 to slice 0-based index 0, in:\n{out}"
    );
}

// ---- Task 12: multivariate vector `@logdensity` batch -----------------------
//
// MvNormal/Dirichlet/Multinomial, registered alongside the rest of §08 in
// `REGISTRY` with `sample: None` (samplers land in Tasks 14/15/16 — see
// `registry.rs`'s batch doc comment). Unlike every Task 8/9/10/11 fixture,
// `mu`/`cov`/`alpha`/`p` here are vector/matrix-typed free parameters:
// `elementof(cartpow(reals, n))` for a length-`n` vector, `elementof(cartpow
// (reals, [n, n]))` for an `n`x`n` matrix (both real, tested syntax — spec
// `flatppl-design/docs/10-examples.md`'s own worked MvNormal example uses
// exactly this shape: `some_mean = elementof(cartpow(reals, 3))`, `some_cov =
// elementof(cartpow(reals, [3, 3]))`). `crates/infer`'s `cartpow` type rule
// (unlike `stdsimplex`'s — see Task 11's report on why `Categorical`/
// `Categorical0` fall back to a literal `p`) is fully implemented, so these
// fixtures determinize with zero diagnostics and reach the registry as
// ordinary free-parameter func args, exactly like every scalar fixture above.

/// The Task-12 MvNormal anchor fixture: a length-2 free `mu`/`cov`, scored at
/// a pinned length-2 observation.
const MVNORMAL_DENSITY_SRC: &str = "\
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 2]))
a = draw(MvNormal(mu = mu, cov = cov))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1]))
";

/// §08 MvNormal, verbatim: `-(n/2)*log(2*pi) - 1/2*log|Sigma| -
/// 1/2*(x-mu)^T Sigma^-1 (x-mu)`, via `L = cholesky(Sigma)`, `log|Sigma| = 2 *
/// sum(log(diag(L)))`, and the quadratic form via `tri_solve` + `reduce_sum`
/// (never a full matrix inverse). Structural check: exactly one
/// `stablehlo.cholesky` and one (generic-form) `triangular_solve`, the
/// `iota`/`compare`/`select`/`reduce` idiom `Emitter::diag` lowers to, `mu`/
/// `cov` become `tensor<2xf32>`/`tensor<2x2xf32>` func args (not
/// `tensor<f32>` — the free-parameter binding loop in `modes.rs` reads each
/// binding's real inferred shape via `mlir_type_of`, not a scalar default).
#[test]
fn emit_logdensity_mvnormal_has_expected_structure() {
    let d = determinize_src(MVNORMAL_DENSITY_SRC);
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
        out.contains("%arg0: tensor<2xf32>") && out.contains("%arg1: tensor<2x2xf32>"),
        "mu/cov must become vector/matrix func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.cholesky").count(), 1);
    assert_eq!(out.matches("\"stablehlo.triangular_solve\"").count(), 1);
    assert_eq!(
        out.matches("stablehlo.iota").count(),
        2,
        "diag's row/col index tensors, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 3);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_mvnormal_matches_frozen_golden() {
    let d = determinize_src(MVNORMAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/mvnormal_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/mvnormal_logdensity.mlir)"
    );
}

/// A `mu`/`cov` whose vector length is NOT statically known (`cartpow(reals,
/// m)` for a free `m`, not a literal size) must refuse precisely — `n` is
/// baked into a scalar literal constant at emit time, so a dynamic length has
/// no lowering, not merely an inconvenient one.
#[test]
fn mvnormal_logpdf_refuses_dynamic_vector_length() {
    let src = "\
m = elementof(posintegers)
mu = elementof(cartpow(reals, m))
cov = elementof(cartpow(reals, [m, m]))
a = draw(MvNormal(mu = mu, cov = cov))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1]))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("statically-known vector length"),
        "unexpected message: {}",
        err.msg
    );
}

/// A `cov` that is square but the WRONG size for `mu`'s length (`mu`: length
/// 2, `cov`: `[3, 3]`) must refuse precisely, not emit
/// operand-shape-incompatible `stablehlo.triangular_solve` (the previous
/// behavior: `cholesky`/`diag` both accept a `[3, 3]` operand — `cholesky`
/// validates nothing, `diag` only checks rank 2 — so the mismatch only
/// surfaced at the final `tri_solve(L, x-mu)` against a length-2 `x-mu`).
#[test]
fn mvnormal_logpdf_refuses_wrong_size_square_cov() {
    let src = "\
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [3, 3]))
a = draw(MvNormal(mu = mu, cov = cov))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1]))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("MvNormal cov must be an"),
        "unexpected message: {}",
        err.msg
    );
}

/// A non-square `cov` (`mu`: length 2, `cov`: `[2, 3]`) must refuse
/// precisely, not reach `stablehlo.cholesky` on a non-square operand (no
/// real StableHLO consumer accepts that).
#[test]
fn mvnormal_logpdf_refuses_nonsquare_cov() {
    let src = "\
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 3]))
a = draw(MvNormal(mu = mu, cov = cov))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1]))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("MvNormal cov must be an"),
        "unexpected message: {}",
        err.msg
    );
}

/// The Task-12 Dirichlet anchor fixture: a length-3 free `alpha`, scored at a
/// pinned length-3 observation on the simplex.
const DIRICHLET_DENSITY_SRC: &str = "\
alpha = elementof(cartpow(posreals, 3))
a = draw(Dirichlet(alpha = alpha))
lp = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.3, 0.5]))
";

/// §08 Dirichlet, verbatim: `lgamma(sum(alpha)) - sum(lgamma(alpha)) +
/// sum((alpha - 1) * log(x))`. Op counts: two `chlo.lgamma`s (one on the
/// reduced sum, one elementwise on the length-3 `alpha` vector itself), three
/// `stablehlo.reduce`s (`sum(alpha)`, `sum(lgamma(alpha))`, the final
/// `sum((alpha-1)*log(x))`), one `subtract` (`alpha - 1`, a same-shape
/// splat constant per the batch's doc comment in `registry.rs`), one
/// `multiply`. No `cholesky`/`triangular_solve`/`iota` — Dirichlet needs no
/// matrix ops, only reductions.
#[test]
fn emit_logdensity_dirichlet_has_expected_structure() {
    let d = determinize_src(DIRICHLET_DENSITY_SRC);
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
        out.contains("%arg0: tensor<3xf32>"),
        "alpha must become a length-3 vector func arg, in:\n{out}"
    );
    assert_eq!(out.matches("chlo.lgamma").count(), 2);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 3);
    assert_eq!(out.matches("stablehlo.subtract").count(), 1);
    assert_eq!(out.matches("stablehlo.multiply").count(), 1);
    assert!(
        !out.contains("stablehlo.cholesky")
            && !out.contains("triangular_solve")
            && !out.contains("stablehlo.iota"),
        "Dirichlet needs no matrix ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_dirichlet_matches_frozen_golden() {
    let d = determinize_src(DIRICHLET_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/dirichlet_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/dirichlet_logdensity.mlir)"
    );
}

/// The Task-12 Multinomial anchor fixture: a free scalar trial count `n` and
/// a free length-3 `p`, scored at a pinned length-3 count observation.
const MULTINOMIAL_DENSITY_SRC: &str = "\
n = elementof(posintegers)
p = elementof(cartpow(unitinterval, 3))
a = draw(Multinomial(n = n, p = p))
lp = logdensityof(lawof(record(a = a)), record(a = [2, 3, 5]))
";

/// §08 Multinomial, verbatim: `lgamma(n+1) - sum(lgamma(x+1)) + sum(x *
/// log(p))`. Op counts: two `chlo.lgamma`s (`lgamma(n+1)` scalar, `lgamma(x+1)`
/// elementwise on the length-3 `x` vector), two `stablehlo.reduce`s
/// (`sum(lgamma(x+1))`, `sum(x*log(p))`), four `add`s (`n+1`, `x+1` — the
/// latter a same-shape splat constant per the batch's doc comment in
/// `registry.rs` — plus the two final-combination adds), one `multiply` (`x *
/// log(p)`), one `negate`. The `add` count is asserted via the tighter `"=
/// stablehlo.add "` pattern, not a bare `"stablehlo.add"` substring: each
/// `stablehlo.reduce(...)` line's own pretty form embeds the literal text
/// `"applies stablehlo.add across dimensions"` (the combine-op name), which
/// would otherwise double-count as a spurious `add` for every `reduce_sum`
/// call in this formula.
#[test]
fn emit_logdensity_multinomial_has_expected_structure() {
    let d = determinize_src(MULTINOMIAL_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>") && out.contains("%arg1: tensor<3xf32>"),
        "n/p must become scalar/vector func args, in:\n{out}"
    );
    assert_eq!(out.matches("chlo.lgamma").count(), 2);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 2);
    assert_eq!(out.matches("= stablehlo.add ").count(), 4);
    assert_eq!(out.matches("stablehlo.multiply").count(), 1);
    assert_eq!(out.matches("stablehlo.negate").count(), 1);
    assert!(
        !out.contains("stablehlo.cholesky")
            && !out.contains("triangular_solve")
            && !out.contains("stablehlo.iota"),
        "Multinomial needs no matrix ops, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_multinomial_matches_frozen_golden() {
    let d = determinize_src(MULTINOMIAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/multinomial_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/multinomial_logdensity.mlir)"
    );
}

// ---- Task 13: matrix distribution `@logdensity` batch -----------------------
//
// Wishart/InverseWishart/LKJ/LKJCholesky. Every matrix-shaped kwarg (`scale`,
// and the scored variate itself) is declared `elementof(cartpow(reals, [n,
// n]))` — a free parameter, exactly like `MvNormal`'s own `mu`/`cov` (Task
// 12) — rather than a literal nested-array constant (`[[..], [..]]`):
// `Emitter::vector`'s pretty-printed concatenate lowering (Task 3/4) assumes
// every element it is handed is itself a `Scalar` (it reshapes each element
// to `tensor<1x...>` before concatenating), so a NESTED vector literal (a
// vector of vectors, i.e. a literal matrix) silently lowers to the wrong
// rank/shape instead of refusing — verified directly: scoring a Wishart
// fixture at a literal `record(x = [[2.0, 0.3], [0.3, 1.5]])` reaches
// `wishart_logpdf` with a `v` whose `MlirTy` is `Ranked([Some(2)])`, not
// `Ranked([Some(2), Some(2)])`, which this batch's own `require_matrix_dim`
// guard then (correctly) refuses. That gap is in Task 3/4's `Emitter::
// vector`/`ops::lower_vector`, not this batch's own composition — flagged in
// the Task 13 report as a follow-up (a matrix-shaped LITERAL variate should
// either lower correctly or refuse, never silently mislower), worked around
// here by using a free-parameter variate instead, which does not go through
// `Emitter::vector` at all (it becomes an ordinary `%argN` via the same
// free-parameter binding path `scale`/`cov`/`mu` already use).
//
// LKJ/LKJCholesky's `n` is spec's own explicit dimension kwarg and must be
// FIXED phase (a plain literal binding, no `elementof`/`draw` ancestor —
// spec §04) for `literal_fixed_positive_int` to read it as a Rust `u64` at
// emit time; every fixture below binds `n` as a bare top-level literal
// (`n = 3`), never `elementof(posintegers)` (which would make it
// `%parameterized`, i.e. a runtime-only `%argN` with no compile-time value
// to unroll `log_cn_lkj`'s `k` sum against — see
// `lkj_logpdf_refuses_parameterized_n` below).

/// The Task-13 Wishart anchor fixture: a free `2x2` `scale`, free `nu`, and a
/// free `2x2` `x_obs` scored variate (see the batch doc comment for why the
/// variate is a free parameter, not a literal matrix).
const WISHART_DENSITY_SRC: &str = "\
scale = elementof(cartpow(reals, [2, 2]))
nu = elementof(posreals)
x_obs = elementof(cartpow(reals, [2, 2]))
x = draw(Wishart(nu = nu, scale = scale))
lp = logdensityof(lawof(record(x = x)), record(x = x_obs))
";

/// §08 Wishart, verbatim: `((nu-n-1)/2) log|X| - (1/2) tr(V^-1 X) -
/// (nu*n/2) log2 - (nu/2) log|V| - logGamma_n(nu/2)`, `n = 2` (read off
/// `scale`'s own shape). Op counts: two `stablehlo.cholesky` (`L_V`, `L_X`),
/// one (generic-form) `triangular_solve` (the Frobenius trace), four
/// `stablehlo.iota`/two `compare`/two `select` (two `diag` calls, one per
/// `log|.|` term — each `diag` lowers to its own iota/compare/select/reduce
/// idiom, see `Emitter::diag`'s doc comment), six `stablehlo.reduce(` (two
/// per `log|.|` term's `diag`-row-sum + final `reduce_sum`, plus two for the
/// trace's full matrix `reduce_sum`), two `chlo.lgamma` (`log_mv_gamma`'s `n
/// = 2` unrolled loop).
#[test]
fn emit_logdensity_wishart_has_expected_structure() {
    let d = determinize_src(WISHART_DENSITY_SRC);
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
        out.contains("%arg0: tensor<2x2xf32>")
            && out.contains("%arg1: tensor<f32>")
            && out.contains("%arg2: tensor<2x2xf32>"),
        "scale/nu/x_obs must become matrix/scalar/matrix func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.cholesky").count(), 2);
    assert_eq!(out.matches("\"stablehlo.triangular_solve\"").count(), 1);
    assert_eq!(out.matches("stablehlo.iota").count(), 4);
    assert_eq!(out.matches("stablehlo.compare").count(), 2);
    assert_eq!(out.matches("stablehlo.select").count(), 2);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 6);
    assert_eq!(out.matches("chlo.lgamma").count(), 2);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_wishart_matches_frozen_golden() {
    let d = determinize_src(WISHART_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/wishart_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/wishart_logdensity.mlir)"
    );
}

/// A non-square `scale` (`[2, 3]`) must refuse precisely — `n` (the row/
/// column count `static_square_matrix_dim` reads off `scale`) has no value
/// to read at all, so this must never reach `stablehlo.cholesky` on a
/// non-square operand.
#[test]
fn wishart_logpdf_refuses_nonsquare_scale() {
    let src = "\
scale = elementof(cartpow(reals, [2, 3]))
nu = elementof(posreals)
x_obs = elementof(cartpow(reals, [2, 2]))
x = draw(Wishart(nu = nu, scale = scale))
lp = logdensityof(lawof(record(x = x)), record(x = x_obs))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("square matrix"),
        "unexpected message: {}",
        err.msg
    );
}

/// The Task-13 InverseWishart anchor fixture — same shape as
/// [`WISHART_DENSITY_SRC`], only the ctor differs.
const INVERSE_WISHART_DENSITY_SRC: &str = "\
scale = elementof(cartpow(reals, [2, 2]))
nu = elementof(posreals)
x_obs = elementof(cartpow(reals, [2, 2]))
x = draw(InverseWishart(nu = nu, scale = scale))
lp = logdensityof(lawof(record(x = x)), record(x = x_obs))
";

/// §08 InverseWishart, verbatim: `(nu/2) log|Psi| - ((nu+n+1)/2) log|X| -
/// (1/2) tr(Psi X^-1) - (nu*n/2) log2 - logGamma_n(nu/2)`. Same op-count
/// shape as [`emit_logdensity_wishart_has_expected_structure`] (the formula
/// rearranges the same five terms; `tr(Psi X^-1)` is computed as `tr(X^-1
/// Psi)` instead — see [`inverse_wishart_logpdf`]'s doc comment — but that
/// is still exactly one `tri_solve`).
#[test]
fn emit_logdensity_inverse_wishart_has_expected_structure() {
    let d = determinize_src(INVERSE_WISHART_DENSITY_SRC);
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
        out.contains("%arg0: tensor<2x2xf32>")
            && out.contains("%arg1: tensor<f32>")
            && out.contains("%arg2: tensor<2x2xf32>"),
        "scale/nu/x_obs must become matrix/scalar/matrix func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.cholesky").count(), 2);
    assert_eq!(out.matches("\"stablehlo.triangular_solve\"").count(), 1);
    assert_eq!(out.matches("stablehlo.iota").count(), 4);
    assert_eq!(out.matches("stablehlo.compare").count(), 2);
    assert_eq!(out.matches("stablehlo.select").count(), 2);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 6);
    assert_eq!(out.matches("chlo.lgamma").count(), 2);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_inverse_wishart_matches_frozen_golden() {
    let d = determinize_src(INVERSE_WISHART_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/inverse_wishart_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/inverse_wishart_logdensity.mlir)"
    );
}

/// A scored variate `X` (`[3, 3]`) that mismatches `scale`'s dimension (`n =
/// 2`) must refuse precisely, not reach `stablehlo.cholesky` on a variate
/// whose shape silently disagrees with `scale`'s.
#[test]
fn inverse_wishart_logpdf_refuses_mismatched_variate() {
    let src = "\
scale = elementof(cartpow(reals, [2, 2]))
nu = elementof(posreals)
x_obs = elementof(cartpow(reals, [3, 3]))
x = draw(InverseWishart(nu = nu, scale = scale))
lp = logdensityof(lawof(record(x = x)), record(x = x_obs))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("InverseWishart X must be an 2x2 matrix"),
        "unexpected message: {}",
        err.msg
    );
}

/// The Task-13 LKJ anchor fixture: a FIXED `n = 3`, free `eta`, and a free
/// `3x3` `c_obs` scored variate.
const LKJ_DENSITY_SRC: &str = "\
n = 3
eta = elementof(posreals)
c_obs = elementof(cartpow(reals, [3, 3]))
c = draw(LKJ(n = n, eta = eta))
lp = logdensityof(lawof(record(c = c)), record(c = c_obs))
";

/// §08 LKJ, verbatim: `(eta-1) log det(C) - log c_n(eta)`, `n = 3` (spec's
/// own fixed dimension kwarg). Op counts: one `stablehlo.cholesky` (`L_C`),
/// no `triangular_solve` (LKJ needs no trace), two `stablehlo.iota`/one
/// `compare`/one `select` (`diag`'s own idiom, called once), two
/// `stablehlo.reduce(` (`diag`'s row-sum + `log_det_from_chol`'s
/// `reduce_sum`), four `chlo.lgamma` (`log_cn_lkj`'s `k = 1..n-1` loop, `n =
/// 3` so 2 iterations, 2 `lgamma`s each).
#[test]
fn emit_logdensity_lkj_has_expected_structure() {
    let d = determinize_src(LKJ_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>") && out.contains("%arg1: tensor<3x3xf32>"),
        "eta/c_obs must become scalar/matrix func args, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.cholesky").count(), 1);
    assert!(
        !out.contains("triangular_solve"),
        "LKJ needs no trace/tri_solve, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.iota").count(), 2);
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 2);
    assert_eq!(out.matches("chlo.lgamma").count(), 4);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_lkj_matches_frozen_golden() {
    let d = determinize_src(LKJ_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/lkj_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/lkj_logdensity.mlir)"
    );
}

/// A scored variate `C` (`[2, 2]`) that mismatches the FIXED `n = 3` kwarg
/// must refuse precisely, not reach `stablehlo.cholesky` on a `[2, 2]`
/// operand while `log_cn_lkj`'s Rust loop unrolls for `n = 3`.
#[test]
fn lkj_logpdf_refuses_mismatched_variate() {
    let src = "\
n = 3
eta = elementof(posreals)
c_obs = elementof(cartpow(reals, [2, 2]))
c = draw(LKJ(n = n, eta = eta))
lp = logdensityof(lawof(record(c = c)), record(c = c_obs))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("LKJ C must be an 3x3 matrix"),
        "unexpected message: {}",
        err.msg
    );
}

/// A `%parameterized` (`elementof`-declared) `n` — rather than a FIXED-phase
/// literal binding — has no Rust `u64` to unroll `log_cn_lkj`'s `k` sum
/// against and must refuse precisely, not panic reaching a `for k in 1..n`
/// with no `n` at all. Exercises [`literal_fixed_positive_int`]'s refusal
/// arm directly (distinct from every other guard in this batch, which all
/// check a matrix *shape* — this one checks a scalar kwarg's *phase*).
#[test]
fn lkj_logpdf_refuses_parameterized_n() {
    let src = "\
n = elementof(posintegers)
eta = elementof(posreals)
c_obs = elementof(cartpow(reals, [3, 3]))
c = draw(LKJ(n = n, eta = eta))
lp = logdensityof(lawof(record(c = c)), record(c = c_obs))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("fixed-phase positive integer literal"),
        "unexpected message: {}",
        err.msg
    );
}

/// The Task-13 LKJCholesky anchor fixture: a FIXED `n = 3`, free `eta`, and a
/// free `3x3` `l_obs` scored variate (already itself a Cholesky factor).
const LKJ_CHOLESKY_DENSITY_SRC: &str = "\
n = 3
eta = elementof(posreals)
l_obs = elementof(cartpow(reals, [3, 3]))
l = draw(LKJCholesky(n = n, eta = eta))
lp = logdensityof(lawof(record(l = l)), record(l = l_obs))
";

/// §08 LKJCholesky, verbatim: `sum_{i=2}^{n} (n-i+2*eta-2) log L_ii - log
/// c_n(eta)`. Op counts: NO `stablehlo.cholesky` at all (the variate `L` is
/// already the Cholesky factor — [`lkj_cholesky_logpdf`]'s doc comment), two
/// `stablehlo.iota`/one `compare`/one `select` (one `diag` call, called
/// directly rather than through `log_det_from_chol`), one `stablehlo.reduce(`
/// (`diag`'s own row-sum; no further `reduce_sum` — [`vector_elem`] slices
/// two of the three diagonal entries individually instead of summing all
/// their logs), two `stablehlo.slice`/two `stablehlo.reshape` ([`vector_elem`]
/// called for `i = 2, 3`), four `chlo.lgamma` (`log_cn_lkj`'s `n = 3` loop,
/// same as [`emit_logdensity_lkj_has_expected_structure`]).
#[test]
fn emit_logdensity_lkj_cholesky_has_expected_structure() {
    let d = determinize_src(LKJ_CHOLESKY_DENSITY_SRC);
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
        out.contains("%arg0: tensor<f32>") && out.contains("%arg1: tensor<3x3xf32>"),
        "eta/l_obs must become scalar/matrix func args, in:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.cholesky"),
        "LKJCholesky's variate is already the Cholesky factor, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.iota").count(), 2);
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 1);
    assert_eq!(out.matches("stablehlo.slice").count(), 2);
    assert_eq!(out.matches("stablehlo.reshape").count(), 2);
    assert_eq!(out.matches("chlo.lgamma").count(), 4);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text: any drift (op count, ordering, arg naming,
/// formula) must be a deliberate, reviewed change to this golden file.
#[test]
fn emit_logdensity_lkj_cholesky_matches_frozen_golden() {
    let d = determinize_src(LKJ_CHOLESKY_DENSITY_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/lkjcholesky_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from the frozen golden (tests/goldens/lkjcholesky_logdensity.mlir)"
    );
}

/// A non-square scored variate `L` (`[2, 3]`) must refuse precisely, not
/// reach `Emitter::diag` on a non-square operand (which only asserts rank 2,
/// never squareness — see that function's doc comment).
#[test]
fn lkj_cholesky_logpdf_refuses_nonsquare_variate() {
    let src = "\
n = 3
eta = elementof(posreals)
l_obs = elementof(cartpow(reals, [2, 3]))
l = draw(LKJCholesky(n = n, eta = eta))
lp = logdensityof(lawof(record(l = l)), record(l = l_obs))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("LKJCholesky L must be an 3x3 matrix"),
        "unexpected message: {}",
        err.msg
    );
}

// ---- Task 14: straight-line continuous `@sample` batch + MvNormal ----------
//
// LogNormal/Exponential/Uniform/Cauchy/Logistic/Laplace/Weibull/Pareto give
// `registry.rs`'s "gamma-family"/"location-scale"/"remaining univariate"
// batches a straight-line inverse-CDF or reparameterization `@sample`
// builder (Gamma/InverseGamma/ChiSquared/Beta/StudentT/GeneralizedNormal/
// VonMises stay `sample: None` — no closed-form inverse-CDF, or need
// rejection sampling — see each batch's own doc comment); MvNormal gets
// `mu + cholesky(cov) @ z`. Same anchor-fixture shape as `NORMAL_SAMPLE_SRC`
// above: FIXED (literal, not `elementof`) hyperparameters via `s =
// rnginit(0)`/`draw(...)`/`rand(s, lawof(x))`, so every `func.func @sample`
// below is zero-arg — except MvNormal, whose `mu`/`cov` stay free
// (`elementof`-declared) parameters, exactly like `MVNORMAL_DENSITY_SRC`
// (Task 12), sidestepping any question about a literal nested-array (matrix)
// nested-vector lowering.
//
// Distributional correctness (KS statistic / moment match against scipy, at
// N = 100k draws per transform) is verified out-of-band (Task 14 report),
// NOT re-derived here: these tests only lock the STRUCTURE (op counts/kinds)
// and the frozen `.mlir` text, mirroring every `emit_logdensity_*` golden
// test's own division of labour (formula correctness is a paper/oracle
// derivation, structure+text is what regresses silently).

const LOGNORMAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(LogNormal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(x))
";

/// §08 LogNormal's `exp(mu + sigma * Z)` transform: exactly one
/// `stablehlo.rng` with `distribution = NORMAL`, and exactly one
/// `stablehlo.exponential` (the trailing `exp`).
#[test]
fn emit_sample_lognormal_has_expected_structure() {
    let d = determinize_src(LOGNORMAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing func.func @sample(%key: tensor<2xui64>) (no free params) in:\n{out}"
    );
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("chlo.erf_inv"));
    assert_eq!(
        out.matches("stablehlo.exponential").count(),
        1,
        "expected exactly one exp (the trailing exp(...)), in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_lognormal_matches_frozen_golden() {
    let d = determinize_src(LOGNORMAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/lognormal_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/lognormal_sample.mlir)"
    );
}

const EXPONENTIAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Exponential(rate = 1.0))
draws = rand(s, lawof(x))
";

/// §08 Exponential's `-log(U) / rate` transform: exactly one
/// `stablehlo.rng` with `distribution = UNIFORM`.
#[test]
fn emit_sample_exponential_has_expected_structure() {
    let d = determinize_src(EXPONENTIAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.log").count(), 1);
    assert_eq!(out.matches("stablehlo.negate").count(), 1);
    assert_eq!(out.matches("stablehlo.divide").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_exponential_matches_frozen_golden() {
    let d = determinize_src(EXPONENTIAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/exponential_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/exponential_sample.mlir)"
    );
}

const UNIFORM_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Uniform(support = interval(-1.0, 3.0)))
draws = rand(s, lawof(x))
";

/// §08 Uniform's `a + (b - a) * U` transform: exactly one `stablehlo.rng`
/// with `distribution = UNIFORM`, and the two folded bound constants
/// (`-1.0`, `4.0` = `3.0 - (-1.0)`) alongside it.
#[test]
fn emit_sample_uniform_has_expected_structure() {
    let d = determinize_src(UNIFORM_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    // Two multiplies (bits→uniform scale, the Uniform transform's `4.0 * u`)
    // and one add (the transform's `-1.0 + …`) — `Emitter::rng` returns the
    // standard uniform directly (no rng-affine identity ops), so the only
    // arithmetic left is Uniform's own transform. The exact text is pinned
    // by the frozen golden.
    assert_eq!(out.matches("stablehlo.multiply").count(), 2);
    assert_eq!(out.matches("stablehlo.add").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_uniform_matches_frozen_golden() {
    let d = determinize_src(UNIFORM_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/uniform_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/uniform_sample.mlir)"
    );
}

const CAUCHY_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Cauchy(location = 0.0, scale = 1.0))
draws = rand(s, lawof(x))
";

/// §08 Cauchy's `x0 + gamma * tan(pi * (U - 1/2))` transform: exactly one
/// `stablehlo.rng` with `distribution = UNIFORM`, and `tan` composed as
/// exactly one `stablehlo.sine` / `stablehlo.cosine` pair (no native `tan`
/// op — see [`Emitter::sin`]'s doc comment).
#[test]
fn emit_sample_cauchy_has_expected_structure() {
    let d = determinize_src(CAUCHY_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(
        out.matches("stablehlo.sine").count(),
        1,
        "expected exactly one sine (tan = sin/cos), in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.cosine").count(),
        1,
        "expected exactly one cosine (tan = sin/cos), in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.divide").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_cauchy_matches_frozen_golden() {
    let d = determinize_src(CAUCHY_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/cauchy_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/cauchy_sample.mlir)"
    );
}

const LOGISTIC_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Logistic(mu = 0.0, s = 1.0))
draws = rand(s, lawof(x))
";

/// §08 Logistic's `mu + s * log(U / (1 - U))` transform: exactly one
/// `stablehlo.rng` with `distribution = UNIFORM`.
#[test]
fn emit_sample_logistic_has_expected_structure() {
    let d = determinize_src(LOGISTIC_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.log").count(), 1);
    assert_eq!(out.matches("stablehlo.divide").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_logistic_matches_frozen_golden() {
    let d = determinize_src(LOGISTIC_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/logistic_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/logistic_sample.mlir)"
    );
}

const LAPLACE_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Laplace(location = 0.0, scale = 1.0))
draws = rand(s, lawof(x))
";

/// §08 Laplace's `mu - b * sgn(U - 1/2) * log(1 - 2|U - 1/2|)` transform:
/// exactly one `stablehlo.rng` with `distribution = UNIFORM`, and `sgn`
/// composed via exactly one `stablehlo.compare`/`stablehlo.select` pair (no
/// `stablehlo.sign` op — see [`laplace_sample`]'s doc comment).
#[test]
fn emit_sample_laplace_has_expected_structure() {
    let d = determinize_src(LAPLACE_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(
        out.matches("stablehlo.compare").count(),
        1,
        "expected exactly one compare (sgn's U - 1/2 >= 0 branch), in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.select").count(),
        1,
        "expected exactly one select (sgn's +-1 branch), in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.abs").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_laplace_matches_frozen_golden() {
    let d = determinize_src(LAPLACE_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/laplace_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/laplace_sample.mlir)"
    );
}

const WEIBULL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Weibull(shape = 2.0, scale = 3.0))
draws = rand(s, lawof(x))
";

/// §08 Weibull's `scale * (-log(U))^(1 / shape)` transform: exactly one
/// `stablehlo.rng` with `distribution = UNIFORM`, and exactly one
/// `stablehlo.power`.
#[test]
fn emit_sample_weibull_has_expected_structure() {
    let d = determinize_src(WEIBULL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.power").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_weibull_matches_frozen_golden() {
    let d = determinize_src(WEIBULL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/weibull_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/weibull_sample.mlir)"
    );
}

const PARETO_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Pareto(shape = 3.0, scale = 1.0))
draws = rand(s, lawof(x))
";

/// §08 Pareto's `scale * U^(-1 / shape)` transform: exactly one
/// `stablehlo.rng` with `distribution = UNIFORM`, and exactly one
/// `stablehlo.power`.
#[test]
fn emit_sample_pareto_has_expected_structure() {
    let d = determinize_src(PARETO_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.power").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_pareto_matches_frozen_golden() {
    let d = determinize_src(PARETO_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/pareto_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/pareto_sample.mlir)"
    );
}

/// The Task-14 MvNormal anchor fixture: a length-2 free `mu`/`cov` (exactly
/// [`MVNORMAL_DENSITY_SRC`]'s shape, minus the pinned observation — `@sample`
/// scores no variate), so `mu`/`cov` become `tensor<2xf32>`/`tensor<2x2xf32>`
/// func args rather than a literal nested-array (matrix) constant — see the
/// batch doc comment above.
const MVNORMAL_SAMPLE_SRC: &str = "\
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 2]))
s = rnginit(0)
x = draw(MvNormal(mu = mu, cov = cov))
draws = rand(s, lawof(x))
";

/// §08 MvNormal's `mu + cholesky(cov) @ z` transform: `mu`/`cov` become
/// `tensor<2xf32>`/`tensor<2x2xf32>` func args, exactly one `stablehlo.rng`
/// with `distribution = NORMAL` drawing a length-2 `z`, exactly one
/// `stablehlo.cholesky`, and exactly one `stablehlo.dot_general` (the
/// `matvec`).
#[test]
fn emit_sample_mvnormal_has_expected_structure() {
    let d = determinize_src(MVNORMAL_SAMPLE_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_sample(&d);

    assert!(
        out.contains("%arg0: tensor<2xf32>") && out.contains("%arg1: tensor<2x2xf32>"),
        "mu/cov must become vector/matrix func args, in:\n{out}"
    );
    assert!(out.contains("-> (tensor<2xf32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("chlo.erf_inv"));
    assert_eq!(out.matches("stablehlo.cholesky").count(), 1);
    assert_eq!(out.matches("stablehlo.dot_general").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_mvnormal_matches_frozen_golden() {
    let d = determinize_src(MVNORMAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/mvnormal_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/mvnormal_sample.mlir)"
    );
}

// ---- Task 15: rejection-based continuous `@sample` batch + Dirichlet -------
//
// Gamma/Beta/ChiSquared/StudentT/InverseGamma/GeneralizedNormal + Dirichlet
// get a `@sample` builder for the first time: a hand-emitted Marsaglia–Tsang
// Gamma rejection loop (`stablehlo.while`, via `Emitter::while_loop`) that
// every one of them reduces to (§08 equivalences — see `registry.rs`'s Task-15
// batch doc comment). Same anchor-fixture shape as `NORMAL_SAMPLE_SRC` (FIXED
// literal hyperparameters via `s = rnginit(0)`/`draw(...)`/`rand(s, lawof(x))`,
// so each `func.func @sample` is zero-arg) — except Dirichlet, whose `alpha`
// stays a free (`elementof`-declared) length-3 vector parameter, exactly like
// `DIRICHLET_DENSITY_SRC`, so it becomes a `tensor<3xf32>` func arg.
//
// The GAMMA-based ones assert exactly the `stablehlo.while` count (one per
// underlying Gamma: 1 for Gamma/ChiSquared/StudentT/InverseGamma/
// GeneralizedNormal, 2 for Beta's `X`/`Y`, 3 for Dirichlet's 3 components) plus
// the frozen `.mlir` text; distributional correctness (KS statistic vs
// scipy at N = 100k, plus Dirichlet per-component moments) is verified
// out-of-band (Task 15 report), NOT re-derived here — same division of labour
// as Task 14's straight-line batch (structure+text is what regresses
// silently; formula correctness is an oracle derivation).

const GAMMA_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Gamma(shape = 2.0, rate = 1.0))
draws = rand(s, lawof(x))
";

/// §08 Gamma's Marsaglia–Tsang rejection sampler: exactly one
/// `stablehlo.while` (the rejection loop) and three `stablehlo.rng` (the
/// pre-drawn `Z`/`U` candidate batches + the shape-`< 1` boost's `U0`).
#[test]
fn emit_sample_gamma_has_expected_structure() {
    let d = determinize_src(GAMMA_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing func.func @sample(%key: tensor<2xui64>) (no free params) in:\n{out}"
    );
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(
        out.matches("stablehlo.while").count(),
        1,
        "expected exactly one rejection loop, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng").count(),
        3,
        "expected Z + U candidate batches + boost U0, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.dynamic_slice").count(),
        2,
        "expected Z[i]/U[i] runtime indexing, in:\n{out}"
    );
    assert!(
        out.contains("chlo.erf_inv") && out.contains("stablehlo.rng_bit_generator"),
        "expected both a NORMAL (Z) and UNIFORM (U) batch, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_gamma_matches_frozen_golden() {
    let d = determinize_src(GAMMA_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/gamma_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/gamma_sample.mlir)"
    );
}

const BETA_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Beta(alpha = 2.0, beta = 3.0))
draws = rand(s, lawof(x))
";

/// §08 Beta's `X / (X + Y)`, `X ~ Gamma(alpha, 1)`, `Y ~ Gamma(beta, 1)`: TWO
/// `stablehlo.while` rejection loops (one per underlying Gamma), six
/// `stablehlo.rng` (each Gamma's `Z`/`U`/`U0`).
#[test]
fn emit_sample_beta_has_expected_structure() {
    let d = determinize_src(BETA_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(
        out.matches("stablehlo.while").count(),
        2,
        "expected two rejection loops (X and Y Gammas), in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.rng").count(), 6);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_beta_matches_frozen_golden() {
    let d = determinize_src(BETA_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/beta_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/beta_sample.mlir)"
    );
}

const CHI_SQUARED_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(ChiSquared(k = 3.0))
draws = rand(s, lawof(x))
";

/// §08 ChiSquared's `Gamma(k/2, 1/2)`: exactly one `stablehlo.while`.
#[test]
fn emit_sample_chi_squared_has_expected_structure() {
    let d = determinize_src(CHI_SQUARED_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.while").count(), 1);
    assert_eq!(out.matches("stablehlo.rng").count(), 3);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_chi_squared_matches_frozen_golden() {
    let d = determinize_src(CHI_SQUARED_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/chi_squared_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/chi_squared_sample.mlir)"
    );
}

const STUDENTT_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(StudentT(nu = 5.0))
draws = rand(s, lawof(x))
";

/// §08 StudentT's `Z / sqrt(V / nu)`, `V ~ ChiSquared(nu)`: exactly one
/// `stablehlo.while` (the ChiSquared/Gamma loop) and four `stablehlo.rng`
/// (the Gamma's `Z`/`U`/`U0` + StudentT's own standard-normal `Z`).
#[test]
fn emit_sample_studentt_has_expected_structure() {
    let d = determinize_src(STUDENTT_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.while").count(), 1);
    assert_eq!(out.matches("stablehlo.rng").count(), 4);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_studentt_matches_frozen_golden() {
    let d = determinize_src(STUDENTT_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/studentt_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/studentt_sample.mlir)"
    );
}

const INVERSE_GAMMA_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(InverseGamma(shape = 3.0, scale = 1.0))
draws = rand(s, lawof(x))
";

/// §08 InverseGamma's `1 / Gamma(shape, rate = scale)`: exactly one
/// `stablehlo.while`, and the trailing reciprocal (`divide` of `1` by the
/// Gamma draw).
#[test]
fn emit_sample_inverse_gamma_has_expected_structure() {
    let d = determinize_src(INVERSE_GAMMA_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.while").count(), 1);
    assert_eq!(out.matches("stablehlo.rng").count(), 3);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_inverse_gamma_matches_frozen_golden() {
    let d = determinize_src(INVERSE_GAMMA_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/inverse_gamma_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/inverse_gamma_sample.mlir)"
    );
}

const GENERALIZED_NORMAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(GeneralizedNormal(mean = 0.0, alpha = 1.0, beta = 2.0))
draws = rand(s, lawof(x))
";

/// §08 GeneralizedNormal's `mean + alpha * sgn(U - 1/2) * Gamma(1/beta,
/// 1)^(1/beta)`: exactly one `stablehlo.while` (the Gamma loop), four
/// `stablehlo.rng` (the Gamma's `Z`/`U`/`U0` + the sign's `U`), and `sgn`
/// composed via one `compare`/`select` pair (like [`laplace_sample`]).
#[test]
fn emit_sample_generalized_normal_has_expected_structure() {
    let d = determinize_src(GENERALIZED_NORMAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.while").count(), 1);
    assert_eq!(out.matches("stablehlo.rng").count(), 4);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_generalized_normal_matches_frozen_golden() {
    let d = determinize_src(GENERALIZED_NORMAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/generalized_normal_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/generalized_normal_sample.mlir)"
    );
}

/// The Task-15 Dirichlet anchor fixture: a free (`elementof`-declared)
/// length-3 `alpha` (exactly [`DIRICHLET_DENSITY_SRC`]'s shape, minus the
/// pinned observation — `@sample` scores no variate), so `alpha` becomes a
/// `tensor<3xf32>` func arg and the sampler unrolls into one Gamma draw per
/// component.
const DIRICHLET_SAMPLE_SRC: &str = "\
alpha = elementof(cartpow(posreals, 3))
s = rnginit(0)
x = draw(Dirichlet(alpha = alpha))
draws = rand(s, lawof(x))
";

/// §08 Dirichlet's `g_i ~ Gamma(alpha_i, 1)`, return `g / sum(g)`: `alpha`
/// becomes a `tensor<3xf32>` func arg, one `stablehlo.while` PER component
/// (three, statically unrolled), returning a normalized `tensor<3xf32>`.
#[test]
fn emit_sample_dirichlet_has_expected_structure() {
    let d = determinize_src(DIRICHLET_SAMPLE_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );

    let out = emit_sample(&d);

    assert!(
        out.contains("%arg0: tensor<3xf32>"),
        "alpha must become a length-3 vector func arg, in:\n{out}"
    );
    assert!(out.contains("-> (tensor<3xf32>, tensor<2xui64>)"));
    assert_eq!(
        out.matches("stablehlo.while").count(),
        3,
        "expected one rejection loop per Dirichlet component, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.rng").count(), 9);
    assert!(
        out.contains("stablehlo.concatenate"),
        "expected the per-component Gamma draws packed into a vector, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_dirichlet_matches_frozen_golden() {
    let d = determinize_src(DIRICHLET_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/dirichlet_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/dirichlet_sample.mlir)"
    );
}

/// The `emit_sample` mirror of `mvnormal_logpdf_refuses_dynamic_vector_length`:
/// an `alpha` whose vector length is NOT statically known (`cartpow(posreals,
/// m)` for a free `m`, not a literal size) must refuse precisely — `n` is
/// unrolled into `n` separate Gamma draws at emit time (one [`draw_gamma`]
/// call per component), so a dynamic length has no lowering, not merely an
/// inconvenient one.
#[test]
fn dirichlet_sample_refuses_dynamic_vector_length() {
    let src = "\
m = elementof(posintegers)
alpha = elementof(cartpow(posreals, m))
s = rnginit(0)
x = draw(Dirichlet(alpha = alpha))
draws = rand(s, lawof(x))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("Dirichlet sample needs a statically-known vector length"),
        "unexpected message: {}",
        err.msg
    );
    assert!(
        err.node.is_some(),
        "expected the refusal localized to the 'alpha' node, got node: None"
    );
}

/// The `emit_sample` mirror of `mvnormal_logpdf_refuses_dynamic_vector_length`
/// for rank, not length: a rank-2 `alpha` (`cartpow(posreals, [2, 2])`) must
/// refuse precisely, not reach [`vector_elem`]'s slice+reshape idiom on an
/// operand it was never built to accept.
#[test]
fn dirichlet_sample_refuses_nonrank1_alpha() {
    let src = "\
alpha = elementof(cartpow(posreals, [2, 2]))
s = rnginit(0)
x = draw(Dirichlet(alpha = alpha))
draws = rand(s, lawof(x))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("Dirichlet sample: 'alpha' must be a rank-1 vector"),
        "unexpected message: {}",
        err.msg
    );
    assert!(
        err.node.is_some(),
        "expected the refusal localized to the 'alpha' node, got node: None"
    );
}

// ---- Task 16: discrete + Multinomial `@sample` batch, refuse-7 finalized ---
//
// Bernoulli/Geometric/Categorical/Categorical0/Binomial/Poisson/
// NegativeBinomial/NegativeBinomial2 + Multinomial — the last `@sample`
// batch, completing the §08 sampler set (see `registry.rs`'s Task-16 batch
// doc comment for the three sampler shapes: straight-line, Poisson's bounded
// inverse-CDF `while`, and the Gamma-Poisson mixture). Every fixture below
// uses FIXED literal hyperparameters (no `elementof`), so `emit_sample`
// produces a zero-arg `func.func @sample()` — same convention as
// `NORMAL_SAMPLE_SRC`/`GAMMA_SAMPLE_SRC`, not the free-parameter convention
// `MVNORMAL_SAMPLE_SRC`/`DIRICHLET_SAMPLE_SRC` use. Categorical/Categorical0's
// `p` is a literal array (`[0.2, 0.3, 0.5]`), same reasoning as
// `CATEGORICAL_DENSITY_SRC`'s (the `stdsimplex` typing gap noted there);
// Binomial/Multinomial's `n` is a FIXED top-level literal binding (`n = 5`),
// same convention as `LKJ_DENSITY_SRC`'s `n = 3` (`literal_fixed_positive_int`
// needs it at EMIT time, not merely well-typed — see `registry.rs`'s doc
// comment on that helper, fixed in this same batch to say "sample" rather
// than a hardcoded "logdensity" when raised from these two builders).

const BERNOULLI_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Bernoulli(p = 0.3))
draws = rand(s, lawof(x))
";

/// §08 Bernoulli's `select(U < p, 1, 0)`: exactly one `stablehlo.rng`
/// (`distribution = UNIFORM`), one `stablehlo.compare`, one
/// `stablehlo.select`.
#[test]
fn emit_sample_bernoulli_has_expected_structure() {
    let d = determinize_src(BERNOULLI_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_bernoulli_matches_frozen_golden() {
    let d = determinize_src(BERNOULLI_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/bernoulli_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/bernoulli_sample.mlir)"
    );
}

const GEOMETRIC_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Geometric(p = 0.3))
draws = rand(s, lawof(x))
";

/// §08 Geometric's `floor(log(U) / log(1 - p))`: exactly one `stablehlo.rng`
/// (`distribution = UNIFORM`), two `stablehlo.log` (`log(U)`, `log(1-p)`),
/// one `stablehlo.floor` — the only discrete sampler needing it.
#[test]
fn emit_sample_geometric_has_expected_structure() {
    let d = determinize_src(GEOMETRIC_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.log").count(), 2);
    assert_eq!(out.matches("stablehlo.floor").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_geometric_matches_frozen_golden() {
    let d = determinize_src(GEOMETRIC_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/geometric_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/geometric_sample.mlir)"
    );
}

const CATEGORICAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Categorical(p = [0.2, 0.3, 0.5]))
draws = rand(s, lawof(x))
";

/// §08 Categorical's (1-based) shared [`draw_categorical`] inverse-CDF index
/// draw: length-3 `p` unrolls into `n - 1 = 2` prefix-sum comparisons —
/// exactly one `stablehlo.rng` (`distribution = UNIFORM`), two
/// `stablehlo.compare`, two `stablehlo.select`.
#[test]
fn emit_sample_categorical_has_expected_structure() {
    let d = determinize_src(CATEGORICAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.compare").count(), 2);
    assert_eq!(out.matches("stablehlo.select").count(), 2);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_categorical_matches_frozen_golden() {
    let d = determinize_src(CATEGORICAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/categorical_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/categorical_sample.mlir)"
    );
}

const CATEGORICAL0_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Categorical0(p = [0.2, 0.3, 0.5]))
draws = rand(s, lawof(x))
";

/// The `base = 0.0` mirror of [`emit_sample_categorical_has_expected_structure`]
/// — identical op counts, differing only in the returned `base` constant
/// (checked by the frozen-golden test below).
#[test]
fn emit_sample_categorical0_has_expected_structure() {
    let d = determinize_src(CATEGORICAL0_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.compare").count(), 2);
    assert_eq!(out.matches("stablehlo.select").count(), 2);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_categorical0_matches_frozen_golden() {
    let d = determinize_src(CATEGORICAL0_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/categorical0_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/categorical0_sample.mlir)"
    );
}

const BINOMIAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
n = 5
x = draw(Binomial(n = n, p = 0.3))
draws = rand(s, lawof(x))
";

/// §08 Binomial's exact `sum of n Bernoulli(p)`: a FIXED `n = 5` drives a
/// single length-5 `stablehlo.rng` (`distribution = UNIFORM`), one
/// `stablehlo.broadcast_in_dim` (`p` broadcast to the batch shape), one
/// `stablehlo.compare`, one `stablehlo.select`, and one `stablehlo.reduce(`
/// (the `reduce_sum`).
#[test]
fn emit_sample_binomial_has_expected_structure() {
    let d = determinize_src(BINOMIAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert!(out.contains("tensor<5xf32>"));
    // One broadcast_in_dim: `p` broadcast to the batch shape by the sampler.
    // `Emitter::rng` returns the standard uniform directly, so there are no
    // rng-affine scalar bounds left to lift onto the batch.
    assert_eq!(out.matches("stablehlo.broadcast_in_dim").count(), 1);
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert_eq!(out.matches("stablehlo.reduce(").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_binomial_matches_frozen_golden() {
    let d = determinize_src(BINOMIAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/binomial_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/binomial_sample.mlir)"
    );
}

/// A non-literal (`elementof`-declared) `n` must refuse precisely — Binomial
/// sample's uniform batch needs `n` as a Rust `u64` at EMIT time to size a
/// static-length `tensor<NxT>`, not merely a well-typed runtime value. Mirrors
/// `lkj_logpdf_refuses_parameterized_n`'s guard on the `@logdensity` side.
#[test]
fn binomial_sample_refuses_parameterized_n() {
    let src = "\
n = elementof(posintegers)
s = rnginit(0)
x = draw(Binomial(n = n, p = 0.3))
draws = rand(s, lawof(x))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("Binomial sample needs a fixed-phase positive integer literal"),
        "unexpected message: {}",
        err.msg
    );
}

const POISSON_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Poisson(rate = 4.0))
draws = rand(s, lawof(x))
";

/// §08 Poisson's bounded inverse-CDF sampler ([`draw_poisson`]): exactly one
/// `stablehlo.rng` (the single pre-loop `U`, `distribution = UNIFORM`) and
/// exactly one `stablehlo.while` (the incremental-CDF walk) — no second `rng`
/// inside the loop (CDF inversion of a SINGLE uniform, unlike the Gamma
/// rejection loop's per-iteration batches).
#[test]
fn emit_sample_poisson_has_expected_structure() {
    let d = determinize_src(POISSON_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.while").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_poisson_matches_frozen_golden() {
    let d = determinize_src(POISSON_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/poisson_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/poisson_sample.mlir)"
    );
}

const NEGATIVE_BINOMIAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(NegativeBinomial(alpha = 5.0, beta = 2.0))
draws = rand(s, lawof(x))
";

/// §08 NegativeBinomial's Gamma-Poisson mixture: [`draw_gamma`] (Task 15,
/// three `stablehlo.rng` — `Z`/`U`/boost `U0` — plus one `stablehlo.while`)
/// feeding [`draw_poisson`] (one more `stablehlo.rng` plus one more
/// `stablehlo.while`) — four `stablehlo.rng` and two `stablehlo.while` total.
#[test]
fn emit_sample_negative_binomial_has_expected_structure() {
    let d = determinize_src(NEGATIVE_BINOMIAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(
        out.matches("stablehlo.rng").count(),
        4,
        "expected Gamma's Z/U/U0 + Poisson's U, in:\n{out}"
    );
    assert!(
        out.contains("chlo.erf_inv") && out.contains("stablehlo.rng_bit_generator"),
        "expected both a NORMAL (Gamma's Z) and UNIFORM batch, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.while").count(),
        2,
        "expected the Gamma rejection loop plus the Poisson inverse-CDF loop, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_negative_binomial_matches_frozen_golden() {
    let d = determinize_src(NEGATIVE_BINOMIAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/negative_binomial_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/negative_binomial_sample.mlir)"
    );
}

const NEGATIVE_BINOMIAL2_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(NegativeBinomial2(mu = 3.0, psi = 5.0))
draws = rand(s, lawof(x))
";

/// §08 NegativeBinomial2's Gamma-Poisson mixture: same op-count shape as
/// [`emit_sample_negative_binomial_has_expected_structure`], plus the extra
/// `stablehlo.divide` computing `rate = psi / mu` before [`draw_gamma`].
#[test]
fn emit_sample_negative_binomial2_has_expected_structure() {
    let d = determinize_src(NEGATIVE_BINOMIAL2_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(out.contains("-> (tensor<f32>, tensor<2xui64>)"));
    assert_eq!(out.matches("stablehlo.rng").count(), 4);
    assert!(out.contains("chlo.erf_inv") && out.contains("stablehlo.rng_bit_generator"));
    assert_eq!(out.matches("stablehlo.while").count(), 2);
    assert!(
        out.matches("stablehlo.divide").count() >= 1,
        "expected at least the rate = psi / mu divide, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_negative_binomial2_matches_frozen_golden() {
    let d = determinize_src(NEGATIVE_BINOMIAL2_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/negative_binomial2_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/negative_binomial2_sample.mlir)"
    );
}

const MULTINOMIAL_SAMPLE_SRC: &str = "\
s = rnginit(0)
n = 4
x = draw(Multinomial(n = n, p = [0.2, 0.3, 0.5]))
draws = rand(s, lawof(x))
";

/// §08 Multinomial's bounded `while` over `n = 4` Categorical(p) draws
/// (`p` length-3, so a length-3 count vector): exactly one `stablehlo.rng`
/// (the length-4 pre-drawn uniform batch, `distribution = UNIFORM`), one
/// `stablehlo.while` (the `n`-bounded accumulation loop), one
/// `stablehlo.dynamic_slice` (indexing the pre-drawn batch by the loop
/// counter), returning a `tensor<3xf32>` count vector.
#[test]
fn emit_sample_multinomial_has_expected_structure() {
    let d = determinize_src(MULTINOMIAL_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(out.contains("func.func @sample(%key: tensor<2xui64>)"));
    assert!(
        out.contains("-> (tensor<3xf32>, tensor<2xui64>)"),
        "must return a length-3 count vector, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.rng").count(), 1);
    assert!(out.contains("stablehlo.rng_bit_generator"));
    assert!(out.contains("tensor<4xf32>"));
    assert_eq!(out.matches("stablehlo.while").count(), 1);
    assert_eq!(out.matches("stablehlo.dynamic_slice").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emit_sample_multinomial_matches_frozen_golden() {
    let d = determinize_src(MULTINOMIAL_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/multinomial_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted @sample drifted from the frozen golden (tests/goldens/multinomial_sample.mlir)"
    );
}

/// A non-literal (`elementof`-declared) `n` must refuse precisely — same
/// reasoning as [`binomial_sample_refuses_parameterized_n`]: Multinomial's `n`
/// sizes both the pre-drawn uniform batch and the `while` bound, so it must
/// be known at EMIT time.
#[test]
fn multinomial_sample_refuses_parameterized_n() {
    let src = "\
n = elementof(posintegers)
s = rnginit(0)
x = draw(Multinomial(n = n, p = [0.2, 0.3, 0.5]))
draws = rand(s, lawof(x))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("Multinomial sample needs a fixed-phase positive integer literal"),
        "unexpected message: {}",
        err.msg
    );
}

/// The `emit_sample` mirror of `dirichlet_sample_refuses_dynamic_vector_length`
/// for [`multinomial_sample`]'s own vector-length guard (structurally the
/// same "statically-known length" check [`draw_categorical`] uses, but a
/// separate guard site in [`multinomial_sample`] itself): a `p` whose vector
/// length is NOT statically known must refuse precisely, not reach
/// [`vector_elem`] on an operand with no static length to unroll against.
#[test]
fn multinomial_sample_refuses_dynamic_vector_length() {
    let src = "\
n = 4
m = elementof(posintegers)
p = elementof(cartpow(unitinterval, m))
s = rnginit(0)
x = draw(Multinomial(n = n, p = p))
draws = rand(s, lawof(x))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("Multinomial sample needs a statically-known vector length"),
        "unexpected message: {}",
        err.msg
    );
    assert!(
        err.node.is_some(),
        "expected the refusal localized to the 'p' node, got node: None"
    );
}

/// The rank mirror of [`multinomial_sample_refuses_dynamic_vector_length`]: a
/// rank-2 `p` must refuse precisely, not reach [`vector_elem`]'s slice+reshape
/// idiom on an operand it was never built to accept.
#[test]
fn multinomial_sample_refuses_nonrank1_p() {
    let src = "\
n = 4
p = elementof(cartpow(unitinterval, [3, 1]))
s = rnginit(0)
x = draw(Multinomial(n = n, p = p))
draws = rand(s, lawof(x))
";
    let d = determinize_src(src);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::Sample,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg
            .contains("Multinomial sample: 'p' must be a rank-1 vector"),
        "unexpected message: {}",
        err.msg
    );
    assert!(
        err.node.is_some(),
        "expected the refusal localized to the 'p' node, got node: None"
    );
}

// ---- Task 16: refuse-7 finalized -------------------------------------------
//
// The seven distributions with NO `@sample` builder, confirmed final: five
// registered-but-`sample: None` (VonMises, Wishart, InverseWishart, LKJ,
// LKJCholesky — each needs its own dedicated sampler design, none planned in
// this batch) plus two never registered at all (PoissonProcess/
// BinnedPoissonProcess — point-process measures with no `@logdensity`
// builder either; spec §08). `VonMises` already has a locking test
// (`builtin_sample_refuses_registered_ctor_without_sample_builder`, Task 7)
// for the SHARED `dist.sample.ok_or_else` code path — not duplicated here.
// The six tests below each lock one of the remaining six distributions to
// its own specific ctor name, so a future accidental sampler registration for
// any ONE of them (not just the shared code path) is caught.

/// `Wishart` is registered (`@logdensity`, Task 13) but has no `@sample`
/// builder (`sample: None`).
#[test]
fn builtin_sample_refuses_wishart_without_sample_builder() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "Wishart");
    let nu = real(&mut m, 5.0);
    let scale = real(&mut m, 1.0); // stand-in; lookup fails before params are read
    let kernel_input = record_node(&mut m, &[("nu", nu), ("scale", scale)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no @sample lowering for 'Wishart'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `InverseWishart` is registered (`@logdensity`, Task 13) but has no
/// `@sample` builder (`sample: None`).
#[test]
fn builtin_sample_refuses_inverse_wishart_without_sample_builder() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "InverseWishart");
    let nu = real(&mut m, 5.0);
    let psi = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("nu", nu), ("psi", psi)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no @sample lowering for 'InverseWishart'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `LKJ` is registered (`@logdensity`, Task 13) but has no `@sample` builder
/// (`sample: None`).
#[test]
fn builtin_sample_refuses_lkj_without_sample_builder() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "LKJ");
    let n = real(&mut m, 3.0);
    let eta = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("n", n), ("eta", eta)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no @sample lowering for 'LKJ'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `LKJCholesky` is registered (`@logdensity`, Task 13) but has no `@sample`
/// builder (`sample: None`).
#[test]
fn builtin_sample_refuses_lkj_cholesky_without_sample_builder() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "LKJCholesky");
    let n = real(&mut m, 3.0);
    let eta = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("n", n), ("eta", eta)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no @sample lowering for 'LKJCholesky'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

// ---- rng-threaded rand: chained @sample regression golden ------------------
//
// Two SEPARATE destructured `rand`s where the second consumes the first's
// advanced rng (`crates/determinizer/tests/sample_golden.rs`'s
// `chained_rand_threads_advanced_rng_not_source`, minus the record wrapping —
// bare `lawof(x)` destructures the same way, see `lower_measure_sample`'s
// "draw" dispatch arm). Guards the threaded-key ABI (Tasks 6-7) against a
// regression where the second sample re-reads the source `%key` instead of
// the first sample's advanced state.
const CHAINED_RAND_SAMPLE_SRC: &str = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = 1.0, sigma = 1.0))
d1, s2 = rand(s, lawof(x))
d2, s3 = rand(s2, lawof(y))
out = d2
";

/// Freeze the exact emitted text: any drift (op count, ordering, key
/// threading, arg naming) must be a deliberate, reviewed change to this
/// golden file.
#[test]
fn emit_sample_chained_rand_matches_frozen_golden() {
    let d = determinize_src(CHAINED_RAND_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/chained_rand_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted chained @sample drifted from the frozen golden (tests/goldens/chained_rand_sample.mlir)"
    );
}

/// The structural threading guarantee the golden above freezes textually:
/// TWO `rng_bit_generator` draws (one per `rand`), and the SECOND's key
/// operand is the FIRST's advanced `%state` result — not the original
/// `%key` func argument — which is what proves the chain actually threads
/// rather than each sample independently reading the source key. The
/// `func.func` also returns that same advanced state as its final key
/// result (spec §07: `@sample` returns the LAST advanced rngstate).
#[test]
fn emit_sample_chained_rand_second_draw_consumes_first_advanced_key() {
    let d = determinize_src(CHAINED_RAND_SAMPLE_SRC);
    let out = emit_sample(&d);

    let gens: Vec<&str> = out
        .lines()
        .filter(|l| l.contains("stablehlo.rng_bit_generator"))
        .collect();
    assert_eq!(
        gens.len(),
        2,
        "expected exactly two threaded rng_bit_generator draws, in:\n{out}"
    );

    // `%state, %bits = stablehlo.rng_bit_generator %keyoperand, algorithm = ...`
    // — the key operand is the token right after the op name.
    fn key_operand(line: &str) -> &str {
        line.split("stablehlo.rng_bit_generator ")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
    }
    // `%state, %bits = stablehlo.rng_bit_generator ...` — the state result is
    // the first of the two comma-separated SSA names before `=`.
    fn state_result(line: &str) -> &str {
        line.trim_start()
            .split(" =")
            .next()
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .trim()
    }

    assert_eq!(
        key_operand(gens[0]),
        "%key",
        "first draw must consume the source %key, in:\n{out}"
    );
    let first_state = state_result(gens[0]);
    let second_key_operand = key_operand(gens[1]);
    assert_eq!(
        second_key_operand, first_state,
        "second draw must consume the FIRST draw's advanced state, not the \
         source %key, in:\n{out}"
    );
    assert_ne!(
        second_key_operand, "%key",
        "second draw must NOT re-read the source %key, in:\n{out}"
    );

    // The func's final key result is the SECOND (last) draw's advanced state.
    let return_line = out
        .lines()
        .find(|l| l.trim_start().starts_with("return"))
        .expect("missing return");
    let second_state = state_result(gens[1]);
    assert!(
        return_line.contains(second_state),
        "return must thread out the LAST draw's advanced key ({second_state}), in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// `PoissonProcess` (spec §08) is never registered at all — no
/// `@logdensity` builder either, so it hits the SAME `registry::lookup` miss
/// [`builtin_sample_refuses_unregistered_ctor`] exercises via `Bogus`, but
/// pinned to this specific real distribution name so a future
/// logdensity-only registration (which would leave `@sample` still
/// unreachable via a DIFFERENT message, `"no @sample lowering for ..."`) is
/// caught by this test starting to fail.
#[test]
fn builtin_sample_refuses_poisson_process_unregistered() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "PoissonProcess");
    let intensity = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("intensity", intensity)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("no lowering for distribution 'PoissonProcess'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// The `BinnedPoissonProcess` mirror of
/// [`builtin_sample_refuses_poisson_process_unregistered`].
#[test]
fn builtin_sample_refuses_binned_poisson_process_unregistered() {
    let mut m = Module::new();
    let rng = real(&mut m, 0.0);
    let ctor = const_node(&mut m, "BinnedPoissonProcess");
    let bins = real(&mut m, 1.0);
    let intensity = real(&mut m, 1.0);
    let kernel_input = record_node(&mut m, &[("bins", bins), ("intensity", intensity)]);
    let node = call(&mut m, "builtin_sample", &[rng, ctor, kernel_input]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("no lowering for distribution 'BinnedPoissonProcess'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}
// ---- Task 9: fan-out Tier 1 (shaped iid straight-line draws) ----------------
//
// A fanned `iid(K, n)` sample (fixed straight-line kernel `K`, static `n`)
// draws `n` iid values with ONE `rng_bit_generator` advance sized to `[n]`
// (spec §07 size dims): the determiniser emits `builtin_sample(rng, ctor,
// input, n)` (Task 4), `lower_sample` sets a `[n]` batch shape around the
// straight-line builder, `Emitter::rng` draws the batch, and the builder's
// scalar params broadcast over it via `Emitter::binary`'s auto-broadcast. Both
// goldens below are parse-validated against the real StableHLO parser
// (jax 0.10.2). Rejection/multivariate fan-out (Tier 2) refuses here.

/// A fanned scalar-Normal iid draw: `iid(Normal(mu=0, sigma=1), 4)`, sampled
/// value-terminal (`draws = rand(s, lawof(xs))`). Fixed hyperparameters, so no
/// free-param args beyond `%key`.
const NORMAL_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Normal(mu = 0.0, sigma = 1.0), 4)
draws = rand(s, lawof(xs))
";

/// A fanned inverse-CDF iid draw: `iid(Exponential(rate=2), 4)` (`-log(U)/rate`).
const EXPONENTIAL_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Exponential(rate = 2.0), 4)
draws = rand(s, lawof(xs))
";

/// The brief's Step-2 structural check: the fanned draw returns a
/// `tensor<4xf32>` value (not a scalar) alongside the advanced key, with
/// EXACTLY ONE `rng_bit_generator` — sized to the `[4]` batch — for the whole
/// iid array, and the scalar params broadcast over that batch.
#[test]
fn emit_sample_normal_iid_has_expected_structure() {
    let d = determinize_src(NORMAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing @sample(%key) (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned draw must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<2xui64>, tensor<4xui32>)"),
        "the single rng_bit_generator must be sized to the [4] batch, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.broadcast_in_dim"),
        "scalar params must broadcast over the batch, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_normal_iid_matches_frozen_golden() {
    let d = determinize_src(NORMAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/normal_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/normal_iid_sample.mlir"
    );
}

/// The inverse-CDF fan-out mirror of the Normal case: `Exponential`'s
/// `-log(U)/rate` batched to `[4]` with one rng_bit_generator advance.
#[test]
fn emit_sample_exponential_iid_matches_frozen_golden() {
    let d = determinize_src(EXPONENTIAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Exponential must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );

    let golden = include_str!("goldens/exponential_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/exponential_iid_sample.mlir"
    );
}

// ---- Task 10a: fan-out Tier 2 (batched rejection — Gamma family) ------------
//
// A fanned `iid(K, n)` sample whose kernel `K` is a Marsaglia–Tsang rejection
// sampler (Gamma, or a reducer that composes it — ChiSquared/StudentT/
// InverseGamma/Beta/GeneralizedNormal) draws `n` iid values via
// `draw_gamma_batched`'s PER-LANE masked `stablehlo.while`: a `tensor<n×i1>`
// accept mask keeps each lane's FIRST accepted candidate and redraws only the
// still-rejected lanes, looping until `all(accepted)` (or MAXITER). Candidates
// are pre-drawn OUTSIDE the loop at `[MAXITER, n]` (fixed key advance →
// reproducible), read one `[n]` row per iteration by `dynamic_slice_row`. Both
// frozen goldens below are parse-validated against the real StableHLO parser
// (jax 0.10.2). Vector-variate Dirichlet and the discrete samplers were still
// refusing at this point in the file (Dirichlet is batched in the Tier-2
// multivariate batch, the discrete ones in the later Tier-3/Tier-4 batches,
// both below).

/// A fanned rejection draw: `iid(Gamma(shape=2, rate=1), 4)`, value-terminal.
const GAMMA_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Gamma(shape = 2.0, rate = 1.0), 4)
draws = rand(s, lawof(xs))
";

/// The strongest reducer: `X / (X + Y)`, TWO independent batched Gamma draws
/// (→ TWO masked `while`s) plus the elementwise ratio, all at `[4]`.
const BETA_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Beta(alpha = 2.0, beta = 3.0), 4)
draws = rand(s, lawof(xs))
";

/// The batched Gamma draw returns a `tensor<4xf32>` (not a scalar) alongside
/// the advanced key, via exactly ONE masked `stablehlo.while` carrying a
/// `tensor<4xi1>` per-lane accept mask, reading its pre-drawn `[MAXITER, 4]`
/// candidate batch with `dynamic_slice`.
#[test]
fn emit_sample_gamma_iid_has_expected_structure() {
    let d = determinize_src(GAMMA_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing @sample(%key) (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "batched Gamma must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.while").count(),
        1,
        "one masked while for the whole [4] batch, in:\n{out}"
    );
    assert!(
        out.contains("tensor<4xi1>"),
        "per-lane accept mask must be tensor<4xi1>, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.reduce(") && out.contains("stablehlo.or "),
        "the masked loop needs an all-reduce over the accept mask + an OR carry, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_gamma_iid_matches_frozen_golden() {
    let d = determinize_src(GAMMA_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/gamma_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted batched-Gamma @sample drifted from tests/goldens/gamma_iid_sample.mlir"
    );
}

/// Beta fans out to TWO batched Gammas (its `X / (X + Y)`): two masked
/// `while`s, one `[4]` value return, one advanced key.
#[test]
fn emit_sample_beta_iid_matches_frozen_golden() {
    let d = determinize_src(BETA_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "batched Beta must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.while").count(),
        2,
        "two masked whiles (Beta = ratio of two batched Gammas), in:\n{out}"
    );

    let golden = include_str!("goldens/beta_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted batched-Beta @sample drifted from tests/goldens/beta_iid_sample.mlir"
    );
}

/// Every reducer that composes `draw_gamma_batched` fans out to a `[4]` batch
/// value with a `tensor<4xi1>` per-lane accept mask — proving the WHOLE
/// rejection family (not just Gamma/Beta) is batch-correct. GeneralizedNormal
/// additionally exercises the now-auto-broadcasting `compare`/`select` (its
/// per-lane `sgn`).
#[test]
fn emit_sample_iid_gamma_reducers_fan_out() {
    let cases = [
        (
            "ChiSquared",
            "s = rnginit(0)\nxs ~ iid(ChiSquared(k = 3.0), 4)\ndraws = rand(s, lawof(xs))\n",
        ),
        (
            "StudentT",
            "s = rnginit(0)\nxs ~ iid(StudentT(nu = 5.0), 4)\ndraws = rand(s, lawof(xs))\n",
        ),
        (
            "InverseGamma",
            "s = rnginit(0)\nxs ~ iid(InverseGamma(shape = 3.0, scale = 1.0), 4)\ndraws = rand(s, lawof(xs))\n",
        ),
        (
            "GeneralizedNormal",
            "s = rnginit(0)\nxs ~ iid(GeneralizedNormal(mean = 0.0, alpha = 1.0, beta = 2.0), 4)\ndraws = rand(s, lawof(xs))\n",
        ),
    ];
    for (name, src) in cases {
        let d = determinize_src(src);
        let out = emit_sample(&d);
        assert!(
            out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
            "{name} iid must return the [4] batch + advanced key, in:\n{out}"
        );
        assert!(
            out.contains("tensor<4xi1>"),
            "{name} iid must carry a [4] per-lane accept mask, in:\n{out}"
        );
        assert!(is_delimiter_balanced(&out), "{name} iid unbalanced:\n{out}");
    }
}

// ---- Task 10b: fan-out Tier 2 (batched multivariate — MvNormal) -------------
//
// A fanned `iid(MvNormal(mu, cov), n)` draws `n` iid `d`-vectors as a rank-2
// `[n, d]` batch: ONE `stablehlo.cholesky` on the shared `[d, d]` cov, ONE
// `rng_bit_generator` advance sized to `[n, d]` (a genuine `tensor<n×d>` draw —
// the n rows are independent, NOT a `[d]` draw broadcast across n), and the
// row-wise affine `mu + L·z_i` for all rows as a batched `dot_general`
// (`z · Lᵀ`, `contracting_dims = [1] x [1]`), with `mu` broadcast across the
// rows (`broadcast_in_dim dims = [1]`). The frozen golden below is
// parse-validated against the real StableHLO parser (jax 0.10.2). The scalar
// MvNormal path (`emit_sample_mvnormal_*`) is byte-identical to before.
// Dirichlet (the other vector-variate sampler) is batched just below, reusing
// the batched Gamma per component + a new `stablehlo.transpose`.

/// A fanned multivariate draw: `iid(MvNormal(mu, cov), 3)` at `d = 2`, with
/// free `mu`/`cov` (so `@sample(%key, %arg0, %arg1)`).
const MVNORMAL_IID_SAMPLE_SRC: &str = "\
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 2]))
s = rnginit(0)
xs ~ iid(MvNormal(mu = mu, cov = cov), 3)
draws = rand(s, lawof(xs))
";

/// The batched MvNormal draw returns a `tensor<3x2xf32>` (`[n, d]`, not a bare
/// `[d]` vector) alongside the advanced key, via exactly ONE
/// `stablehlo.cholesky` and ONE `rng_bit_generator` sized to the `[3, 2]` batch
/// (row-independence), plus the batched affine (`dot_general` + `mu`
/// broadcast).
#[test]
fn emit_sample_mvnormal_iid_has_expected_structure() {
    let d = determinize_src(MVNORMAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains(
            "func.func @sample(%key: tensor<2xui64>, %arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>)"
        ),
        "mu/cov must become vector/matrix func args after %key, in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<3x2xf32>, tensor<2xui64>)"),
        "fanned MvNormal must return the [3, 2] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.cholesky").count(),
        1,
        "one shared cholesky on the [2, 2] cov, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [3, 2] batch, in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<2xui64>, tensor<3x2xui32>)"),
        "the single rng_bit_generator must be sized to the [3, 2] batch (a genuine tensor<3x2> draw, not a [2] broadcast), in:\n{out}"
    );
    assert!(
        out.contains("contracting_dims = [1] x [1]"),
        "the row-wise L·z affine must be a batched dot_general (z · Lᵀ), in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.broadcast_in_dim %arg0, dims = [1]"),
        "mu must broadcast across the n rows, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_mvnormal_iid_matches_frozen_golden() {
    let d = determinize_src(MVNORMAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/mvnormal_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted batched-MvNormal @sample drifted from tests/goldens/mvnormal_iid_sample.mlir"
    );
}

/// A fanned Dirichlet draw: `iid(Dirichlet(alpha = [2, 3, 4]), 5)` — the OTHER
/// vector-variate sampler (cf. MvNormal). Its per-component `g_j ~ Gamma(α_j,
/// 1)` unroll fans out with NO rank-3 machinery: under the `[5]` fan-out shape
/// each component's `draw_gamma` dispatches to the batched Marsaglia–Tsang
/// `draw_gamma_batched` (an `[5]` column, one masked `while` per component),
/// the `d = 3` columns stack on axis 0 → `[3, 5]`, ONE new `stablehlo.transpose`
/// reorients to `[5, 3]` (rows = draws), then each row is normalized by its
/// row-sum (`reduce_sum_last_axis` over the component axis + broadcast +
/// divide) — `[5, 3]`, each row a simplex, the 5 rows independent. The frozen
/// golden below is parse-validated against the real StableHLO parser (jax
/// 0.10.2). The scalar Dirichlet path (`emit_sample_dirichlet_*`) is
/// byte-identical to before.
const DIRICHLET_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Dirichlet(alpha = [2.0, 3.0, 4.0]), 5)
draws = rand(s, lawof(xs))
";

/// The fanned Dirichlet draw returns a `tensor<5x3xf32>` (`[m, d]`, not a bare
/// `[d]` simplex) alongside the advanced key, via one masked `stablehlo.while`
/// PER component (three, each carrying a `tensor<5xi1>` per-lane accept mask),
/// the `[3, 5]` column stack reoriented by a `stablehlo.transpose dims = [1, 0]`
/// to `[5, 3]`, then the per-row normalize (last-axis reduce + `dims = [0]`
/// row-broadcast + divide).
#[test]
fn emit_sample_dirichlet_iid_has_expected_structure() {
    let d = determinize_src(DIRICHLET_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing @sample(%key) (literal alpha → no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<5x3xf32>, tensor<2xui64>)"),
        "fanned Dirichlet must return the [5, 3] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.while").count(),
        3,
        "one masked batched rejection loop per Dirichlet component, in:\n{out}"
    );
    assert!(
        out.contains("tensor<5xi1>"),
        "each component's per-lane accept mask must be tensor<5xi1>, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.transpose") && out.contains("dims = [1, 0]"),
        "the axis-0 [3, 5] column stack must be transposed to the [5, 3] batch, in:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.add across dimensions = [1] : (tensor<5x3xf32>"),
        "each row's Gamma sum reduces the INNER component axis (the m rows survive), in:\n{out}"
    );
    assert!(
        out.contains("dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>"),
        "the row-sum [5] must broadcast back across the component axis → [5, 3], in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_dirichlet_iid_matches_frozen_golden() {
    let d = determinize_src(DIRICHLET_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/dirichlet_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned Dirichlet @sample drifted from tests/goldens/dirichlet_iid_sample.mlir"
    );
}

// ---- fan-out for discrete inverse-CDF Poisson + NegativeBinomial (buffy #148,
// #156b) ---------------------------------------------------------------------
//
// Poisson's scalar draw is a bounded inverse-CDF `stablehlo.while` (walk the CDF
// until `U <= F(k)`). The fanned draw batches that walk PER LANE via
// `draw_poisson_batched`: one `U` drawn as a genuine `[m]` tensor, `[m]` per-lane
// `cum`/`pmf`/`done`/`result`, ONE scalar counter `k` walked in lockstep, and a
// per-lane latch (the `draw_gamma_batched` masked-while + `reduce_all(done)`
// pattern). The two NegativeBinomial Gamma–Poisson mixtures fan out with NO
// extra machinery — `draw_gamma_batched` yields the `[m]` per-lane `lambda`,
// which `draw_poisson_batched` accepts as its `[m]` rate (so a NegBin fan-out is
// the batched Gamma `while` feeding the batched Poisson `while`). All three
// goldens are parse-validated against the real StableHLO parser (jax 0.10.2).
// The one `while` discrete sampler still refusing is Multinomial.

/// A fanned Poisson draw: `iid(Poisson(rate=3), 4)`, the discrete inverse-CDF
/// batched-walk case.
const POISSON_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Poisson(rate = 3.0), 4)
draws = rand(s, lawof(xs))
";

/// A fanned NegativeBinomial draw: `iid(NegativeBinomial(alpha=5, beta=2), 4)`,
/// the Gamma–Poisson mixture — batched Gamma `while` feeding batched Poisson
/// `while`.
const NEGATIVE_BINOMIAL_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(NegativeBinomial(alpha = 5.0, beta = 2.0), 4)
draws = rand(s, lawof(xs))
";

/// A fanned NegativeBinomial2 draw: `iid(NegativeBinomial2(mu=3, psi=5), 4)`,
/// the mean-dispersion mixture (same shape as the `NegativeBinomial` case, with
/// the `rate = psi/mu` reparameterization).
const NEGATIVE_BINOMIAL2_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(NegativeBinomial2(mu = 3.0, psi = 5.0), 4)
draws = rand(s, lawof(xs))
";

/// The fanned Poisson draw returns a `tensor<4xf32>` (not a scalar) alongside
/// the advanced key, drawn from EXACTLY ONE `rng_bit_generator` sized to the
/// `[4]` batch (a genuine `tensor<4xui32>` output — one advance, one advanced
/// key returned — not a scalar broadcast), via ONE masked `stablehlo.while`
/// carrying a `tensor<4xi1>` per-lane `done` mask, with `reduce_all(done)` in
/// the loop condition and an `or` carry.
#[test]
fn emit_sample_poisson_iid_has_expected_structure() {
    let d = determinize_src(POISSON_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("func.func @sample(%key: tensor<2xui64>)"),
        "missing @sample(%key) (no free params) in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Poisson must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance (one advanced key) for the whole [4] batch, in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<2xui64>, tensor<4xui32>)"),
        "the single rng_bit_generator must be sized to the [4] batch (genuine tensor<4> draw), in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.while").count(),
        1,
        "one masked while for the whole [4] batch, in:\n{out}"
    );
    assert!(
        out.contains("tensor<4xi1>"),
        "per-lane done mask must be tensor<4xi1>, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.reduce(") && out.contains("stablehlo.or "),
        "the batched walk needs an all-reduce over the done mask + an OR carry, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_poisson_iid_matches_frozen_golden() {
    let d = determinize_src(POISSON_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/poisson_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/poisson_iid_sample.mlir"
    );
}

/// NegativeBinomial fans out to the batched Gamma `while` (its `[4]` per-lane
/// `lambda`) feeding the batched Poisson `while` (that `[4]` rate): TWO
/// `stablehlo.while`s, one `[4]` value return, one advanced key.
#[test]
fn emit_sample_negative_binomial_iid_has_expected_structure() {
    let d = determinize_src(NEGATIVE_BINOMIAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned NegativeBinomial must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.while").count(),
        2,
        "the batched Gamma while feeding the batched Poisson while, in:\n{out}"
    );
    assert!(
        out.contains("tensor<4xi1>"),
        "both batched loops carry a tensor<4xi1> per-lane mask, in:\n{out}"
    );

    let golden = include_str!("goldens/negative_binomial_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/negative_binomial_iid_sample.mlir"
    );
}

/// The `rate = psi/mu` mirror of the NegativeBinomial fan-out: same
/// batched-Gamma-then-batched-Poisson shape, frozen against its golden.
#[test]
fn emit_sample_negative_binomial2_iid_matches_frozen_golden() {
    let d = determinize_src(NEGATIVE_BINOMIAL2_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    assert_eq!(
        out.matches("stablehlo.while").count(),
        2,
        "the batched Gamma while feeding the batched Poisson while, in:\n{out}"
    );
    let golden = include_str!("goldens/negative_binomial2_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/negative_binomial2_iid_sample.mlir"
    );
}

// ---- fan-out for elementwise Laplace/Bernoulli/Geometric (buffy #148, #155,
// #156a) --------------------------------------------------------------------
//
// Bernoulli/Geometric were left off `FANOUT_SAFE` alongside the genuinely
// `while`/unrolled discrete samplers, and Laplace was left off scoped to Task
// 10a's rejection-family change (see the doc comment above `FANOUT_SAFE`), but
// all three are straight-line elementwise: Laplace's `sgn` and Bernoulli's
// indicator both compose from the auto-broadcasting `compare`/`select` pair
// Task 10a already gave `Emitter`, and Geometric's `floor` is shape-preserving
// like every other Tier-1 unary. No new primitive needed — this batch just
// admits them to `FANOUT_SAFE`. Both goldens are parse-validated against the
// real StableHLO parser (jax 0.10.2). The genuinely `while` discrete samplers
// (Poisson + the NegativeBinomial mixtures) are batched in the Tier-4 section
// below; only Multinomial still refuses.

/// A fanned Laplace draw: `iid(Laplace(location=0, scale=1), 4)`, exercising
/// the `compare`/`select` `sgn` idiom under a `[4]` batch.
const LAPLACE_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Laplace(location = 0.0, scale = 1.0), 4)
draws = rand(s, lawof(xs))
";

/// A fanned Bernoulli draw: `iid(Bernoulli(p=0.3), 4)`, exercising the
/// `select(U < p, 1, 0)` idiom under a `[4]` batch.
const BERNOULLI_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Bernoulli(p = 0.3), 4)
draws = rand(s, lawof(xs))
";

/// A fanned Geometric draw: `iid(Geometric(p=0.3), 4)`, exercising `floor`
/// under a `[4]` batch.
const GEOMETRIC_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Geometric(p = 0.3), 4)
draws = rand(s, lawof(xs))
";

/// The fanned Laplace draw returns a `tensor<4xf32>` (not a scalar) alongside
/// the advanced key, with exactly ONE `rng_bit_generator` sized to `[4]`, one
/// `compare`/`select` pair for `sgn`, and the scalar params broadcast over the
/// batch.
#[test]
fn emit_sample_laplace_iid_has_expected_structure() {
    let d = determinize_src(LAPLACE_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Laplace must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.compare").count(),
        1,
        "sgn(U - 1/2) needs exactly one compare, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.select").count(),
        1,
        "sgn(U - 1/2) needs exactly one select, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.broadcast_in_dim"),
        "scalar params must broadcast over the batch, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_laplace_iid_matches_frozen_golden() {
    let d = determinize_src(LAPLACE_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/laplace_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/laplace_iid_sample.mlir"
    );
}

/// The fanned Bernoulli draw returns a `tensor<4xf32>` alongside the advanced
/// key, with exactly ONE `rng_bit_generator` sized to `[4]`, one `compare`,
/// one `select`.
#[test]
fn emit_sample_bernoulli_iid_has_expected_structure() {
    let d = determinize_src(BERNOULLI_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Bernoulli must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.compare").count(), 1);
    assert_eq!(out.matches("stablehlo.select").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_bernoulli_iid_matches_frozen_golden() {
    let d = determinize_src(BERNOULLI_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/bernoulli_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/bernoulli_iid_sample.mlir"
    );
}

/// The fanned Geometric draw returns a `tensor<4xf32>` alongside the advanced
/// key, with exactly ONE `rng_bit_generator` sized to `[4]`, two `log`s, one
/// `floor`.
#[test]
fn emit_sample_geometric_iid_has_expected_structure() {
    let d = determinize_src(GEOMETRIC_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Geometric must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.log").count(), 2);
    assert_eq!(out.matches("stablehlo.floor").count(), 1);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_geometric_iid_matches_frozen_golden() {
    let d = determinize_src(GEOMETRIC_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/geometric_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/geometric_iid_sample.mlir"
    );
}

// ---- fan-out for discrete non-elementwise Binomial + Categorical (buffy #148,
// #156b) ---------------------------------------------------------------------
//
// Unlike the Tier-1 elementwise discrete samplers (Bernoulli/Geometric), these
// two have a scalar draw that is NOT purely elementwise, yet each fans out
// without a `while`:
//   - Binomial's scalar draw already owns an inner axis (its `n` Bernoulli
//     trials, `reduce_sum`ed to one count). The fanned draw is a rank-2 `[m, n]`
//     uniform — a genuine `rng_bit_generator` output, ONE advance, rows
//     independent — reduced over the INNER count axis (last) to `[m]` by the new
//     `Emitter::reduce_sum_last_axis` (the outer `m` lanes survive, unlike the
//     scalar path's full `reduce_sum`).
//   - Categorical/Categorical0 fan out with NO new primitive: one scalar `U`
//     becomes a `[m]` draw under the batch shape, and the inverse-CDF unroll's
//     running count is promoted to `[m]` by the auto-broadcasting
//     `compare`/`select`/`add`.
// Both goldens are parse-validated against the real StableHLO parser (jax
// 0.10.2). Poisson + the NegativeBinomial mixtures are batched in the Tier-4
// section below.

/// A fanned Binomial draw: `iid(Binomial(n=5, p=0.3), 4)`, `n` the FIXED inner
/// count, `4` the outer fan-out — the `[4, 5]` uniform, inner-axis reduce case.
const BINOMIAL_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
n = 5
xs ~ iid(Binomial(n = n, p = 0.3), 4)
draws = rand(s, lawof(xs))
";

/// A fanned Categorical draw: `iid(Categorical(p=[0.2,0.3,0.5]), 4)`, exercising
/// the inverse-CDF unroll broadcasting its running count over a `[4]` batch.
const CATEGORICAL_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Categorical(p = [0.2, 0.3, 0.5]), 4)
draws = rand(s, lawof(xs))
";

/// A fanned Categorical0 draw (0-based): same as the Categorical case but with
/// `base = 0.0` — the only difference is the initial count constant.
const CATEGORICAL0_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Categorical0(p = [0.2, 0.3, 0.5]), 4)
draws = rand(s, lawof(xs))
";

/// A fanned SINGLE-CATEGORY Categorical draw: `iid(Categorical(p=[1.0]), 4)`.
/// `n = 1` runs the inverse-CDF unroll zero times (`n - 1 = 0` iterations), so
/// `count` never leaves the scalar `base` — this exercises `draw_categorical`'s
/// degenerate-guard broadcast (`registry.rs`), which must still lift it to the
/// `[4]` batch: a fanned draw is `[m]`-shaped like every other count, even for
/// this constant one-category variate.
const CATEGORICAL_SINGLE_IID_SAMPLE_SRC: &str = "\
s = rnginit(0)
xs ~ iid(Categorical(p = [1.0]), 4)
draws = rand(s, lawof(xs))
";

/// The fanned Binomial draw returns a `tensor<4xf32>` (not a scalar) alongside
/// the advanced key, drawn as a GENUINE rank-2 `[4, 5]` batch (one
/// `rng_bit_generator` sized to `tensor<4x5xui32>` — 4 independent variates,
/// each 5 Bernoulli trials — not a broadcast of a `[5]` draw), then reduced over
/// the inner count axis by EXACTLY ONE `stablehlo.reduce` over
/// `dimensions = [1]` → `[4]`.
#[test]
fn emit_sample_binomial_iid_has_expected_structure() {
    let d = determinize_src(BINOMIAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Binomial must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4, 5] batch, in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<2xui64>, tensor<4x5xui32>)"),
        "the draw must be a genuine rank-2 [4, 5] rng output (rows independent), in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.reduce(").count(),
        1,
        "exactly one reduce (the inner-count-axis sum), in:\n{out}"
    );
    assert!(
        out.contains("across dimensions = [1] : (tensor<4x5xf32>, tensor<f32>) -> tensor<4xf32>"),
        "the reduce must be over the LAST axis only ([4, 5] -> [4]), in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_binomial_iid_matches_frozen_golden() {
    let d = determinize_src(BINOMIAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/binomial_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/binomial_iid_sample.mlir"
    );
}

/// The fanned Categorical draw returns a `tensor<4xf32>` alongside the advanced
/// key, with exactly ONE `rng_bit_generator` sized to `[4]` and NO new
/// primitive — the running count broadcasts over the batch via the length-3
/// `p`'s `n - 1 = 2` compare/select prefix-sum comparisons.
#[test]
fn emit_sample_categorical_iid_has_expected_structure() {
    let d = determinize_src(CATEGORICAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Categorical must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert!(
        out.contains("-> (tensor<2xui64>, tensor<4xui32>)"),
        "the single rng_bit_generator must be sized to the [4] batch, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.reduce(").count(),
        0,
        "Categorical fans out via broadcast, not a reduce, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.compare").count(), 2);
    assert_eq!(out.matches("stablehlo.select").count(), 2);
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_categorical_iid_matches_frozen_golden() {
    let d = determinize_src(CATEGORICAL_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/categorical_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/categorical_iid_sample.mlir"
    );
}

/// The `base = 0.0` mirror of the Categorical fan-out: same `[4]` batch, same op
/// counts, differing only in the initial count constant (checked structurally —
/// the difference is exercised by the 1-based golden above).
#[test]
fn emit_sample_categorical0_iid_has_expected_structure() {
    let d = determinize_src(CATEGORICAL0_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned Categorical0 must return the [4] batch + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert_eq!(out.matches("stablehlo.reduce(").count(), 0);
    assert_eq!(out.matches("stablehlo.compare").count(), 2);
    assert_eq!(out.matches("stablehlo.select").count(), 2);
    assert!(is_delimiter_balanced(&out));
}

/// The degenerate `n = 1` case: the inverse-CDF unroll runs ZERO iterations
/// (`n - 1 = 0`), so `count` never leaves the scalar `base` — this is the one
/// path `draw_categorical`'s fan-out relies on an explicit
/// `broadcast_in_dim` guard for (see `registry.rs`) rather than the unroll's
/// auto-broadcasting compare/select/add. Must still return the `[4]` batch,
/// NOT a scalar `tensor<f32>` — a fanned draw is `[m]`-shaped like every
/// other count, even for this constant one-category variate.
#[test]
fn emit_sample_categorical_single_category_iid_has_expected_structure() {
    let d = determinize_src(CATEGORICAL_SINGLE_IID_SAMPLE_SRC);
    let out = emit_sample(&d);

    assert!(
        out.contains("-> (tensor<4xf32>, tensor<2xui64>)"),
        "fanned single-category Categorical must return the [4] batch (not a \
         scalar) + advanced key, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.rng_bit_generator").count(),
        1,
        "one rng_bit_generator advance for the whole [4] batch, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.compare").count(),
        0,
        "n - 1 = 0 comparisons: the unroll never runs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.select").count(),
        0,
        "n - 1 = 0 selects: the unroll never runs, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.broadcast_in_dim").count(),
        1,
        "the degenerate guard's explicit broadcast of the scalar base to [4], \
         in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text (parse-validated against the real StableHLO
/// parser): any drift must be a deliberate, reviewed change to this golden.
#[test]
fn emit_sample_categorical_single_category_iid_matches_frozen_golden() {
    let d = determinize_src(CATEGORICAL_SINGLE_IID_SAMPLE_SRC);
    let out = emit_sample(&d);
    let golden = include_str!("goldens/categorical_single_iid_sample.mlir");
    assert_eq!(
        out, golden,
        "emitted fanned @sample drifted from tests/goldens/categorical_single_iid_sample.mlir"
    );
}

// ---- PR-1: the `inputs`/`outputs` compilation ABI --------------------------
//
// `inputs = (a, b)` / `outputs = (q1, q2)` are reserved top-level bindings
// (design doc `docs/superpowers/specs/2026-07-17-inputs-outputs-abi-design.md`)
// that survive determinization (they are the DCE roots) and give the emitted
// `func.func` an explicit, ordered arg/result list — superseding
// `is_free_param`'s source-order convention and `select_query`'s
// last-public-binding convention. PR-1 scope: `elementof` inputs only,
// `Mode::LogDensity` only. PR-2 extends the arg path to the fixed-phase input
// constructs `external`/`load_data` (see the "PR-2" section below); a binding
// named in `inputs` that is neither `elementof` nor a fixed-phase input
// construct (a literal, a computed value) still refuses — see
// `emit_logdensity_abi_refuses_non_elementof_input`.

/// Parse + infer `src`, then `determinize_with_roots` rooted on `roots` (the
/// `inputs`/`outputs` binding names) — mirrors
/// `crates/determinizer/tests/canon_dce_golden.rs`'s `determinize_roots`
/// helper, needed here (rather than the crate's plain `determinize_src`,
/// which always passes `roots: None`) so DCE actually prunes bindings
/// unreached from the ABI roots.
fn determinize_abi_roots(src: &str, roots: &[&str]) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    let syms: Vec<flatppl_core::Symbol> = roots.iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse")
}

const ABI_TWO_OUTPUT_SRC: &str = "\
a = elementof(reals)
b = elementof(reals)
dead_helper = a * 2.0
m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0)), b = draw(Normal(mu = 0.0, sigma = 1.0))))
q1 = logdensityof(m, record(a = a, b = b))
q2 = logdensityof(m, record(a = a, b = b))
inputs = (a, b)
outputs = (q1, q2)
";

/// The brief's Step-4 golden: a determinized module declaring `inputs = (a, b)`
/// / `outputs = (q1, q2)` emits a `func.func` whose args are `a`, `b` (in
/// `inputs` order) and whose results are `q1`, `q2` (in `outputs` order,
/// multi-result) — and `dead_helper` (unreachable from either root) is pruned
/// by determinization, not merely ignored by the emitter.
#[test]
fn emit_logdensity_abi_ordered_args_and_multi_output() {
    let d = determinize_abi_roots(ABI_TWO_OUTPUT_SRC, &["inputs", "outputs"]);
    let pir = flatppl_flatpir::write(&d);
    assert!(
        !pir.contains("dead_helper"),
        "dead_helper must be pruned by root-DCE:\n{pir}"
    );

    let out = emit_logdensity(&d);
    assert!(
        out.contains("func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>)"),
        "expected a 2-arg/2-result signature in `inputs`/`outputs` order:\n{out}"
    );
    assert_eq!(
        out.matches("return").count(),
        1,
        "exactly one multi-value return, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// A declared `inputs` entry that no `outputs` query reaches is still kept as
/// a stable ABI arg (design doc: "a declared input that no output reaches is
/// still kept as an argument — a stable ABI is not DCE'd") — `c` here is
/// unused by `q1` yet survives determinization and becomes `%arg2`.
#[test]
fn emit_logdensity_abi_unused_declared_input_survives_as_stable_arg() {
    let src = "\
a = elementof(reals)
b = elementof(reals)
c = elementof(reals)
m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0)), b = draw(Normal(mu = 0.0, sigma = 1.0))))
q1 = logdensityof(m, record(a = a, b = b))
inputs = (a, b, c)
outputs = q1
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let out = emit_logdensity(&d);
    assert!(
        out.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>"
        ),
        "unused declared input `c` must still become a stable %arg2, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Exhaustiveness (design doc: `inputs` is "authoritative and exhaustive" —
/// every `elementof` parameter must be listed): `b` is reachable from `q1`
/// (root-DCE keeps it — the query needs it) but is NOT listed in `inputs`,
/// which declares only `a`. `emit_logdensity_abi` must refuse naming `b`
/// rather than silently emitting a 1-arg function that leaves `b` unbound.
#[test]
fn emit_logdensity_abi_refuses_non_exhaustive_inputs() {
    let src = "\
a = elementof(reals)
b = elementof(reals)
m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0)), b = draw(Normal(mu = 0.0, sigma = 1.0))))
q1 = logdensityof(m, record(a = a, b = b))
inputs = a
outputs = q1
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("not listed in `inputs`") && err.msg.contains('b'),
        "expected an exhaustiveness refusal naming `b`, got: {}",
        err.msg
    );
}

/// A binding named in `inputs` that is neither an `elementof` parameter nor a
/// fixed-phase input construct (`external`/`load_data`) must REFUSE rather than
/// emit a partial signature: here a literal array `y = [1.0, 2.0]` (not a data
/// construct — its values ARE known, but it is not an ABI-argument construct).
/// `mu` is a listed `elementof` so the model determinizes as a standard
/// parameterized density (isolating the non-input-construct refusal from the
/// exhaustiveness check, which passes here). Fixed-phase `external`/`load_data`
/// inputs are now accepted (PR-2) — see the PR-2 section.
#[test]
fn emit_logdensity_abi_refuses_non_elementof_input() {
    let src = "\
mu = elementof(reals)
y = [1.0, 2.0]
m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0))))
q1 = logdensityof(m, record(a = mu))
inputs = (mu, y)
outputs = q1
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("not an elementof parameter") && err.msg.contains('y'),
        "expected a non-elementof-input refusal naming `y`, got: {}",
        err.msg
    );
}

/// Fallback (design doc "Fallback + migration"): a model declaring NEITHER
/// `inputs` nor `outputs` still emits via the legacy last-public-binding
/// path, byte-for-byte the same as before this PR (every other golden test
/// in this file exercises the same fallback; this test exists to name it
/// explicitly as the ABI's negative case). The CLI-level deprecation warning
/// is a `stablehlo_cmd` concern, not this crate's `emit` — see
/// `crates/cli/tests/stablehlo.rs`.
#[test]
fn emit_logdensity_legacy_path_unaffected_when_abi_absent() {
    let d = determinize_src(NORMAL_DENSITY_SRC);
    let out = emit_logdensity(&d);
    assert!(
        out.contains("func.func @logdensity"),
        "legacy last-public-binding path must still emit, in:\n{out}"
    );
}

/// Rust-side ABI tolerance (brief step 6(c), full JS tolerance is PR-3): a
/// model carrying `inputs`/`outputs` parses and infers without diagnostics —
/// the reserved names are ordinary top-level bindings to the rest of the
/// Rust toolchain (they only carry ABI meaning to the StableHLO emitter).
#[test]
fn inputs_outputs_bindings_parse_and_infer_without_error() {
    let mut m = flatppl_syntax::parse(ABI_TWO_OUTPUT_SRC).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
}

// ---- PR-2: fixed-phase inputs (`external`, `load_data`) as runtime args -----
//
// A fixed-phase binding (`external(S)` / `load_data(...)`) listed in `inputs`
// becomes a `func.func` argument instead of refusing (PR-1) — extending the
// SAME ABI arg path PR-1 built for `elementof`. `external(S)` types from `S`
// (scalar first); `load_data(...)` types `tensor<N×f32>` with `N` pinned from
// the compile-time file read threaded via `EmitOptions::input_shapes` (design
// doc "load_data — shape, not values": only the shape is pinned, the values are
// the runtime argument, never baked). A fixed-phase binding NOT in `inputs`
// still refuses, pointing at the ABI.

/// Emit `@logdensity` with a compile-time shape-pin map (the CLI's
/// [`EmitOptions::input_shapes`], populated from a `load_data` file read).
fn emit_logdensity_with_shapes(m: &Module, shapes: &[(&str, Vec<u64>)]) -> String {
    let opts = flatppl_stablehlo::EmitOptions {
        input_shapes: shapes
            .iter()
            .map(|(name, dims)| (name.to_string(), dims.clone()))
            .collect(),
        ..Default::default()
    };
    flatppl_stablehlo::emit(m, flatppl_stablehlo::Mode::LogDensity, &opts)
        .expect("must emit @logdensity")
}

/// Step 2 (external, scalar): a scalar `external(reals)` listed in `inputs`
/// becomes a scalar `func.func` argument (`tensor<f32>`), NOT a PR-1 refusal.
/// `mu_ext` is used as the Normal's mean so the output reaches it; `a` is the
/// elementof variate. Args are in `inputs` order: `a` → `%arg0`,
/// `mu_ext` → `%arg1`.
#[test]
fn emit_logdensity_abi_external_scalar_becomes_arg() {
    let src = "\
a = elementof(reals)
mu_ext = external(reals)
m = lawof(record(a = draw(Normal(mu = mu_ext, sigma = 1.0))))
q1 = logdensityof(m, record(a = a))
inputs = (a, mu_ext)
outputs = q1
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let out = emit_logdensity(&d);
    assert!(
        out.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>"
        ),
        "scalar external `mu_ext` must become `%arg1: tensor<f32>`, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Step 3 (load_data): a `load_data(...)` binding listed in `inputs` emits a
/// ranked-tensor argument whose length is the compile-time-pinned `N`
/// (`tensor<3×f32>` here) — NOT a `tensor<?×f32>` (unusable downstream), NOT a
/// refusal, and NOT a baked `stablehlo.constant` (its values are the runtime
/// argument). `y` is a declared-but-unused input (a stable ABI arg is not
/// DCE'd), isolating the shape-pin from any density-over-vector lowering.
#[test]
fn emit_logdensity_abi_load_data_pinned_tensor_arg() {
    let src = "\
a = elementof(reals)
y = load_data(\"data.csv\", reals)
m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0))))
q1 = logdensityof(m, record(a = a))
inputs = (a, y)
outputs = q1
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let out = emit_logdensity_with_shapes(&d, &[("y", vec![3])]);
    assert!(
        out.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> tensor<f32>"
        ),
        "load_data `y` must become a shape-pinned `%arg1: tensor<3xf32>`, in:\n{out}"
    );
    assert!(
        !out.contains("tensor<?x"),
        "the pinned load_data arg must not carry a dynamic `?` dim, in:\n{out}"
    );
    // Values are the runtime argument, never inlined: no data constant for `y`.
    assert!(
        !out.contains("dense<[") && !out.contains("data.csv"),
        "load_data values must not be baked as a constant, in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Step 4 (refuse when not in `inputs`): a fixed-phase binding that an output
/// reaches (so root-DCE keeps it) but that is NOT listed in `inputs` refuses,
/// with a message pointing at the ABI (`list it in inputs …`) — fixed data
/// becomes a runtime argument only by being declared, never baked. `mu_ext`
/// (a scalar `external`) feeds the Normal's mean but is absent from
/// `inputs = a`.
#[test]
fn emit_logdensity_abi_refuses_fixed_input_not_in_inputs() {
    let src = "\
a = elementof(reals)
mu_ext = external(reals)
m = lawof(record(a = draw(Normal(mu = mu_ext, sigma = 1.0))))
q1 = logdensityof(m, record(a = a))
inputs = a
outputs = q1
";
    let d = determinize_abi_roots(src, &["inputs", "outputs"]);
    let err = flatppl_stablehlo::emit(
        &d,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.msg.contains("list it in inputs") && err.msg.contains("mu_ext"),
        "expected an ABI-pointing refusal naming `mu_ext`, got: {}",
        err.msg
    );
}

// ---- Buffy #327: `builtin_touniform` (univariate continuous CDF) ------------
//
// Spec §07 "Measure kernel evaluation primitives": for kernels of univariate
// continuous measures, `builtin_touniform(kernel, kernel_input, x)` is the
// cumulative distribution function `F`. The determiniser emits it as the
// truncation normaliser `F(hi) - F(lo)` — e.g. a `normalize(truncate(Normal /
// Cauchy, interval(0, inf)))` prior — so these anchor fixtures reach the CDF
// builders through that route rather than a bare `builtin_touniform` call.
// Only Normal (`chlo.erf`) and Cauchy (`stablehlo.atan2`) carry a CDF builder;
// every other distribution refuses (refuse-don't-mislower). Numeric end-to-end
// correctness against the scipy oracle lives in `flatppl-testsuite`'s
// stablehlo/examples gates (ex_eight_schools uses the Cauchy CDF).

const NORMAL_TRUNC_TOUNIFORM_SRC: &str = "\
mu = elementof(reals)
sigma = elementof(posreals)
a = draw(normalize(truncate(Normal(mu = mu, sigma = sigma), interval(0, inf))))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

const CAUCHY_TRUNC_TOUNIFORM_SRC: &str = "\
location = elementof(reals)
scale = elementof(posreals)
a = draw(normalize(truncate(Cauchy(location = location, scale = scale), interval(0, inf))))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))
";

/// The Normal CDF `F(x) = ½·(1 + erf((x − μ)/(σ·√2)))` reaches the emitter via
/// the truncation normaliser: `chlo.erf` must appear (exactly twice — `F(inf)`
/// and `F(0)`), and no `stablehlo.atan2` (that is Cauchy's CDF, not Normal's).
#[test]
fn emit_touniform_normal_via_truncation_has_expected_structure() {
    let d = determinize_src(NORMAL_TRUNC_TOUNIFORM_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );
    let out = emit_logdensity(&d);
    assert_eq!(
        out.matches("chlo.erf ").count(),
        2,
        "Normal CDF must emit two chlo.erf (F(hi), F(lo)), in:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.atan2"),
        "Normal CDF must not use atan2 (that is Cauchy's CDF), in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// The Cauchy CDF `F(x) = ½ + (1/π)·atan((x − x₀)/γ)` reaches the emitter via
/// the truncation normaliser: `stablehlo.atan2` must appear (exactly twice —
/// `F(inf)` and `F(0)`), and no `chlo.erf` (that is Normal's CDF).
#[test]
fn emit_touniform_cauchy_via_truncation_has_expected_structure() {
    let d = determinize_src(CAUCHY_TRUNC_TOUNIFORM_SRC);
    assert!(
        flatppl_determinizer::is_flatpdl(&d).is_ok(),
        "determinized module must be FlatPDL-conformant (no residual measure node)"
    );
    let out = emit_logdensity(&d);
    assert_eq!(
        out.matches("stablehlo.atan2").count(),
        2,
        "Cauchy CDF must emit two stablehlo.atan2 (F(hi), F(lo)), in:\n{out}"
    );
    assert!(
        !out.contains("chlo.erf "),
        "Cauchy CDF must not use chlo.erf (that is Normal's CDF), in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Freeze the exact emitted text for the Normal-CDF truncation path.
#[test]
fn emit_touniform_normal_matches_frozen_golden() {
    let d = determinize_src(NORMAL_TRUNC_TOUNIFORM_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/normal_touniform_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from goldens/normal_touniform_logdensity.mlir"
    );
}

/// Freeze the exact emitted text for the Cauchy-CDF truncation path.
#[test]
fn emit_touniform_cauchy_matches_frozen_golden() {
    let d = determinize_src(CAUCHY_TRUNC_TOUNIFORM_SRC);
    let out = emit_logdensity(&d);
    let golden = include_str!("goldens/cauchy_touniform_logdensity.mlir");
    assert_eq!(
        out, golden,
        "emitted @logdensity drifted from goldens/cauchy_touniform_logdensity.mlir"
    );
}

/// `builtin_touniform(Beta, ...)` — a registered distribution with NO CDF
/// builder (spec §07: transport is defined only for continuous kernels whose
/// canonical transport is specified; Beta's is not rendered here) — must
/// refuse precisely, not guess a lowering.
#[test]
fn builtin_touniform_refuses_dist_without_cdf() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Beta");
    let alpha = real(&mut m, 2.0);
    let beta = real(&mut m, 2.0);
    let kernel_input = record_node(&mut m, &[("alpha", alpha), ("beta", beta)]);
    let x = real(&mut m, 0.5);
    let node = call(&mut m, "builtin_touniform", &[ctor, kernel_input, x]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg
            .contains("builtin_touniform (CDF) not defined for distribution 'Beta'"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(node));
}

/// `builtin_touniform(Bogus, ...)` — an unregistered constructor — refuses via
/// the same `lookup` gate as `builtin_logdensityof`, before the CDF check.
#[test]
fn builtin_touniform_refuses_unregistered_ctor() {
    let mut m = Module::new();
    let ctor = const_node(&mut m, "Bogus");
    let field_val = real(&mut m, 0.0);
    let kernel_input = record_node(&mut m, &[("x0", field_val)]);
    let x = real(&mut m, 1.0);
    let node = call(&mut m, "builtin_touniform", &[ctor, kernel_input, x]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("no lowering for distribution 'Bogus'"),
        "unexpected message: {}",
        err.msg
    );
}

/// `builtin_touniform`'s kernel must be a bare `Const` distribution
/// constructor — a `Ref` in that position refuses rather than being silently
/// mis-resolved (mirrors `builtin_logdensityof`'s identical gate).
#[test]
fn builtin_touniform_refuses_non_const_kernel() {
    let mut m = Module::new();
    let kernel = local_ref(&mut m, "k");
    let kernel_input = call(&mut m, "record", &[]);
    let x = real(&mut m, 1.0);
    let node = call(&mut m, "builtin_touniform", &[kernel, kernel_input, x]);

    let mut e = Emitter::new(&m, Dtype::F32);
    let err = e.lower_node(node).unwrap_err();
    assert!(
        err.msg.contains("bare distribution constructor"),
        "unexpected message: {}",
        err.msg
    );
    assert_eq!(err.node, Some(kernel));
}
