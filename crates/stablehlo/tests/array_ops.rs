//! §07 array and index operations as emitter heads: `onehot`, `reverse`, `tile`,
//! `splitblocks`, `joinblocks`, `partition`, `blockdiagmat`, `bandedmat`,
//! `bincounts` and `selectbins`, and the `filter` refusal. Structural only: the
//! values and gradients are checked by executing the modules under Enzyme-JAX
//! against numpy in the PR.

use flatppl_stablehlo::{Mode, emit};

const PRE: &str = "v = [1.0, z0, 3.0]\nw = [4.0, 5.0, z0]\n\
    A = rowstack([[1.0, z0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])\nB = rowstack([[9.0]])\n";

fn emitted(expr: &str) -> Result<String, String> {
    let src = format!(
        "flatppl_compat = \"0.1\"\nz0 = elementof(reals)\n{PRE}x = {expr}\ninputs = (z0)\noutputs = (x)"
    );
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "inference errors: {diags:?}"
    );
    let syms: Vec<flatppl_core::Symbol> =
        ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    let d = flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must lower, not refuse");
    emit(&d, Mode::LogDensity, &Default::default()).map_err(|e| e.msg)
}

fn returns(expr: &str, ty: &str) {
    let out = emitted(expr).unwrap_or_else(|e| panic!("{expr} must emit: {e}"));
    assert!(
        out.contains(&format!("-> {ty}")),
        "{expr} returns {ty}:\n{out}"
    );
}

fn refuses(expr: &str, needle: &str) {
    let e = emitted(expr).expect_err("must refuse");
    assert!(e.contains(needle), "{expr}: {e}");
}

#[test]
fn shapes_follow_the_written_sizes() {
    returns("onehot(2, 3)", "tensor<3xf32>");
    returns("reverse(v)", "tensor<3xf32>");
    returns("reverse([v, w])", "tensor<2x3xf32>");
    returns("tile(v, 3)", "tensor<9xf32>");
    returns("tile(A, [2, 3])", "tensor<4x12xf32>");
    returns("splitblocks([1.0, 2.0, 3.0, z0], 2)", "tensor<2x2xf32>");
    returns("splitblocks(A, [1, 2])", "tensor<2x2x1x2xf32>");
    returns("joinblocks([v, w])", "tensor<6xf32>");
    returns("joinblocks(splitblocks(A, [1, 2]))", "tensor<2x4xf32>");
    returns(
        "partition([1.0, 2.0, 3.0, 4.0, 5.0, z0], 3)",
        "tensor<2x3xf32>",
    );
    returns(
        "partition([1.0, 2.0, 3.0, 4.0, 5.0, z0], [2, 2, 2])",
        "tensor<3x2xf32>",
    );
    returns("blockdiagmat([A, B])", "tensor<3x5xf32>");
    returns("bandedmat(v, 3)", "tensor<3x5xf32>");
    returns(
        "bincounts([0.0, 1.0, 2.0], [0.5, z0, 2.0])",
        "tensor<2xi32>",
    );
}

// Bins are [eᵢ, eᵢ₊₁), the last closed (§07 "Binning"). A bin that only touches a
// region at its own open end, or `posreals` at its open zero, does not meet it.
#[test]
fn selectbins_keeps_the_bins_that_meet_the_region() {
    let c = "[5.0, z0, 7.0]";
    returns(
        &format!("selectbins([0.0, 1.0, 2.0, 3.0], interval(0.5, 1.5), {c})"),
        "tensor<2xf32>",
    );
    returns(
        &format!("selectbins([0.0, 1.0, 2.0, 3.0], interval(2.0, 2.5), {c})"),
        "tensor<1xf32>",
    );
    returns(
        &format!("selectbins([0.0, 1.0, 2.0, 3.0], interval(3.0, 4.0), {c})"),
        "tensor<1xf32>",
    );
    returns(
        &format!("selectbins([-1.0, 0.0, 1.0, 2.0], posreals, {c})"),
        "tensor<2xf32>",
    );
}

#[test]
fn what_has_no_static_shape_refuses_with_its_reason() {
    refuses(
        "partition([1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])",
        "ragged array",
    );
    refuses(
        "filter(fn(_ > 1.5), v)",
        "data-dependent number of elements",
    );
    refuses(
        "onehot.([1, 2], 3)",
        "`onehot` under a broadcast has no lowering here",
    );
    refuses("reverse(A)", "`reverse` lowers a vector here");
    refuses(
        "splitblocks([1.0, 2.0, 3.0], 2)",
        "divisible by its block size",
    );
}
