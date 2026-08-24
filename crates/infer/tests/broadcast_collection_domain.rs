//! A reduction under a broadcast is applied PER ELEMENT, so over a vector of scalars
//! it is out of domain — and over a vector of vectors it reduces each inner vector.
//!
//! §04 "Broadcasting": `broadcast(f_or_K, ...)` "maps a function or kernel elementwise
//! over arrays", and its "Collection arguments" paragraph iterates every collection
//! argument ("All collection arguments (arrays and tables) must have the same number
//! of axes"). Its "Non-collection inputs" paragraph is the contrast: only scalars,
//! functions, kernels and measures are "held constant while collection arguments are
//! iterated over". An array argument is iterated, so the head sees the ELEMENT.
//!
//! §07 "Reductions" gives `sum` the Domains cell "real/complex arrays", `median` "real
//! arrays", `lengthof` "vectors, tables"; §07 "Boolean reductions" gives `lany`/`lall`
//! "boolean arrays"; §07 "Norms and normalization" gives `l2norm` "real/complex
//! vectors"; §07 "Linear algebra" gives `transpose` "vectors, matrices" and `self_outer`
//! "vectors"; §07 "Array and table operations" gives `reverse` "vectors, tables". None of
//! them admits a scalar. So `sum.(v)` over a vector of scalars asks for the sum of a
//! scalar and is ill-typed, and the same argument reaches every head in those six tables.
//!
//! `cat` is the one row swept and REJECTED: §07 gives it "scalars, vectors, or records",
//! so a per-element `cat` of two scalars is well formed. `min`/`max` are scalar-domain
//! for the same reason and stay in the elementwise cell table.
//!
//! §03 "Arrays" makes the legal case reachable: arrays are "collections of scalar
//! values ... or arrays", and "Vectors of vectors are not interpreted as matrices
//! implicitly". Over such a vector the cells ARE arrays, so `sum.(vv)` is the vector
//! of per-inner-vector sums.
//!
//! Before this every one of these typed `%deferred` with no diagnostic, and the
//! StableHLO emitter discarded the `broadcast` wrapper and emitted the WHOLE-array
//! reduction — `sum.(v)` answered with `sum(v)`, exit 0. Both halves are pinned:
//! the scalar-cell spelling refuses here, and the nested spelling types the vector of
//! per-cell reductions.

use flatppl_infer::infer;

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn bind_line(src: &str, name: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    let out = flatppl_flatpir::write(&m);
    out.lines()
        .find(|l| l.contains(&format!("(%bind {name} ")))
        .unwrap_or_else(|| panic!("no `{name}` binding in:\n{out}"))
        .trim()
        .to_string()
}

/// The vector operand whose elements are scalars — the shape every witness below
/// starts from.
fn scalar_cells(head: &str, set: &str) -> String {
    format!("v = elementof(cartpow({set}, [4]))\ny = {head}.(v)\n")
}

/// Every collection-domain head, dotted over a vector of SCALARS. This is the
/// mislowering the refusal closes: each of these compiled to the undotted reduction's
/// number at exit 0.
#[test]
fn a_dotted_reduction_over_scalar_cells_is_refused() {
    for head in [
        "sum",
        "mean",
        "var",
        "std",
        "prod",
        "maximum",
        "minimum",
        "median",
        "lengthof",
        "l1norm",
        "l2norm",
        "linfnorm",
        "logsumexp",
        "softmax",
        "logsoftmax",
        "l1unit",
        "l2unit",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "sizeof",
        "indicesof",
        "indicesof0",
    ] {
        let errs = errors(&scalar_cells(head, "reals"));
        assert_eq!(
            errs.len(),
            1,
            "`{head}.(v)` must refuse exactly once: {errs:?}"
        );
        assert!(
            errs[0].contains(&format!("`{head}` under a broadcast")),
            "`{head}`'s refusal must name the head and the broadcast: {}",
            errs[0]
        );
    }
    for head in ["lany", "lall"] {
        let errs = errors(&scalar_cells(head, "booleans"));
        assert_eq!(
            errs.len(),
            1,
            "`{head}.(b)` must refuse exactly once: {errs:?}"
        );
    }
}

/// §07 "Linear algebra" and §07 "Array and table operations" are the same rule one table
/// over, and three of them were MEASURED mislowering at exit 0 with the wrapper
/// discarded: `transpose.(v)` and `adjoint.(v)` returned `%arg0` unchanged, and
/// `self_outer.(v)` emitted a `tensor<4x4xf32>` outer product.
#[test]
fn the_linear_algebra_and_array_op_heads_are_refused_too() {
    for head in [
        // The three measured mislowerers first.
        "transpose",
        "adjoint",
        "self_outer",
        // The rest of §07 "Linear algebra".
        "det",
        "logabsdet",
        "inv",
        "trace",
        "linsolve",
        "qr",
        "lower_cholesky",
        "row_gram",
        "col_gram",
        "cross",
        "diagmat",
        "diag",
        "quadform",
        // §07 "Array and table operations", less `cat`.
        "rowstack",
        "colstack",
        "tile",
        "splitblocks",
        "joinblocks",
        "partition",
        "reverse",
        "addaxes",
        "blockdiagmat",
        "bandedmat",
    ] {
        let errs = errors(&scalar_cells(head, "reals"));
        assert_eq!(
            errs.len(),
            1,
            "`{head}.(v)` must refuse exactly once: {errs:?}"
        );
        assert!(
            errs[0].contains(&format!("`{head}` under a broadcast")),
            "`{head}`'s refusal must name the head: {}",
            errs[0]
        );
    }
    assert!(
        errors(&scalar_cells("transpose", "reals"))[0].contains("§07 \"Linear algebra\""),
        "`transpose` belongs to §07's linear-algebra table"
    );
    assert!(
        errors(&scalar_cells("reverse", "reals"))[0].contains("§07 \"Array and table operations\""),
        "`reverse` belongs to §07's array-ops table"
    );
}

/// `cat` is the one row of §07 "Array and table operations" that ADMITS a scalar —
/// "`cat(scalar1, scalar2, ...)` … produces a vector of those scalars" — so a per-element
/// `cat` is well formed and must not be swept up with its table-mates.
#[test]
fn cat_admits_a_scalar_cell_and_is_not_refused() {
    let errs = errors(
        "v = elementof(cartpow(reals, [4]))\nw = elementof(cartpow(reals, [4]))\ny = cat.(v, w)\n",
    );
    assert!(
        errs.is_empty(),
        "`cat` takes scalars per §07, so the dotted form is legal: {errs:?}"
    );
}

/// The message has to carry the two things a reader needs: WHY it is refused (the
/// §07 domain the cell misses, and §04's per-element rule) and WHAT to write instead.
#[test]
fn the_refusal_cites_the_domain_and_names_both_remedies() {
    let errs = errors(&scalar_cells("sum", "reals"));
    let msg = &errs[0];
    assert!(
        msg.contains("§07 \"Reductions\"") && msg.contains("real/complex arrays"),
        "must quote §07's Domains cell: {msg}"
    );
    assert!(
        msg.contains("§04 \"Broadcasting\"") && msg.contains("elementwise"),
        "must cite §04's per-element rule: {msg}"
    );
    assert!(
        msg.contains("`sum(v)`") && msg.contains("aggregate(sum, [.i]"),
        "must name the whole-array reduction and the per-axis one: {msg}"
    );
    // The remedy may only name a spelling that is itself callable. §07 gives `quantile`
    // two arguments, and §04's ten eligible built-ins exclude it — so neither
    // `quantile(v)` nor `aggregate(quantile, …)` may appear.
    let q = errors("v = elementof(cartpow(reals, [4]))\ny = quantile.(v, 0.5)\n")[0].clone();
    assert!(
        q.contains("`quantile(v, p)`") && !q.contains("aggregate"),
        "`quantile`'s remedy must carry its second argument and skip aggregate: {q}"
    );
    // §07 "Cumulative operations": the scans "are not eligible reductions for
    // multi-axis aggregation", so the remedy must not offer one.
    let c = errors(&scalar_cells("cumsum", "reals"))[0].clone();
    assert!(
        c.contains("`cumsum(v)`") && !c.contains("aggregate"),
        "a scan is not §04-eligible: {c}"
    );
    // The per-table citation must follow the head, not default to "Reductions".
    assert!(
        errors(&scalar_cells("lany", "booleans"))[0].contains("§07 \"Boolean reductions\""),
        "`lany` belongs to §07's boolean-reduction table"
    );
    assert!(
        errors(&scalar_cells("l2norm", "reals"))[0].contains("§07 \"Norms and normalization\""),
        "`l2norm` belongs to §07's norms table"
    );
    assert!(
        errors(&scalar_cells("cumsum", "reals"))[0].contains("§07 \"Cumulative operations\""),
        "`cumsum` belongs to §07's cumulative table"
    );
    // §07 "Reductions" gives `quantile` the cell "real arrays, `interval(0, 1)`", and the
    // table claims to carry the cell as §07 writes it — so the whole cell, not just the
    // operand's half.
    assert!(
        errors("v = elementof(cartpow(reals, [4]))\ny = quantile.(v, 0.5)\n")[0]
            .contains("real arrays, `interval(0, 1)`"),
        "`quantile`'s Domains cell must be carried whole"
    );
}

/// A matrix operand is iterated over BOTH axes, so its cells are scalars too — and
/// this spelling was the worst of the family, contracting a `[2, 3]` matrix all the
/// way to a scalar.
#[test]
fn a_dotted_reduction_over_a_matrix_is_refused() {
    let errs = errors("M = elementof(cartpow(reals, [2, 3]))\ny = sum.(M)\n");
    assert_eq!(errs.len(), 1, "`sum.(M)` must refuse: {errs:?}");
    assert!(errs[0].contains("`sum` under a broadcast"), "{}", errs[0]);
}

/// The explicit spelling is the same call — §04 makes `f.(args)` sugar for
/// `broadcast(f, args)`, so neither may be the loophole.
#[test]
fn the_explicit_broadcast_spelling_refuses_alike() {
    for src in [
        "v = elementof(cartpow(reals, [4]))\ny = broadcast(sum, v)\n",
        "v = elementof(cartpow(reals, [4]))\ny = sum.(v)\n",
    ] {
        assert_eq!(errors(src).len(), 1, "must refuse: {src}");
    }
}

/// The LEGAL case, and the one that makes the refusal a domain rule rather than a
/// blanket ban: over a vector of vectors the cells are arrays, so the dotted form is
/// the vector of per-inner-vector reductions.
#[test]
fn a_dotted_reduction_over_nested_arrays_types_the_per_cell_reduction() {
    let src = "\
a = elementof(cartpow(reals, [3]))
b = elementof(cartpow(reals, [3]))
vv = [a, b]
y = sum.(vv)
";
    // The operand really is a vector of vectors, not a matrix (§03).
    assert!(
        bind_line(src, "vv").contains("(%array 1 (2) (%array 1 (3) (%scalar real)))"),
        "{}",
        bind_line(src, "vv")
    );
    // Two inner vectors in, two sums out — NOT the scalar the emitter used to give.
    assert!(
        bind_line(src, "y").contains("(%array 1 (2) (%scalar real))"),
        "`sum.(vv)` is the vector of per-inner-vector sums: {}",
        bind_line(src, "y")
    );
}

/// The bare form is untouched: §07 reduces the whole array, and over a nested array
/// that is the reduction along the OUTER axis. Held beside the dotted form so the two
/// answers cannot merge again.
#[test]
fn the_bare_reduction_still_reduces_the_whole_array() {
    let flat = "v = elementof(cartpow(reals, [4]))\ny = sum(v)\n";
    assert!(
        bind_line(flat, "y").contains("(%scalar real)"),
        "{}",
        bind_line(flat, "y")
    );
    let nested = "\
a = elementof(cartpow(reals, [3]))
b = elementof(cartpow(reals, [3]))
vv = [a, b]
y = sum(vv)
";
    assert!(
        bind_line(nested, "y").contains("(%array 1 (3) (%scalar real))"),
        "bare `sum` reduces the outer axis: {}",
        bind_line(nested, "y")
    );
}

/// Per-head cell types over nested arrays, so no head silently borrows another's
/// answer. `reduced_scalar`'s promotions have to survive the broadcast: `sum` over
/// boolean cells is `Integer`, `lany` is `Boolean`, `median`/`var` are `Real` over
/// integer cells.
#[test]
fn the_per_cell_type_follows_the_head_not_the_broadcast() {
    let nested = |set: &str, head: &str| {
        format!(
            "a = elementof(cartpow({set}, [3]))\nb = elementof(cartpow({set}, [3]))\n\
             vv = [a, b]\ny = {head}.(vv)\n"
        )
    };
    for (set, head, elem) in [
        ("booleans", "sum", "(%scalar integer)"),
        ("booleans", "lany", "(%scalar boolean)"),
        ("booleans", "lall", "(%scalar boolean)"),
        ("integers", "sum", "(%scalar integer)"),
        ("integers", "maximum", "(%scalar integer)"),
        ("integers", "median", "(%scalar real)"),
        ("integers", "var", "(%scalar real)"),
        ("reals", "l2norm", "(%scalar real)"),
        ("reals", "lengthof", "(%scalar integer)"),
    ] {
        let line = bind_line(&nested(set, head), "y");
        assert!(
            line.contains(&format!("(%array 1 (2) {elem})")),
            "`{head}.(vv)` over {set} cells must be a [2] array of {elem}: {line}"
        );
    }
}

/// The refusal must not spill onto the ELEMENTWISE heads that share the cell table —
/// `min`/`max` are §07's two-argument scalar ops, not the `minimum`/`maximum`
/// reductions, and the dotted comparisons are the route §07 gives a mask.
#[test]
fn the_elementwise_heads_are_untouched() {
    for src in [
        "v = elementof(cartpow(reals, [4]))\ny = max.(v, 0.0)\n",
        "v = elementof(cartpow(reals, [4]))\ny = min.(v, 0.0)\n",
        "v = elementof(cartpow(reals, [4]))\ny = v .+ 1.0\n",
        "v = elementof(cartpow(reals, [4]))\ny = exp.(v)\n",
        "v = elementof(cartpow(reals, [4]))\ny = gt.(v, 3.0)\n",
    ] {
        assert!(errors(src).is_empty(), "must still type clean: {src}");
    }
}

/// A head whose cell type is not yet known must not be accused. The refusal keys on a
/// cell that IS a scalar, never on the absence of an answer.
#[test]
fn an_unresolved_cell_is_not_refused() {
    // A kernel-broadcast operand: the head is a distribution, not a reduction, and
    // nothing here is a collection-domain head at all.
    let src = "\
v = elementof(cartpow(reals, [4]))
d ~ Normal.(v, 1.0)
";
    assert!(
        errors(src).is_empty(),
        "a distribution broadcast is unaffected"
    );
}
