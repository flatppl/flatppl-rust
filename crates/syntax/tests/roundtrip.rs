//! FlatPPL surface round-trip tests.
//!
//! Three contracts (all canonical, not source-preserving), checked per
//! fixture at both printer syntax levels:
//!   1. **Semantic preservation** — `parse → print → parse` yields a module
//!      whose FlatPIR projection is byte-identical to the original's.
//!   2. **Printer fixpoint** — `parse → print → parse → print` is byte-stable
//!      (the printed form re-parses to the same text).
//!   3. **FlatPPL → FlatPIR** — lowering to core then writing FlatPIR yields a
//!      module that re-reads to a stable FlatPIR fixpoint.
//!
//! `minimal.flatppl` is copied from the flatppl-js corpus (upstream `be8cc1a`);
//! the rest are hand-written to exercise the supported grammar.

use std::fs;
use std::path::PathBuf;

use flatppl_core::{CallHead, Node};
use flatppl_syntax::{Syntax, parse, print, print_with};

fn fixture(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../fixtures/flatppl", name]
        .iter()
        .collect();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn assert_print_roundtrip(name: &str, syntax: Syntax) {
    let src = fixture(name);
    let m1 = parse(&src).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
    let t1 = print_with(&m1, syntax);
    let m2 = parse(&t1).unwrap_or_else(|e| {
        panic!("{name}: re-parse of printed form failed: {e}\n--- printed ---\n{t1}")
    });
    assert_eq!(
        flatppl_flatpir::write(&m1),
        flatppl_flatpir::write(&m2),
        "{name}: printing changed the module\n--- printed ---\n{t1}"
    );
    let t2 = print_with(&m2, syntax);
    assert_eq!(t1, t2, "{name}: printer is not idempotent");
}

fn assert_flatpir_stable(name: &str) {
    let m1 = parse(&fixture(name)).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
    let pir1 = flatppl_flatpir::write(&m1);
    let m2 = flatppl_flatpir::read(&pir1)
        .unwrap_or_else(|e| panic!("{name}: FlatPIR re-read failed: {e}\n--- FlatPIR ---\n{pir1}"));
    let pir2 = flatppl_flatpir::write(&m2);
    assert_eq!(pir1, pir2, "{name}: FlatPPL→FlatPIR is not stable");
}

macro_rules! roundtrip_tests {
    ($($mod:ident => $file:literal),* $(,)?) => {
        $(mod $mod {
            use super::*;
            #[test] fn full_roundtrip() { assert_print_roundtrip($file, Syntax::Full); }
            #[test] fn minimal_roundtrip() { assert_print_roundtrip($file, Syntax::Minimal); }
            #[test] fn flatpir_stable() { assert_flatpir_stable($file); }
        })*
    };
}

roundtrip_tests! {
    minimal => "minimal.flatppl",
    eight_schools => "eight-schools.flatppl",
    einsum_matmul => "einsum-matmul.flatppl",
    expressions => "expressions.flatppl",
    values => "values.flatppl",
    modules => "modules.flatppl",
    lambdas => "lambdas.flatppl",
    metricsum => "metricsum.flatppl",
}

/// Pins both printer levels so accidental drift is caught.
#[test]
fn print_level_goldens() {
    let m = parse("x = 1 + 2\ny ~ Normal(0, 1)").unwrap();
    assert_eq!(print(&m), "x = 1 + 2\ny ~ Normal(0, 1)");
    assert_eq!(print(&m), print_with(&m, Syntax::Full));
    assert_eq!(
        print_with(&m, Syntax::Minimal),
        "x = add(1, 2)\ny ~ Normal(0, 1)"
    );
}

/// Full-syntax printing of already-canonical source is the identity —
/// operator precedence, associativity, and parenthesization survive exactly.
#[test]
fn full_syntax_is_identity_on_canonical_source() {
    let src = "\
a = elementof(reals)
b = elementof(reals)
c = elementof(reals)
d = elementof(reals)
p = (a + b) * c
q = a - (b - c)
r = a ^ b ^ c
s = (a + b) ^ 2
t = -a ^ b
u = (-a) ^ b
v = a / b / c
w = a < b && c < d
x = a < b <= c
y = !(a < b) || c < d
z = a .+ b .* c
n = .-a
e = exp.(a)
g = (f, k) -> f(k)
h = a in interval(0.0, 1.0)";
    let m = parse(src).unwrap();
    assert_eq!(print(&m), src);
}

/// Built-in-vs-user call resolution: a call to a module binding is a user call;
/// a call to an unbound name is a built-in.
#[test]
fn user_vs_builtin_calls() {
    let m = parse(&fixture("minimal.flatppl")).unwrap();
    let rhs_head = |name: &str| -> CallHead {
        let id = m
            .bindings()
            .find(|(_, b)| m.resolve(b.name) == name)
            .map(|(id, _)| id)
            .unwrap_or_else(|| panic!("no binding {name}"));
        match m.node(m.binding(id).rhs) {
            Node::Call(c) => c.head,
            other => panic!("{name} RHS is not a call: {other:?}"),
        }
    };
    // `sigma = f_sqrt(sigma2)` — f_sqrt is a binding → user call.
    assert!(matches!(rhs_head("sigma"), CallHead::User(_)));
    // `dist = kernel(kernel_input)` — kernel is a binding → user call.
    assert!(matches!(rhs_head("dist"), CallHead::User(_)));
    // `a = elementof(nonnegreals)` — elementof is a built-in.
    assert!(matches!(rhs_head("a"), CallHead::Builtin(_)));
}

/// `~` round-trips through `draw`: the printed form re-sugars to `~`.
#[test]
fn tilde_resugars() {
    let printed = print(&parse("z ~ Normal(0, 1)").unwrap());
    assert_eq!(printed, "z ~ Normal(0, 1)");
}

/// Reified callables (spec §11): an explicit boundary round-trips as kwargs
/// (`%specinputs`), a boundary-less reify prints bare (`%autoinputs`), and an
/// all-placeholder `functionof` boundary re-sugars to a lambda in full syntax
/// while staying spelled (`_x_`, `%local`) in minimal.
#[test]
fn reification_forms() {
    let src = "a = elementof(reals)\n\
               b = a ^ 0.5\n\
               f = functionof(b)\n\
               g = functionof(b, p = a)\n\
               h = functionof(_x_ * 2, x = _x_)\n\
               k = kernelof(b, p = a)";
    let m = parse(src).unwrap();
    let full = print(&m);
    assert!(full.contains("f = functionof(b)"), "got:\n{full}");
    assert!(full.contains("g = functionof(b, p = a)"), "got:\n{full}");
    assert!(full.contains("h = x -> x * 2"), "got:\n{full}");
    assert!(full.contains("k = kernelof(b, p = a)"), "got:\n{full}");

    let minimal = print_with(&m, Syntax::Minimal);
    assert!(
        minimal.contains("h = functionof(mul(_x_, 2), x = _x_)"),
        "got:\n{minimal}"
    );

    // The FlatPIR projection carries the origin tags.
    let pir = flatppl_flatpir::write(&m);
    assert!(
        pir.contains("(functionof (%ref self b) %autoinputs %deferred)"),
        "got:\n{pir}"
    );
    assert!(
        pir.contains("(functionof (%ref self b) %specinputs ((p (%ref self a))))"),
        "got:\n{pir}"
    );
    assert!(
        pir.contains("%specinputs ((x (%ref %local _x_)))"),
        "got:\n{pir}"
    );

    // A boundary value must be a module binding or a placeholder.
    assert!(parse("f = functionof(x, p = unknown)").is_err());
}

/// Inline application (spec §05 `Postfix Call` / §11 expression-headed
/// `%call`): a callable-valued expression is applied directly — the callee of
/// `(%call <callable> …)` is an expression, a `(%ref …)` in the common case.
#[test]
fn inline_application_lowers() {
    let src = "a = elementof(reals)\n\
               b = a * 2.0\n\
               y = functionof(b, p = a)(2.5)";
    let printed = print(&parse(src).unwrap());
    assert!(
        printed.contains("y = functionof(b, p = a)(2.5)"),
        "got:\n{printed}"
    );
    let pir = flatppl_flatpir::write(&parse(src).unwrap());
    assert!(
        pir.contains("(%call (functionof (%ref self b) %specinputs ((p (%ref self a)))) 2.5)"),
        "got:\n{pir}"
    );

    // Chained application: the result of a call is itself applicable.
    let src = "f = functionof(_x_ * 2, x = _x_)\nz = f(1.0)(2.0)";
    let printed = print(&parse(src).unwrap());
    assert!(printed.contains("z = f(1.0)(2.0)"), "got:\n{printed}");
    let pir = flatppl_flatpir::write(&parse(src).unwrap());
    assert!(
        pir.contains("(%call (%call (%ref self f) 1.0) 2.0)"),
        "got:\n{pir}"
    );
}

/// Member access: `mod.x` is a cross-module ref (never `get`), `self.x` is a
/// current-module ref (prints bare — the canonical form), `base.x` is the
/// built-in (lowers to the bare form, spec §11); each also works as a call
/// head. Field access on a non-module value lowers to `get` and re-sugars to
/// dot syntax in full output only.
#[test]
fn member_access_lowers() {
    let src = "m = load_module(\"x.flatppl\")\n\
               r = record(obs = 1.0)\n\
               y = m.obs\n\
               z = m.f(1.0)\n\
               s = self.y\n\
               b = base.pi\n\
               c = base.add(1, 2)\n\
               f = r.obs";
    let m = parse(src).unwrap();
    let full = print(&m);
    assert!(full.contains("y = m.obs"), "got:\n{full}");
    assert!(full.contains("z = m.f(1.0)"), "got:\n{full}");
    assert!(full.contains("s = y"), "got:\n{full}");
    assert!(full.contains("b = pi"), "got:\n{full}");
    assert!(full.contains("c = 1 + 2"), "got:\n{full}");
    assert!(full.contains("f = r.obs"), "got:\n{full}");

    let minimal = print_with(&m, Syntax::Minimal);
    assert!(minimal.contains("c = add(1, 2)"), "got:\n{minimal}");
    assert!(minimal.contains("f = get(r, \"obs\")"), "got:\n{minimal}");

    let pir = flatppl_flatpir::write(&m);
    assert!(pir.contains("(%ref m obs)"), "got:\n{pir}");
    assert!(pir.contains("(%call (%ref m f) 1.0)"), "got:\n{pir}");
    assert!(pir.contains("(%ref self y)"), "got:\n{pir}");
}

/// Lambda and `fn` hole sugar lower to `functionof` with placeholders
/// (spec §04): lambda args rewrite their free occurrences to `_name_`; each
/// `_` hole is a distinct positional input `arg<n>` in reading order. The
/// minimal printer pins the lowering; full syntax re-sugars to a lambda
/// (`fn` holes canonicalize to a lambda too — indistinguishable once lowered).
#[test]
fn lambda_and_fn_lower() {
    let lower = |src: &str| print_with(&parse(src).unwrap(), Syntax::Minimal);

    assert_eq!(
        lower("f = x -> 2 * x + 1"),
        "f = functionof(add(mul(2, _x_), 1), x = _x_)"
    );
    assert_eq!(
        print(&parse("f = x -> 2 * x + 1").unwrap()),
        "f = x -> 2 * x + 1"
    );

    assert_eq!(
        lower("g = (x, y) -> x * y + 1"),
        "g = functionof(add(mul(_x_, _y_), 1), x = _x_, y = _y_)"
    );
    assert_eq!(
        print(&parse("g = (x, y) -> x * y + 1").unwrap()),
        "g = (x, y) -> x * y + 1"
    );

    // Lambda args shadow module bindings of the same name.
    let printed = lower("x = elementof(reals)\nf = x -> x + x");
    assert!(
        printed.contains("f = functionof(add(_x_, _x_), x = _x_)"),
        "got:\n{printed}"
    );

    // fn holes: each `_` is distinct, left-to-right; full output is a lambda.
    assert_eq!(
        lower("neg = fn(0 - _)"),
        "neg = functionof(sub(0, _arg1_), arg1 = _arg1_)"
    );
    assert_eq!(
        print(&parse("neg = fn(0 - _)").unwrap()),
        "neg = arg1 -> 0 - arg1"
    );

    assert_eq!(
        lower("h = fn((_ / _) ^ 2)"),
        "h = functionof(pow(divide(_arg1_, _arg2_), 2), arg1 = _arg1_, arg2 = _arg2_)"
    );
    assert_eq!(
        print(&parse("h = fn((_ / _) ^ 2)").unwrap()),
        "h = (arg1, arg2) -> (arg1 / arg2) ^ 2"
    );

    // A lambda argument used as a call head becomes a `%local`-headed call.
    let pir = flatppl_flatpir::write(&parse("apply = (f, x) -> f(x)").unwrap());
    assert!(
        pir.contains("(%call (%ref %local _f_) (%ref %local _x_))"),
        "got:\n{pir}"
    );
}

#[test]
fn lambda_and_fn_errors() {
    assert!(parse("f = (x) -> x + 1").is_err()); // parenthesised single arg
    assert!(parse("f = fn(x + 1)").is_err()); // no holes
    assert!(parse("y = 1 + _").is_err()); // hole outside fn
    assert!(parse("f = true -> 1").is_err()); // reserved word as arg
}

/// Function-definition sugar (spec §05 "Function definition syntax"):
/// `f(args) = expr` is exactly the lambda binding `f = (args) -> expr`, so it
/// lowers to the same `functionof` and produces byte-identical FlatPIR. There
/// is no dedicated node or printed form — full syntax re-sugars to the lambda.
#[test]
fn function_definition_lowers() {
    // Multi- and single-argument forms are identical to their lambda bindings.
    for (def, lam) in [
        ("f(x, y) = x * y + 1", "f = (x, y) -> x * y + 1"),
        ("g(x) = 2 * x + 1", "g = x -> 2 * x + 1"),
        // The body type is the output type — a record body is multi-output.
        (
            "h(x, y) = record(p = x + y, q = x * y)",
            "h = (x, y) -> record(p = x + y, q = x * y)",
        ),
    ] {
        assert_eq!(
            flatppl_flatpir::write(&parse(def).unwrap()),
            flatppl_flatpir::write(&parse(lam).unwrap()),
            "`{def}` should lower identically to `{lam}`"
        );
        // Full syntax has no `f(args) =` form; it re-sugars to the lambda.
        assert_eq!(print(&parse(def).unwrap()), lam);
    }

    // Definition args shadow module bindings, exactly as lambda args do.
    assert_eq!(
        print_with(
            &parse("x = elementof(reals)\nf(x) = x + x").unwrap(),
            Syntax::Minimal
        ),
        print_with(
            &parse("x = elementof(reals)\nf = x -> x + x").unwrap(),
            Syntax::Minimal
        ),
    );
}

#[test]
fn function_definition_errors() {
    assert!(parse("f() = 1").is_err()); // no nullary definitions
    assert!(parse("f(x = a) = 1").is_err()); // params are bare names, not kwargs
    assert!(parse("f(true) = 1").is_err()); // reserved word as arg
    assert!(parse("f(_) = 1").is_err()); // placeholder/discard as arg
    assert!(parse("f(x,) = 1").is_err()); // trailing comma, no name
    // A bare call is still not a statement, with or without the new arm.
    assert!(parse("f(a, b)").is_err());
}

/// Placeholders are scoped to the nearest enclosing reification (spec §04):
/// referencing an enclosing lambda's argument — or an enclosing `fn`'s hole —
/// from inside a nested lambda / `fn` / `functionof` would leave the
/// placeholder unbound there (the spec's DISALLOWED case), so the parser
/// rejects the capture at the point it would introduce the placeholder.
#[test]
fn cross_reification_capture_rejected() {
    // Outer lambda arg inside a nested lambda.
    assert!(parse("f = x -> broadcast(y -> x * y, v)").is_err());
    // Outer lambda arg inside a nested `fn(…)`.
    assert!(parse("f = x -> fn(_ + x)").is_err());
    // Outer lambda arg inside a nested explicit reification (body or kwargs).
    assert!(parse("a = elementof(reals)\nf = x -> functionof(a + x, p = a)").is_err());
    assert!(parse("a = elementof(reals)\nf = x -> kernelof(a, p = x)").is_err());
    // A hole separated from its `fn(…)` by a nested lambda.
    assert!(parse("g = fn(broadcast(y -> _ * y, v))").is_err());

    // Shadowing the same name in the inner lambda stays legal (distinct
    // scopes, spec §04), and so does an inner lambda over module bindings.
    assert!(parse("h = x -> broadcast(x -> 2 * x, v)").is_ok());
    let src = "c = elementof(reals)\nk = x -> broadcast(y -> c * y, x)";
    let printed = print_with(&parse(src).unwrap(), Syntax::Minimal);
    assert!(
        printed.contains("functionof(mul(c, _y_), y = _y_)"),
        "got:\n{printed}"
    );
}

/// Reserved words, reserved modules, and placeholders cannot be bound;
/// `_` discards by binding to a fresh auto-generated private name.
#[test]
fn binding_name_rules() {
    for src in [
        "in = 1",
        "all = 1",
        "only = 1",
        "true = 1",
        "false = 1",
        "self = 1",
        "base = 1",
        "_x_ = 1",
    ] {
        assert!(parse(src).is_err(), "`{src}` should be rejected");
    }
    let printed = print(&parse("_ = exp(1.0)").unwrap());
    assert_eq!(printed, "__0x1 = exp(1.0)");
}

/// `:` lowers to the `all` selector, a trailing `!` to `only`, and a `!` not
/// followed by `,` / `]` is unary logical-not (spec §05 disambiguation).
/// `in` lowers to the membership builtin. Full syntax inverts each lowering.
#[test]
fn slicing_and_membership_lower() {
    for (src, minimal) in [
        ("col = M[:, 2]", "col = get(M, all, 2)"),
        ("sole = v[!]", "sole = get(v, only)"),
        ("x = M[!, :]", "x = get(M, only, all)"),
        ("y = v[!false + 1]", "y = get(v, add(lnot(false), 1))"),
        (
            "m = x in interval(0.0, 1.0)",
            "m = in(x, interval(0.0, 1.0))",
        ),
    ] {
        let m = parse(src).unwrap();
        assert_eq!(print_with(&m, Syntax::Minimal), minimal);
        assert_eq!(print(&m), src, "full syntax should invert the lowering");
    }
}

/// The `metric: result[.axes…] := expr` statement lowers to a `metricsum`
/// call; the metric name is a reference, not a binding. Variance markers
/// carry through to `(%uaxis …)` / `(%laxis …)` in FlatPIR (spec §11).
#[test]
fn metricsum_lowers() {
    let src = "g = eye(2)\ng: L[.mu^, .rho_] := T1[.mu^, .nu_] * T2[.nu^, .rho_]";
    let m = parse(src).unwrap();
    let minimal = print_with(&m, Syntax::Minimal);
    assert!(
        minimal.contains(
            "L = metricsum(g, [.mu^, .rho_], \
             mul(get(T1, .mu^, .nu_), get(T2, .nu^, .rho_)))"
        ),
        "got:\n{minimal}"
    );
    // Full syntax restores the statement form exactly.
    assert_eq!(print(&m), src);

    let pir = flatppl_flatpir::write(&m);
    assert!(pir.contains("(%uaxis mu)"), "got:\n{pir}");
    assert!(pir.contains("(%laxis rho)"), "got:\n{pir}");
    // The metric resolves like any name: bound → self-ref.
    assert!(pir.contains("(metricsum (%ref self g)"), "got:\n{pir}");

    // The metric position takes a single name (grammar), and axis names may
    // not end in `_` once the variance marker is stripped.
    assert!(parse("a, b: C[.i] := x").is_err());
    assert!(parse("g: C[.i__] := x").is_err());
}

#[test]
fn reports_malformed() {
    assert!(parse("x = ").is_err()); // missing RHS
    assert!(parse("x + 1").is_err()); // not a binding
}

/// Decomposition lowers to a shared synthetic tmp + positional projections
/// (`_` discarded); `:=` lowers to a sum-aggregate over an axis-array literal.
/// Full syntax restores the `:=` statement form (decomposition stays lowered —
/// the projections are ordinary bindings).
#[test]
fn decomposition_and_aggregate_lower() {
    let m = parse("a, _ = pair").unwrap();
    let minimal = print_with(&m, Syntax::Minimal);
    assert!(minimal.contains("__0x1 = pair"), "got:\n{minimal}");
    assert!(minimal.contains("a = get(__0x1, 1)"), "got:\n{minimal}");
    assert!(print(&m).contains("a = __0x1[1]"), "got:\n{}", print(&m));

    let m = parse("s[] := v").unwrap();
    assert_eq!(print_with(&m, Syntax::Minimal), "s = aggregate(sum, [], v)");
    assert_eq!(print(&m), "s[] := v");

    let m = parse("C[.i, .k] := A[.i, .j] * B[.j, .k]").unwrap();
    assert_eq!(
        print_with(&m, Syntax::Minimal),
        "C = aggregate(sum, [.i, .k], mul(get(A, .i, .j), get(B, .j, .k)))"
    );
    assert_eq!(print(&m), "C[.i, .k] := A[.i, .j] * B[.j, .k]");
}

/// Sugar is re-applied only where the re-parse provably inverts it; these
/// stay in call form.
#[test]
fn full_syntax_guards() {
    // Only the `sum` reduction has the `:=` statement form.
    let printed = print(&parse("s = aggregate(prod, [.i], v)").unwrap());
    assert_eq!(printed, "s = aggregate(prod, [.i], v)");

    // A module-qualified metric is not a single `Name`: no statement form.
    let src = "m = load_module(\"x.flatppl\")\nL = metricsum(m.g, [.i^, .j_], v)";
    let printed = print(&parse(src).unwrap());
    assert!(
        printed.contains("L = metricsum(m.g, [.i^, .j_], v)"),
        "got:\n{printed}"
    );

    // Expression-position aggregates keep the call form (`:=` is a statement).
    let printed = print(&parse("y = 1 + aggregate(sum, [.i], w[.i])").unwrap());
    assert_eq!(printed, "y = 1 + aggregate(sum, [.i], w[.i])");

    // A reserved word is not a surface field name; string-keyed indexing is
    // the fallback (it lowers to the same `get`).
    let printed = print(&parse("z = get(r, \"true\")").unwrap());
    assert_eq!(printed, "z = r[\"true\"]");

    // Dot syntax on a module binding means member access, not field access;
    // again the index form is the safe spelling.
    let src = "m = load_module(\"x.flatppl\")\nz = get(m, \"x\")";
    let printed = print(&parse(src).unwrap());
    assert!(printed.contains("z = m[\"x\"]"), "got:\n{printed}");

    // A built-in shadowed by a module binding prints through `base.` (the
    // bare spelling would re-resolve to the binding).
    let src = "neg = elementof(reals)\ny = base.neg(1.0)";
    let m = parse(src).unwrap();
    assert!(print(&m).contains("y = -1.0"), "got:\n{}", print(&m));
    assert!(
        print_with(&m, Syntax::Minimal).contains("y = base.neg(1.0)"),
        "got:\n{}",
        print_with(&m, Syntax::Minimal)
    );

    // `kernelof` has no lambda form (the body still gets expression sugar).
    let printed = print(&parse("k = kernelof(_x_ * 2, x = _x_)").unwrap());
    assert_eq!(printed, "k = kernelof(_x_ * 2, x = _x_)");
}

/// A binding carries at most one doc-comment, leading or trailing but not
/// both (spec §04 Documentation). Stacked leading docs, a leading+trailing
/// pair, and a doc interrupting a statement are errors — accepting them
/// would silently drop documentation content.
#[test]
fn doc_comment_cardinality() {
    assert!(parse("% one\nx = 1").is_ok());
    assert!(parse("x = 1 % one").is_ok());
    assert!(parse("% one\n% two\nx = 1").is_err());
    assert!(parse("% one\nx = 1 % two").is_err());
    assert!(parse("x = (1 +\n% nope\n2)").is_err());
}

/// A dot-call head that is not postfix-able (a lambda, a numeric literal)
/// is parenthesized so the printed form re-lexes correctly.
#[test]
fn dotcall_head_parenthesized() {
    let src = "mapped = broadcast((a, b) -> a * b, [1.0, 2.0], [3.0, 4.0])";
    assert_eq!(
        print(&parse(src).unwrap()),
        "mapped = ((a, b) -> a * b).([1.0, 2.0], [3.0, 4.0])"
    );
}
