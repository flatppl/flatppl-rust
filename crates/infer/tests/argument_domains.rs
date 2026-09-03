//! Argument-domain rules: the domains a spec sentence states as an EXCLUSION.
//!
//! Three of them — §03 "Bool"'s boolean-only conditional and logical constructs,
//! §03 "Scalar types"'s omission of strings from the value types, and the measure
//! operands §06 names per operation. Every case here typed at exit 0 before,
//! so an ill-typed model reached the determiniser and the StableHLO emitter.
//!
//! The scalar-KIND tags a catalogue row carries are deliberately NOT swept; see
//! `ops::domain_check`'s doc comment for why, and `flatppl-dev/audit-fix-infer.md`
//! for the spec question behind it.
//!
//! Assertions read the DIAGNOSTIC SET, not the rendered module: a substring of
//! the printed IR can be satisfied by a child node, and a refusal that produced
//! no diagnostic would pass such a check silently. Every refusal is also checked
//! to carry a source location, since a diagnostic with none is unusable in an
//! editor and unreportable by the linter.

use flatppl_infer::{Diagnostic, Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d: &Diagnostic| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// Every error diagnostic carries a node to anchor it at. A refusal with no
/// location is not usable in an editor and not reportable by the linter.
fn errors_are_located(src: &str) -> bool {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut m);
    let errs: Vec<&Diagnostic> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    !errs.is_empty() && errs.iter().all(|d| d.node.is_some())
}

fn assert_refuses(src: &str, needle: &str) {
    let errs = errors(src);
    assert!(
        errs.iter().any(|e| e.contains(needle)),
        "`{src}` must refuse mentioning {needle}; got: {errs:?}"
    );
    assert!(errors_are_located(src), "`{src}` refused with no location");
}

fn assert_clean(src: &str) {
    let errs = errors(src);
    assert!(errs.is_empty(), "`{src}` must type cleanly; got: {errs:?}");
}

// ---- §03 "Bool": the conditional and logical domains -------------------------

/// §03 "Bool": the conditional and logical constructs "strictly require boolean
/// arguments; zero and one are not implicitly converted to booleans." Before
/// this, `land(1, 2)` typed `%boolean` and `ifelse(3, 4, 5)` typed the common
/// type of its two branches.
#[test]
fn integer_operands_to_the_logical_heads_are_refused() {
    for src in [
        "x = land(1, 2)\n",
        "x = lor(1, 2)\n",
        "x = lxor(1, 2)\n",
        "x = lnot(1)\n",
        "y = ifelse(3, 4, 5)\n",
    ] {
        assert_refuses(src, "requires a boolean");
    }
}

/// The same heads on genuine booleans keep typing. §03's inclusion
/// `booleans` $\subset$ `integers` runs one way only, so a boolean argument is
/// admissible everywhere and this is the direction that must not regress.
#[test]
fn boolean_operands_to_the_logical_heads_still_type() {
    assert_clean("x = land(true, false)\n");
    assert_clean("x = lnot(true)\n");
    assert_clean("y = ifelse(true, 4, 5)\n");
    assert_clean("a = gt(1.0, 0.0)\nb = lnot(a)\n");
}

/// `ifelse`'s `a` and `b` have the domain `anything` (§07 "Logic and
/// conditionals"), so only `cond` is constrained.
#[test]
fn ifelse_branches_are_unconstrained() {
    assert_clean("y = ifelse(true, 1, 2.5)\n");
}

// ---- §03 "Scalar types": a string is not a value ----------------------------

/// §03 "Scalar types" gives the scalar types as real, integer, boolean and
/// complex. A string is none of them, and this crate types one `%any`, which is
/// what let it reach every numeric rule: `sqrt("text")` typed a non-negative real
/// and `Normal(mu = "bad", sigma = 1.0)` a normalized measure over `reals`.
#[test]
fn a_string_where_no_textual_parameter_is_documented_is_refused() {
    assert_refuses("z = sqrt(\"text\")\n", "which is not a FlatPPL value");
    assert_refuses("d = Normal(mu = \"bad\", sigma = 1.0)\n", "which is not a");
}

/// A string bound to a name and then passed is caught the same way an inline
/// literal is: the check resolves through a self-module reference, because a
/// string carries no type to inspect.
#[test]
fn a_string_reaches_the_check_through_a_binding() {
    assert_refuses("s = \"text\"\nz = sqrt(s)\n", "which is not a");
}

/// A self-referential binding must not hang the string walk. `s = s` is a
/// well-formed parse that inference reports as a cycle, and this check runs on
/// every argument before the cycle is reported — an unbounded walk here hung
/// `named_set_expression.rs::a_self_referential_set_binding_reports_a_cycle_not_a_crash`.
/// The alias walk is bounded, so the pass that owns cycles still reports it.
#[test]
fn a_self_referential_binding_does_not_hang_the_string_walk() {
    let errs = errors("s = s\nx = elementof(s)\n");
    assert!(
        errs.iter().any(|e| e.contains("reference cycle")),
        "the cycle is still reported, and reported by the pass that owns it: {errs:?}"
    );
}

/// The rows that DO document a textual parameter keep working. Each is a spec
/// sentence: §04 "Standard modules" (`name`, `compat`), §04 "Module composition"
/// ("`source` may be a file path or a URL"), §03 "Records" (`get(r, "name1")`),
/// §07 "Operator-equivalent functions" (`equal`/`unequal` over strings), and the
/// field-name selector list the spec always writes as an array literal.
#[test]
fn documented_textual_parameters_still_accept_a_string() {
    assert_clean("pp = standard_module(\"particle-physics\", \"0.1\")\n");
    assert_clean("r = record(a = 1)\nx = get(r, \"a\")\n");
    assert_clean("b = equal(\"a\", \"b\")\n");
    assert_clean("s = [\"obs\", \"aux\"]\n");
}

// ---- §08 / §07 declared parameter value sets --------------------------------

/// §08 states each distribution parameter's set outright — `Normal`'s `sigma` is
/// "`elementof(posreals)`" — and §03 "Sets" fixes what `posreals` contains:
/// $(0, +\infty]$, which excludes `0`. §03 "Bool" makes `false` the value `0`, so
/// `Normal(sigma = false)` is decidable and wrong. It typed a normalized
/// `%measure` over `reals` at exit 0 before, and the determiniser then lowered a
/// density dividing by that zero.
#[test]
fn a_constant_outside_its_declared_value_set_is_refused() {
    for (src, set) in [
        ("d = Normal(mu = 0.0, sigma = false)\n", "posreals"),
        ("d = Normal(mu = 0.0, sigma = 0.0)\n", "posreals"),
        ("d = Normal(mu = 0.0, sigma = -1.0)\n", "posreals"),
        ("d = Poisson(rate = -1.0)\n", "nonnegreals"),
        ("d = Binomial(n = 0, p = 0.5)\n", "posintegers"),
        ("d = Bernoulli(p = 1.5)\n", "unitinterval"),
        ("d = Exponential(rate = 0)\n", "posreals"),
    ] {
        assert_refuses(src, &format!("outside its declared value set `{set}`"));
    }
}

/// The admitting side, and the reason the rule is a MEMBERSHIP TEST rather than a
/// type check. §03 closes the bounds it closes — `unitinterval` is $[0, 1]$ — so
/// `Bernoulli(p = true)` is `p = 1`, which is IN the set. The check runs on it and
/// admits it; it does not decline to look.
#[test]
fn a_constant_inside_its_declared_value_set_still_types() {
    assert_clean("d = Bernoulli(p = true)\n");
    assert_clean("d = Bernoulli(p = false)\n");
    assert_clean("d = Normal(mu = 0.0, sigma = 1.0)\n");
    assert_clean("d = Normal(mu = -3.0, sigma = 0.5)\n");
    assert_clean("d = Poisson(rate = 0.0)\n");
    assert_clean("d = Binomial(n = 1, p = 0.0)\n");
}

/// §03's infinity note is load-bearing: "`posreals`, `nonnegreals`, and `reals`
/// admit `inf`". So `sigma = inf` is admitted rather than refused as out of range —
/// the check answers on the spec's sets, not on a narrower numeric intuition.
#[test]
fn the_infinite_constant_is_admitted_where_the_spec_admits_it() {
    assert_clean("d = Normal(mu = 0.0, sigma = inf)\n");
    assert_clean("d = Poisson(rate = inf)\n");
}

/// A COMPUTED argument is not tested, and that is the rule's scope rather than a
/// gap. §03 says a deterministically computed node's value set is only "a
/// conservative superset of the values that `x` can take", so no engine can decide
/// it statically. The check answers on what it can see and stays silent otherwise.
#[test]
fn a_computed_argument_is_not_checked() {
    assert_clean("x = elementof(reals)\nd = Normal(mu = 0.0, sigma = x)\n");
    assert_clean("v = [1.0, 2.0]\nd = Normal(mu = 0.0, sigma = sum(v))\n");
}

/// A constant reached through a binding is checked the same way an inline literal
/// is — the walk resolves self-module references. This is what answers the old
/// `quantile` test's objection that a literal-only rule would be "a rule the
/// author cannot rely on".
#[test]
fn a_constant_reaches_the_check_through_a_binding() {
    assert_refuses(
        "s = 0.0\nd = Normal(mu = 0.0, sigma = s)\n",
        "outside its declared value set `posreals`",
    );
    assert_refuses(
        "q = 3.0\nv = [1.0, 2.0]\nr = quantile(v, q)\n",
        "outside its declared value set `interval(0, 1)`",
    );
}

/// The positional spelling reaches the same check as the keyword one: §04 binds an
/// argument to a parameter either way, and keyword normalization runs before the
/// domain checks so both see the same positions.
#[test]
fn the_positional_spelling_is_checked_too() {
    assert_refuses(
        "d = Normal(0.0, 0.0)\n",
        "outside its declared value set `posreals`",
    );
}

/// A parameter the spec states no scalar set for is untouched: `Uniform`'s
/// `support` is a SET argument, and every vector or matrix parameter carries a
/// per-element constraint this check does not reach.
#[test]
fn an_unconstrained_parameter_is_not_checked() {
    assert_clean("d = Uniform(interval(0.0, 1.0))\n");
    assert_clean("d = Dirichlet(alpha = [1.0, 1.0])\n");
}

// ---- §06: a measure operation takes a measure -------------------------------

/// §06 gives `normalize(M)` "a measure $M$ with finite total mass", `truncate(M,
/// S)` "the support of measure `M`", and `weighted(weight, base)` a `base` to
/// reweight. Before this, all three threaded a scalar argument straight through:
/// `normalize(1)` typed `%integer` and `truncate(false, interval(0, 1))` typed
/// `%boolean` carrying a unit-interval value set.
#[test]
fn a_value_where_a_measure_operation_wants_a_measure_is_refused() {
    for src in [
        "m = normalize(1)\n",
        "m = weighted(true, 2)\n",
        "m = truncate(false, interval(0, 1))\n",
    ] {
        assert_refuses(src, "must be a measure or a kernel");
    }
}

/// A real measure argument still types, and so does a `%deferred` one: §08's
/// composite rows have no catalogue entry, so `BinnedPoissonProcess(...)` is
/// honestly `%deferred`, and this check must not turn that into an error.
#[test]
fn measure_and_deferred_operands_still_type() {
    assert_clean("m = normalize(Normal(mu = 0.0, sigma = 1.0))\n");
    assert_clean("m = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0))\n");
    assert_clean("m = weighted(2.0, Normal(mu = 0.0, sigma = 1.0))\n");
    assert_clean("b = BinnedPoissonProcess([1.0, 2.0], [0.0, 1.0, 2.0])\nm = normalize(b)\n");
}
