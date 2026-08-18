//! Determiniser lowering of an APPLIED fan-out kernel — `joint(K1, K2, …)(a)`.
//!
//! §06 *Uniform kernel extension* defines the construct pointwise, so the lowering is the
//! rewrite `joint(K1, K2)(a) = joint(K1(a), K2(a))` handed to the measure-`joint` path.
//! Each component becomes `lawof(<its reified body>)`, which is §04 *Kernels and
//! `kernelof`* read literally ("`kernelof(x, kwargs...)` is equivalent to
//! `functionof(lawof(x), kwargs...)`") and which preserves node identity, so §06's
//! ancestry rule decides the answer rather than anything re-derived here.
//!
//! Every oracle below is closed-form and was cross-checked against Distributions.jl. The
//! engine's own output was never the target: the assertions read structure, and the
//! arithmetic is in the wave report.
use flatppl_determinizer::{determinize, determinize_with_roots};
use flatppl_infer::ModuleBundle;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Lower with `lp` as the only requested output, so DCE drops what nothing reaches and a
/// surviving parameter means the SCORED density still reads it.
fn lower(src: &str) -> String {
    let mut m = parse_infer(src);
    let lp = m.intern("lp");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[lp])).expect("must lower");
    flatppl_syntax::print(&out)
}

/// The `lp = …` binding of a lowered module, whitespace-collapsed so the multi-line
/// pretty-printing of a long `add(…)` does not enter the comparison.
fn lp_line(out: &str) -> String {
    let at = out
        .find("lp = ")
        .unwrap_or_else(|| panic!("no `lp` binding in:\n{out}"));
    out[at..].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The probe of `kernel-joint-q4-maths.md` §2: `u` is an internal latent both components
/// trace, `z` is the boundary both reify against.
const PROBE: &str = "\
z  = elementof(reals)
u  ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
K2 = kernelof(a2, z = z)
";

/// Q4 FORCED-RETAIN, the wave's headline value.
///
/// At `z = 0` the fan-out's law is `MvNormal([0,0], [[2,1],[1,2]])` — `Var(a_i) = Var(u|z)
/// + 1 = 2`, `Cov(a1,a2) = Var(u|z) = 1` — so the density at `(1,-1)` is
/// `-log(2*pi) - 0.5*log(3) - 1 = -3.3871832107434003`. COPY, the product of two
/// `Normal(0, sqrt 2)` marginals, would give `-3.0310242469692907`.
///
/// The discriminator is structural: RETAIN emits ONE closed-form record-law term carrying
/// the correlated covariance factor `log1p(2.0)`, COPY emits a SUM of per-component
/// `builtin_logdensityof` factors. The emitted term must also equal the unapplied
/// `lawof(record(p = a1, q = a2))` twin's — `kernel-joint-q4-maths.md` §3.1's commuting
/// identity, `joint(kernelof(a1, z=z), kernelof(a2, z=z)) = kernelof(record(…), z = z)`,
/// as an equality between two emissions rather than a number restated here.
#[test]
fn an_applied_keyword_fan_out_is_the_correlated_record_law() {
    let applied = lower(&format!(
        "{PROBE}\
         KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))"
    ));
    let twin = lower(
        "\
u  ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
lp = logdensityof(lawof(record(p = a1, q = a2)), record(p = 1.0, q = -1.0))",
    );
    assert!(
        applied.contains("log1p(2.0)"),
        "the correlated covariance factor must be emitted (RETAIN); got:\n{applied}"
    );
    assert!(
        !applied.contains("builtin_logdensityof"),
        "the record law is ONE term; per-component factors would mean COPY; got:\n{applied}"
    );
    assert_eq!(
        lp_line(&applied),
        lp_line(&twin),
        "the applied fan-out must equal its commuting-identity twin"
    );
}

/// The positional spelling reaches the same law. §06's keyword form "is equivalent to
/// `joint(relabel(M1, [\"name1\"]), relabel(M2, [\"name2\"]))`", so the two spellings differ
/// in the VARIATE's shape (a `cat` vector against a record) and in nothing else, and the
/// scored value is the same number.
#[test]
fn an_applied_positional_fan_out_is_the_same_record_law() {
    let positional = lower(&format!(
        "{PROBE}\
         KJ = joint(K1, K2)\n\
         lp = logdensityof(KJ(z = 0.0), [1.0, -1.0])"
    ));
    let keyword = lower(&format!(
        "{PROBE}\
         KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))"
    ));
    assert_eq!(
        lp_line(&positional),
        lp_line(&keyword),
        "both spellings score the same law"
    );
}

/// The applied input reaches the shared latent's own prior, so moving the application
/// point moves the law. At `z = 5`, `(1,-1)` the law is `MvNormal([5,5], [[2,1],[1,2]])`
/// and the quadratic form is `(1/3) * (2*16 - 2*24 + 2*36) = 56/3 = 18.666666666666664`;
/// the covariance factor is unchanged, since the boundary reaches the MEAN alone.
#[test]
fn an_applied_fan_out_moves_with_its_input() {
    let out = lower(&format!(
        "{PROBE}\
         KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ(z = 5.0), record(p = 1.0, q = -1.0))"
    ));
    assert!(
        out.contains("18.666666666666664") && out.contains("log1p(2.0)"),
        "the quadratic form moves and the covariance factor does not; got:\n{out}"
    );
    assert!(
        !out.contains("elementof"),
        "no parameter survives; got:\n{out}"
    );
}

/// Trace-DISJOINT components are independent — §06: "Components that share no stochastic
/// node are independent, and their `joint` is the product measure" — so the fan-out lowers
/// to the per-component product, each factor at its own pinned input. The union signature
/// is `{z, w}` (Q1), and the oracle at `(1,-1)` with both inputs `0` is
/// `-log(2*pi) - 1 = -2.8378770664093453`.
#[test]
fn a_trace_disjoint_applied_fan_out_is_the_per_component_product() {
    let out = lower(
        "\
z  = elementof(reals)
w  = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
b2 ~ Normal(mu = w, sigma = 1.0)
K1 = kernelof(b1, z = z)
K2 = kernelof(b2, w = w)
KJ = joint(p = K1, q = K2)
lp = logdensityof(KJ(z = 0.0, w = 0.0), record(p = 1.0, q = -1.0))",
    );
    assert_eq!(
        out.matches("builtin_logdensityof").count(),
        2,
        "disjoint components give a product of two factors; got:\n{out}"
    );
    assert_eq!(
        out.matches("record(mu = 0.0, sigma = 1.0)").count(),
        2,
        "each factor reads its own pinned input; got:\n{out}"
    );
    assert!(!out.contains("elementof"), "got:\n{out}");
}

/// `joint(K, K)` is the singular diagonal at every input (§06 *Singular joints*: "the same
/// draw referenced twice […] has no density w.r.t. the product reference measure"), so the
/// density query refuses.
///
/// This is the check that makes the `lawof(<body>)` coordinate load-bearing rather than
/// cosmetic. Reducing each component to its LAW instead would synthesize a fresh draw per
/// component, and the two references to one draw would score as a product — a density for
/// a shape §06 gives none.
#[test]
fn a_singular_applied_fan_out_refuses() {
    let src = "\
z  = elementof(reals)
u  ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = K1)
lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 1.0))";
    let err = determinize(&parse_infer(src)).expect_err("a singular joint has no density");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("same draw") || msg.contains("fewer distinct draws"),
        "the refusal must name the rank deficiency; got: {msg}"
    );
}

/// Q3: "Measure components are permitted and are the nullary case: they ignore the input."
/// The measure component shares no stochastic node, so the fan-out is the product of the
/// pinned kernel component and the constant measure. Oracle at `(1,-1)`:
/// `log N(1; 0,1) + log N(-1; 0,2) = -1.4189385332046727 + -1.737085713764618 =
/// -3.1560242469692907`.
#[test]
fn a_measure_component_of_an_applied_fan_out_is_the_nullary_case() {
    let out = lower(
        "\
z  = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
K1 = kernelof(b1, z = z)
KJ = joint(p = K1, q = Normal(mu = 0.0, sigma = 2.0))
lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = -1.0))",
    );
    assert!(
        out.contains("record(mu = 0.0, sigma = 1.0)")
            && out.contains("record(mu = 0.0, sigma = 2.0)"),
        "the kernel component is pinned and the measure component is untouched; got:\n{out}"
    );
    assert!(!out.contains("elementof"), "got:\n{out}");
}

/// An UNAPPLIED kernel-`joint` density query stays refused: §06 makes `logdensityof`
/// require "closed measures (i.e. nullary kernels)", and a fan-out with a live input is
/// not one.
#[test]
fn an_unapplied_kernel_joint_density_query_stays_refused() {
    let src = format!(
        "{PROBE}\
         KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ, record(p = 1.0, q = -1.0))"
    );
    determinize(&parse_infer(&src)).expect_err("an unapplied fan-out is not a closed measure");
}

/// The fan-out path gets the query-point guard too.
///
/// The finish rewrites the whole emitted density, so a point written as the ambient `z`
/// would be scored at the applied value. Without the guard on this path,
/// `record(p = z, q = -1.0)` scored the `p` coordinate at `0.0` and exited 0 — a silent
/// wrong number on a shape `a86437d` refused. The two applied paths must carry the same
/// guard; only the reification one did.
#[test]
fn an_applied_fan_out_refuses_a_query_point_naming_the_boundary() {
    let src = format!(
        "{PROBE}\
         KJ = joint(p = K1, q = K2)\n\
         lp = logdensityof(KJ(z = 0.0), record(p = z, q = -1.0))"
    );
    let err = determinize(&parse_infer(&src)).expect_err("the point and the boundary collide");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("boundary input"),
        "the refusal must name the boundary collision; got: {msg}"
    );
}

/// §04's splat binds a sole positional `table(...)` by COLUMN name exactly as it binds a
/// `record(...)` by field name: "`f(record(a = x, b = y, ...))` and `f(table(a = x, b = y,
/// ...))` are equivalent to `f(a = x, b = y, ...)`".
///
/// Recognising only `record` here sent the table down the positional arm, which bound the
/// whole `table(z = 0.0)` node to every input and emitted
/// `record(mu = table(z = 0.0), sigma = 1.0)` at exit 0. The two components are
/// independent given the pinned input, so the oracle at `(1,-1)` is
/// `-log(2*pi) - 1 = -2.8378770664093453`.
#[test]
fn an_applied_fan_out_splats_a_sole_positional_table() {
    let out = lower(
        "\
z  = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
b2 ~ Normal(mu = z, sigma = 1.0)
K1 = kernelof(b1, z = z)
K2 = kernelof(b2, z = z)
KJ = joint(p = K1, q = K2)
lp = logdensityof(KJ(table(z = 0.0)), record(p = 1.0, q = -1.0))",
    );
    assert!(
        !out.contains("table("),
        "the table must be splatted by column name, not bound whole; got:\n{out}"
    );
    assert_eq!(
        out.matches("record(mu = 0.0, sigma = 1.0)").count(),
        2,
        "both factors read the splatted column; got:\n{out}"
    );
}

/// The W1 shape refuses in the DETERMINISER too, not only at the inference front door.
///
/// `kernel-joint-w1-maths.md` §3: with `u` shared and its ancestor `z` bound by `K1`
/// alone, the single node `u` would need `Normal(a,1)` and `Normal(v,1)` at once, so the
/// shape denotes nothing and §9 requires the determiniser to refuse it. `determinize`
/// ignores inference diagnostics, so the static error is not the backstop — this is, and
/// it must never emit reading A's number (`-2.0878770664093453`, the log pdf of
/// `MvNormal([0,0],[[2,1],[1,1]])` at `(1,0.5)`), which is the answer to a rewritten
/// question.
#[test]
fn the_w1_shape_refuses_in_the_determiniser_too() {
    let src = "\
z  = elementof(reals)
u  ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
M  = lawof(u)
KJ = joint(p = K1, q = M)
lp = logdensityof(KJ(z = 0.0), record(p = 1.0, q = 0.5))";
    let err = determinize(&parse_infer(src)).expect_err("the W1 shape has no law");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("under no name") || msg.contains("not a reification"),
        "the refusal must name the ancestry clause; got: {msg}"
    );
}
