//! The bi1 shape: `elementof(reals)` params + a `joint(...)` prior + a forward
//! model whose kernel params are DERIVED bindings (`a = f_a(theta2)`, `b =
//! f_b(theta1, theta2)`). `logdensityof(bayesupdate(likelihoodof(K, obs), prior),
//! θ)` must inline θ THROUGH those derived bindings so the emitted `lp` density is
//! self-contained w.r.t. θ — no `builtin_logdensityof` term is left reading an
//! `elementof(reals)` free param (which would dangle unbound at eval time: the
//! engine reports "no derivation for lp", a silent unevaluable density).
//!
//! Unlike bi2/bi3/bi4 (whose `~`-draw priors incidentally pin the shared derived
//! params globally), bi1's `prior = joint(theta1 = dist, theta2 = dist)` (a joint
//! of DISTRIBUTIONS) pins nothing — so the per-query θ-inlining must reach through
//! the θ-dependent derived bindings itself. See `density::substitute_refs_by_name`.

use flatppl_core::{CallHead, Inputs, Module, Node, NodeId, Ref, RefNs, Symbol};
use flatppl_determinizer::determinize;
use std::collections::HashSet;

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// True iff any node in the transitive dependency closure of `root` is an
/// `elementof(...)` call. The closure follows `Node::children()`, each `(%ref
/// self name)` edge into the referenced binding's RHS, and each reification
/// `%specinputs` boundary `(name, %ref self …)` source ref — the exact edges the
/// determiniser must inline through for the density to be self-contained w.r.t. θ.
/// Cycle-guarded by a visited-nodes set plus a visited-bindings set.
fn closure_reaches_elementof(m: &Module, root: NodeId) -> bool {
    let mut stack = vec![root];
    let mut seen_nodes: HashSet<NodeId> = HashSet::new();
    let mut seen_bindings: HashSet<Symbol> = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen_nodes.insert(id) {
            continue;
        }
        let node = m.node(id);
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = node
        {
            let name = *name;
            if seen_bindings.insert(name) {
                if let Some(bid) = m.binding_by_name(name) {
                    stack.push(m.binding(bid).rhs);
                }
            }
            continue;
        }
        if let Node::Call(c) = node {
            if let CallHead::Builtin(sym) = c.head {
                if m.resolve(sym) == "elementof" {
                    return true;
                }
            }
            if let Some(Inputs::Spec(entries)) = &c.inputs {
                for (_, r) in entries.iter() {
                    if r.ns == RefNs::SelfMod && seen_bindings.insert(r.name) {
                        if let Some(bid) = m.binding_by_name(r.name) {
                            stack.push(m.binding(bid).rhs);
                        }
                    }
                }
            }
        }
        for c in node.children() {
            stack.push(c);
        }
    }
    false
}

/// The RHS node of the (unique) binding named `name`.
fn binding_rhs(m: &Module, name: &str) -> NodeId {
    let (_, b) = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap_or_else(|| panic!("no binding named `{name}`"));
    b.rhs
}

/// bi1: `elementof(reals)` params, a `joint` prior over DISTRIBUTIONS (pins
/// nothing), and derived kernel params `a = f_a(theta2)`, `b = f_b(theta1,
/// theta2)`. This mirrors `fixtures/flatppl/bayesian_inference/bayesian_inference_1.flatppl`
/// as a single self-contained module (no `load_module`).
const BI1_POSTERIOR: &str = "\
flatppl_compat = \"0.1\"
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
theta1 = elementof(reals)
theta2 = elementof(reals)
c = 5
f_a = par -> c * par
f_b = fn(abs(_) * _)
a = f_a(theta2)
b = f_b(theta1, theta2)
obs ~ iid(Normal(mu = a, sigma = b), 10)
forward_kernel = kernelof(record(obs = obs))
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn bi1_posterior_lp_density_is_self_contained_wrt_theta() {
    // The bi1 posterior must lower (Ok), not refuse.
    let out = determinize(&parse_infer(BI1_POSTERIOR))
        .expect("bi1 posterior with derived kernel params must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);

    // SAME 12 terms as bi2/bi3/bi4: 10 obs-likelihood (iid(Normal, 10)) + 2 prior
    // (theta1 Normal, theta2 Exponential). The through-binding θ-inline changes the
    // SHAPE of the kernel-param record (a/b get inlined) but NOT the term count.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms; got:\n{pir}"
    );

    // The heart of the fix: the `lp` density subtree's transitive dependency
    // closure must contain NO `elementof(` node. Before the fix, the likelihood
    // terms read `record(mu = a, sigma = b)` where `a = f_a(theta2)` / `b =
    // f_b(theta1, theta2)` resolve through the unbound `theta1`/`theta2 =
    // elementof(reals)` params — so the closure reaches `elementof` and the density
    // is unevaluable. After the fix, θ is inlined through `a`/`b` (→ `f_a(1.0)` /
    // `f_b(0.5, 1.0)`), so the closure reaches only bound bindings (`f_a`, `f_b`,
    // `observed_data`) and the θ literals.
    assert!(
        !closure_reaches_elementof(&out, binding_rhs(&out, "lp")),
        "the lp density must be self-contained w.r.t. θ (no elementof reachable); got:\n{pir}"
    );

    // The measure/likelihood layer is fully eliminated.
    assert!(
        !pir.contains("likelihoodof") && !pir.contains("bayesupdate") && !pir.contains("(draw "),
        "measure/likelihood layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

/// Two-hop derived kernel param: `a1 = g(theta2)`, `a2 = h(a1)`, kernel mean `a2`
/// depends on θ only THROUGH `a1` (sigma is a plain constant). Regression-pins
/// that `substitute_refs_by_name` inlines through TWO intermediate derived
/// bindings, not just the one hop `BI1_POSTERIOR` exercises (`a = f_a(theta2)`
/// directly). Confirmed via scratch `flatppl determinize`: the emitted `lp` reads
/// `mu = h(g(1.0))` — both `a2` and its dependency `a1` inlined, `g`/`h` (θ-free)
/// left as shared refs.
const TWOHOP_POSTERIOR: &str = "\
flatppl_compat = \"0.1\"
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
theta1 = elementof(reals)
theta2 = elementof(reals)
c = 5
d = 2
g = par -> c * par
h = par2 -> d * par2
a1 = g(theta2)
a2 = h(a1)
obs ~ iid(Normal(mu = a2, sigma = 1.0), 10)
forward_kernel = kernelof(record(obs = obs))
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn twohop_posterior_lp_density_is_self_contained_wrt_theta() {
    // The two-hop posterior must lower (Ok), not refuse.
    let out = determinize(&parse_infer(TWOHOP_POSTERIOR))
        .expect("two-hop derived kernel param posterior must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);

    // Same 12 terms as BI1_POSTERIOR: 10 obs-likelihood + 2 prior.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms; got:\n{pir}"
    );

    // The heart of the fix, exercised through TWO hops: `a2`'s RHS references
    // `a1`, which references `theta2`. Both must inline so the `lp` closure never
    // reaches the unbound `elementof(reals)` param bindings.
    assert!(
        !closure_reaches_elementof(&out, binding_rhs(&out, "lp")),
        "the lp density must be self-contained w.r.t. theta through BOTH hops \
         (a2 -> a1 -> theta2); got:\n{pir}"
    );

    // The measure/likelihood layer is fully eliminated.
    assert!(
        !pir.contains("likelihoodof") && !pir.contains("bayesupdate") && !pir.contains("(draw "),
        "measure/likelihood layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

/// Diamond dependency: two kernel params (`p`, `q`) share a common θ-dependent
/// intermediate `cc = k(theta1)` (`p = f(cc)` feeds `mu`, `q = gg(cc)` feeds
/// `sigma`). Regression-pins that the memoized θ-inline of a shared derived
/// binding is applied CONSISTENTLY at both use sites (not inlined once and left
/// dangling at the other). Confirmed via scratch `flatppl determinize`: the
/// emitted `lp` reads `mu = f(k(0.5)), sigma = gg(k(0.5))` — `cc`'s θ-inlined copy
/// `k(0.5)` appears at both sites.
const DIAMOND_POSTERIOR: &str = "\
flatppl_compat = \"0.1\"
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
theta1 = elementof(reals)
theta2 = elementof(reals)
c = 5
d = 2
k = par -> c * par
f = par2 -> par2 + 1
gg = par3 -> par3 * d
cc = k(theta1)
p = f(cc)
q = gg(cc)
obs ~ iid(Normal(mu = p, sigma = q), 10)
forward_kernel = kernelof(record(obs = obs))
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(theta1 = 0.5, theta2 = 1.0))";

#[test]
fn diamond_posterior_lp_density_is_self_contained_wrt_theta() {
    // The diamond-dependency posterior must lower (Ok), not refuse.
    let out = determinize(&parse_infer(DIAMOND_POSTERIOR))
        .expect("diamond-dependency posterior must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);

    // Same 12 terms as BI1_POSTERIOR: 10 obs-likelihood + 2 prior.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        12,
        "expected 10 obs-likelihood + 2 prior terms; got:\n{pir}"
    );

    // The heart of the fix, exercised on a SHARED derived binding: `cc` is
    // referenced by both `p` (mu) and `q` (sigma), so its memoized θ-inlined copy
    // must be substituted at both use sites for the closure to be free of
    // `elementof`.
    assert!(
        !closure_reaches_elementof(&out, binding_rhs(&out, "lp")),
        "the lp density must be self-contained w.r.t. theta at BOTH diamond branches \
         (p and q via shared cc); got:\n{pir}"
    );

    // The measure/likelihood layer is fully eliminated.
    assert!(
        !pir.contains("likelihoodof") && !pir.contains("bayesupdate") && !pir.contains("(draw "),
        "measure/likelihood layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}
