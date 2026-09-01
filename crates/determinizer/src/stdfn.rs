//! Lower a §09 standard-module FUNCTION member call in a VALUE position into a
//! base-op subtree.
//!
//! Spec §09 "Standard modules" gives each of these members a closed form over the
//! `base` module's own deterministic operations (§07 "Built-in functions"). A
//! member call parses as a USER-headed call whose callee is a
//! `(%ref <alias> member)`, and FlatPDL admits no residual user call
//! (`conformance::is_flatpdl`), so without this pass every one of them refuses
//! with the generic `ResidualUserCall`.
//!
//! This is the FUNCTION counterpart to the constructor-tag path: a §09
//! DISTRIBUTION member keeps its module-qualified call and becomes a kernel tag in
//! `density::split_kernel_constructor`. A function member is not a measure, so it
//! must be inlined here instead.
//!
//! ## Why the pass runs before the measure-reduction loop
//!
//! A member call feeds an ordinary value — typically a distribution parameter
//! (`Normal(mu = hep.interp_pwlin(...), ...)`). Inlining it before the loop leaves
//! the loop and the density lowering looking at base ops only, so no downstream
//! path needs to know §09 exists.
//!
//! ## Refuse-don't-mislower
//!
//! A member with no base-op form (`special-functions`, `ext-linear-algebra`, the
//! Wigner functions, the higher-order `distances` members) is left ALONE here, not
//! refused: a call in a binding that root-based DCE drops must keep being dropped
//! rather than refuse. `conformance::is_flatpdl` names the member in its refusal
//! for the calls that do survive.
//!
//! A member this pass does implement, but whose call shape it cannot read (named
//! arguments, wrong arity, a non-literal degree), is likewise left alone.

use std::collections::HashMap;

use flatppl_core::{CallHead, Module, Node, NodeId, Ref, RefNs, Scalar};

use crate::density::{build_call, resolve_ref_one};
use crate::refuse::RefuseError;

/// Largest `polynomials` degree this pass will unroll. The lowering emits the
/// three-term recursion node by node, so the emitted subtree grows linearly in
/// the degree; a runaway literal would otherwise inflate the module without
/// bound. Well past any degree a model realistically writes.
const MAX_POLY_DEGREE: i64 = 128;

/// Barrier polynomials $\chi_\ell$ for [`blatt_weisskopf`], ascending in $z$.
///
/// §09 spells $\chi_0..\chi_3$ and says the family continues through $\ell = 7$,
/// which is the range it declares the function defined over. The remaining four
/// come from the identity $\chi_\ell(x^2) = x^{2\ell+2}(j_\ell(x)^2 +
/// y_\ell(x)^2)$ over the spherical Bessel functions — the maths behind the
/// Blatt-Weisskopf factor, not another engine's table. The same identity
/// reproduces §09's own four exactly. Every entry is an integer below $2^{53}$,
/// so none is rounded.
const BARRIER_CHI: [&[f64]; 8] = [
    &[1.0],
    &[1.0, 1.0],
    &[9.0, 3.0, 1.0],
    &[225.0, 45.0, 6.0, 1.0],
    &[11025.0, 1575.0, 135.0, 10.0, 1.0],
    &[893025.0, 99225.0, 6300.0, 315.0, 15.0, 1.0],
    &[108056025.0, 9823275.0, 496125.0, 18900.0, 630.0, 21.0, 1.0],
    &[
        18261468225.0,
        1404728325.0,
        58939650.0,
        1819125.0,
        47250.0,
        1134.0,
        28.0,
        1.0,
    ],
];

/// Rewrite every lowerable §09 function-member call in `m` into base ops.
///
/// Iterates to a fixed point: a replacement subtree carries the original call's
/// ARGUMENTS over unchanged, so a member call nested inside another member call's
/// argument (`hep.kallen(hep.kallen(x, y, z), y, z)`) surfaces as a top-level
/// target on the next pass. Each pass replaces at least one call and introduces
/// none, so the target count strictly decreases and the loop terminates; the bound
/// is a guard, not the termination argument.
pub(crate) fn lower_std_module_functions(m: &mut Module) -> Result<(), RefuseError> {
    for _ in 0..MAX_PASSES {
        let targets = collect_member_calls(m);
        let mut replacements: HashMap<NodeId, NodeId> = HashMap::new();
        for (id, module, member) in targets {
            if let Some(new) = lower_member_call(m, id, &module, &member) {
                replacements.insert(id, new);
            }
        }
        if replacements.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(flatppl_core::BindingId, NodeId)> =
            m.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
        for (bid, root) in pairs {
            let new =
                crate::driver::map_tree(m, root, &mut |_m, id| replacements.get(&id).copied());
            if new != root {
                m.set_binding_rhs(bid, new);
            }
        }
    }
    Ok(())
}

/// Pass bound for [`lower_std_module_functions`]. One pass per nesting level of
/// member-call-inside-member-call-argument; no realistic model nests deeply.
const MAX_PASSES: usize = 64;

/// Every `(node, module-name, member-name)` for a call whose head is a §09
/// standard-module member, anywhere in a binding RHS. One `(%ref self …)` hop is
/// followed on the callee, so `f = hep.kallen; f(x, y, z)` is found too — the same
/// hop `density::std_module_ctor_sym` follows for a distribution member.
fn collect_member_calls(m: &Module) -> Vec<(NodeId, String, String)> {
    fn walk(
        m: &Module,
        id: NodeId,
        seen: &mut Vec<NodeId>,
        out: &mut Vec<(NodeId, String, String)>,
    ) {
        if seen.contains(&id) {
            return;
        }
        seen.push(id);
        if let Node::Call(c) = m.node(id) {
            if let CallHead::User(callee) = c.head {
                if let Some(hit) = member_of_callee(m, callee) {
                    out.push((id, hit.0, hit.1));
                }
            }
        }
        for child in m.node(id).children() {
            walk(m, child, seen, out);
        }
    }
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for (_bid, b) in m.bindings() {
        walk(m, b.rhs, &mut seen, &mut out);
    }
    out
}

/// The `(module, member)` a call callee names, directly or through one
/// `(%ref self …)` hop.
fn member_of_callee(m: &Module, callee: NodeId) -> Option<(String, String)> {
    crate::crossmodule::std_module_member(m, callee).or_else(|| {
        let hop = resolve_ref_one(m, callee).0;
        crate::crossmodule::std_module_member(m, hop)
    })
}

/// Build the base-op replacement for the member call at `id`, or `None` to leave
/// the call alone (no base-op form for this member, or a call shape this pass
/// cannot read: named arguments, wrong arity, or a non-literal integer degree).
fn lower_member_call(m: &mut Module, id: NodeId, module: &str, member: &str) -> Option<NodeId> {
    let args: Vec<NodeId> = match m.node(id) {
        Node::Call(c) if c.named.is_empty() => c.args.to_vec(),
        _ => return None,
    };
    match (module, member) {
        ("particle-physics", "interp_pwlin") => arity::<4>(&args).map(|a| interp_pwlin(m, a)),
        ("particle-physics", "interp_pwexp") => arity::<4>(&args).map(|a| interp_pwexp(m, a)),
        ("particle-physics", "interp_poly2_lin") => {
            arity::<4>(&args).map(|a| interp_poly2_lin(m, a))
        }
        ("particle-physics", "interp_poly6_lin") => {
            arity::<4>(&args).map(|a| interp_poly6_lin(m, a))
        }
        ("particle-physics", "interp_poly6_exp") => {
            arity::<4>(&args).map(|a| interp_poly6_exp(m, a))
        }
        ("particle-physics", "kallen") => arity::<3>(&args).map(|a| kallen(m, a)),
        ("particle-physics", "breakup_momentum") => {
            arity::<3>(&args).map(|a| breakup_momentum(m, a))
        }
        ("particle-physics", "blatt_weisskopf") => {
            let a = arity::<3>(&args)?;
            blatt_weisskopf(m, a)
        }
        ("polynomials", name @ ("legendre" | "hermite" | "laguerre" | "chebyshev")) => {
            let [n, x] = arity::<2>(&args)?;
            let degree = literal_degree(m, n)?;
            orthogonal_polynomial(m, name, degree, x)
        }
        ("distances", "euclidean") => arity::<2>(&args).map(|[u, v]| {
            let d = sub(m, u, v);
            build_call(m, "l2norm", &[d])
        }),
        ("distances", "squared_euclidean") => arity::<2>(&args).map(|[u, v]| {
            let d = sub(m, u, v);
            let dt = build_call(m, "transpose", &[d]);
            mul(m, dt, d)
        }),
        ("distances", "manhattan") => arity::<2>(&args).map(|[u, v]| {
            let d = sub(m, u, v);
            build_call(m, "l1norm", &[d])
        }),
        ("distances", "chebyshev") => arity::<2>(&args).map(|[u, v]| {
            let d = sub(m, u, v);
            build_call(m, "linfnorm", &[d])
        }),
        ("distances", "cosine") => arity::<2>(&args).map(|[u, v]| cosine(m, u, v)),
        _ => None,
    }
}

/// The positional arguments as a fixed-size array, or `None` on an arity
/// mismatch — a call shape this pass declines rather than mislowers.
fn arity<const N: usize>(args: &[NodeId]) -> Option<[NodeId; N]> {
    args.try_into().ok()
}

/// A non-negative integer degree written as a literal, directly or through one
/// `(%ref self …)` hop. §09 requires a non-negative integer; a computed degree is
/// declined (this pass runs before constant folding, so `1 + 1` is not a literal
/// here).
fn literal_degree(m: &Module, id: NodeId) -> Option<i64> {
    let node = resolve_ref_one(m, id).0;
    let n = match m.node(node) {
        Node::Lit(Scalar::Int(n)) => *n,
        _ => return None,
    };
    (0..=MAX_POLY_DEGREE).contains(&n).then_some(n)
}

// -- node builders ---------------------------------------------------------

/// A real literal, spelled `neg(<magnitude>)` when negative.
///
/// §11 "Literal values": a scalar literal carries no leading sign, and a negated
/// literal is the call `(neg <magnitude>)`. The FlatPIR reader enforces it, so a
/// signed `Node::Lit` here makes the emitted module unreadable — the same rule
/// `canon::fold` follows at its three fold sites. Every negative constant in this
/// module goes through here, so the rule holds by construction.
fn lit(m: &mut Module, v: f64) -> NodeId {
    if v < 0.0 {
        let magnitude = m.alloc(Node::Lit(Scalar::Real(-v)));
        return neg(m, magnitude);
    }
    m.alloc(Node::Lit(Scalar::Real(v)))
}
fn add(m: &mut Module, a: NodeId, b: NodeId) -> NodeId {
    build_call(m, "add", &[a, b])
}
fn sub(m: &mut Module, a: NodeId, b: NodeId) -> NodeId {
    build_call(m, "sub", &[a, b])
}
fn mul(m: &mut Module, a: NodeId, b: NodeId) -> NodeId {
    build_call(m, "mul", &[a, b])
}
fn divide(m: &mut Module, a: NodeId, b: NodeId) -> NodeId {
    build_call(m, "divide", &[a, b])
}
fn neg(m: &mut Module, a: NodeId) -> NodeId {
    build_call(m, "neg", &[a])
}
fn exp(m: &mut Module, a: NodeId) -> NodeId {
    build_call(m, "exp", &[a])
}
fn log(m: &mut Module, a: NodeId) -> NodeId {
    build_call(m, "log", &[a])
}
fn sqrt(m: &mut Module, a: NodeId) -> NodeId {
    build_call(m, "sqrt", &[a])
}

/// `x * scale` with `scale` a compile-time constant.
fn scale(m: &mut Module, x: NodeId, k: f64) -> NodeId {
    let c = lit(m, k);
    mul(m, x, c)
}

/// `ifelse(gt(a, 1), up, ifelse(lt(a, -1), dn, mid))` — the shared shape of every
/// §09 three-point interpolator that switches at $\alpha = \pm 1$.
fn split_at_unit(m: &mut Module, alpha: NodeId, up: NodeId, dn: NodeId, mid: NodeId) -> NodeId {
    let one = lit(m, 1.0);
    let minus_one = lit(m, -1.0);
    let above = build_call(m, "gt", &[alpha, one]);
    let below = build_call(m, "lt", &[alpha, minus_one]);
    let inner = build_call(m, "ifelse", &[below, dn, mid]);
    build_call(m, "ifelse", &[above, up, inner])
}

// -- particle-physics: three-point interpolation ---------------------------

/// §09 `interp_pwlin`: `center + alpha*(right - center)` for `alpha >= 0`,
/// `center + alpha*(center - left)` otherwise.
fn interp_pwlin(m: &mut Module, [left, center, right, alpha]: [NodeId; 4]) -> NodeId {
    let up_slope = sub(m, right, center);
    let dn_slope = sub(m, center, left);
    let up_step = mul(m, alpha, up_slope);
    let dn_step = mul(m, alpha, dn_slope);
    let up = add(m, center, up_step);
    let dn = add(m, center, dn_step);
    let zero = lit(m, 0.0);
    let nonneg = build_call(m, "ge", &[alpha, zero]);
    build_call(m, "ifelse", &[nonneg, up, dn])
}

/// §09 `interp_pwexp`: `exp(interp_pwlin(log left, log center, log right, alpha))`.
/// §09 requires strictly positive anchors; a non-positive one reaches `log` and
/// the engine's own domain handling, exactly as a hand-written `log` would.
fn interp_pwexp(m: &mut Module, [left, center, right, alpha]: [NodeId; 4]) -> NodeId {
    let ll = log(m, left);
    let lc = log(m, center);
    let lr = log(m, right);
    let inner = interp_pwlin(m, [ll, lc, lr, alpha]);
    exp(m, inner)
}

/// §09 `interp_poly2_lin`: quadratic `center + S*alpha + A*alpha^2` inside
/// `[-1, 1]` with `S = (right - left)/2`, `A = (right + left)/2 - center`;
/// outside, the line continuing with slope `S + 2A` (right) or `S - 2A` (left)
/// from the quadratic's own endpoint value.
fn interp_poly2_lin(m: &mut Module, [left, center, right, alpha]: [NodeId; 4]) -> NodeId {
    let diff = sub(m, right, left);
    let s = scale(m, diff, 0.5);
    let total = add(m, right, left);
    let half = scale(m, total, 0.5);
    let a = sub(m, half, center);

    let alpha2 = mul(m, alpha, alpha);
    let s_term = mul(m, s, alpha);
    let a_term = mul(m, a, alpha2);
    let mid_part = add(m, s_term, a_term);
    let mid = add(m, center, mid_part);

    // f(+1) = center + S + A, continuing with slope S + 2A.
    let two_a = scale(m, a, 2.0);
    let up_slope = add(m, s, two_a);
    let up_end = add(m, s, a);
    let up_base = add(m, center, up_end);
    let one = lit(m, 1.0);
    let up_shift = sub(m, alpha, one);
    let up_step = mul(m, up_shift, up_slope);
    let up = add(m, up_base, up_step);

    // f(-1) = center - S + A, continuing with slope S - 2A.
    let dn_slope = sub(m, s, two_a);
    let dn_end = sub(m, a, s);
    let dn_base = add(m, center, dn_end);
    let dn_shift = add(m, alpha, one);
    let dn_step = mul(m, dn_shift, dn_slope);
    let dn = add(m, dn_base, dn_step);

    split_at_unit(m, alpha, up, dn, mid)
}

/// §09 `interp_poly6_lin`: with `Sup = right - center`, `Sdn = center - left`,
/// `S = (Sup + Sdn)/2` and `A = (Sup - Sdn)/16`, the degree-6 polynomial
/// `center + alpha*(S + alpha*A*(15 - 10*alpha^2 + 3*alpha^4))` inside `[-1, 1]`
/// and the lines `center + alpha*Sup` / `center + alpha*Sdn` outside.
///
/// Those coefficients are the unique solution of §09's six conditions — `f(±1)`,
/// `f'(±1)` and `f''(±1)` matched to the linear extrapolation with `f(0) =
/// center` — checked against an independent 6×6 linear solve, not against another
/// engine.
fn interp_poly6_lin(m: &mut Module, [left, center, right, alpha]: [NodeId; 4]) -> NodeId {
    let sup = sub(m, right, center);
    let sdn = sub(m, center, left);
    let sum = add(m, sup, sdn);
    let s = scale(m, sum, 0.5);
    let diff = sub(m, sup, sdn);
    let a = scale(m, diff, 1.0 / 16.0);

    let alpha2 = mul(m, alpha, alpha);
    // 15 - 10*alpha^2 + 3*alpha^4, by Horner in alpha^2.
    let three = lit(m, 3.0);
    let minus_ten = lit(m, -10.0);
    let fifteen = lit(m, 15.0);
    let t = mul(m, alpha2, three);
    let t = add(m, minus_ten, t);
    let t = mul(m, alpha2, t);
    let bracket = add(m, fifteen, t);

    let a_alpha = mul(m, a, alpha);
    let quad = mul(m, a_alpha, bracket);
    let inner = add(m, s, quad);
    let mid_part = mul(m, alpha, inner);
    let mid = add(m, center, mid_part);

    let up_step = mul(m, alpha, sup);
    let up = add(m, center, up_step);
    let dn_step = mul(m, alpha, sdn);
    let dn = add(m, center, dn_step);

    split_at_unit(m, alpha, up, dn, mid)
}

/// §09 `interp_poly6_exp`: degree-6 polynomial inside `[-1, 1]`, exponential
/// extrapolation `f(±1)*exp((alpha ∓ 1)*f'(±1)/f(±1))` outside.
///
/// §09 fixes the extrapolation from `f(±1)` and `f'(±1)`, which by itself leaves
/// the polynomial underdetermined: with `f(0) = center` and the §09 anchors
/// `f(-1) = left`, `f(+1) = right` there are five conditions for six
/// coefficients. The remaining degree of freedom is fixed by taking the
/// extrapolation to be the exponential through the anchors,
/// `center*(right/center)^alpha` for `alpha > 1` and
/// `center*(left/center)^(-alpha)` for `alpha < -1` — which reproduces §09's own
/// formula (`f'(1)/f(1) = log(right/center)`) and makes the six conditions
/// linear. Checked against an independent 6×6 linear solve of those conditions.
fn interp_poly6_exp(m: &mut Module, [left, center, right, alpha]: [NodeId; 4]) -> NodeId {
    let r_hi = divide(m, right, center);
    let r_lo = divide(m, left, center);
    let log_hi = log(m, r_hi);
    let log_lo = log(m, r_lo);

    // Value, first and second derivative of each extrapolation at its boundary,
    // as ratios to `center`. The sign on the low side follows d/dalpha of
    // (left/center)^(-alpha).
    let up_1 = mul(m, r_hi, log_hi);
    let dn_1_pos = mul(m, r_lo, log_lo);
    let dn_1 = neg(m, dn_1_pos);
    let up_2 = mul(m, up_1, log_hi);
    let dn_2_neg = mul(m, dn_1, log_lo);
    let dn_2 = neg(m, dn_2_neg);

    let sym = |m: &mut Module, hi: NodeId, lo: NodeId| {
        let t = add(m, hi, lo);
        scale(m, t, 0.5)
    };
    let anti = |m: &mut Module, hi: NodeId, lo: NodeId| {
        let t = sub(m, hi, lo);
        scale(m, t, 0.5)
    };
    let s0 = sym(m, r_hi, r_lo);
    let a0 = anti(m, r_hi, r_lo);
    let s1 = sym(m, up_1, dn_1);
    let a1 = anti(m, up_1, dn_1);
    let s2 = sym(m, up_2, dn_2);
    let a2 = anti(m, up_2, dn_2);

    // The six coefficients of `mod(alpha) - 1`, ascending in alpha.
    let coeffs = [
        weighted_sum(m, &[(15.0, a0), (-7.0, s1), (1.0, a2)], 0.0, 1.0 / 8.0),
        weighted_sum(m, &[(24.0, s0), (-9.0, a1), (1.0, s2)], -24.0, 1.0 / 8.0),
        weighted_sum(m, &[(-5.0, a0), (5.0, s1), (-1.0, a2)], 0.0, 0.25),
        weighted_sum(m, &[(-12.0, s0), (7.0, a1), (-1.0, s2)], 12.0, 0.25),
        weighted_sum(m, &[(3.0, a0), (-3.0, s1), (1.0, a2)], 0.0, 1.0 / 8.0),
        weighted_sum(m, &[(8.0, s0), (-5.0, a1), (1.0, s2)], -8.0, 1.0 / 8.0),
    ];
    // Horner from the top coefficient down, then the leading 1.
    let mut acc = coeffs[5];
    for &c in coeffs[..5].iter().rev() {
        let t = mul(m, alpha, acc);
        acc = add(m, c, t);
    }
    let t = mul(m, alpha, acc);
    let one = lit(m, 1.0);
    let mod_val = add(m, one, t);
    let mid = mul(m, center, mod_val);

    let up_arg = mul(m, alpha, log_hi);
    let up_factor = exp(m, up_arg);
    let up = mul(m, center, up_factor);
    let neg_alpha = neg(m, alpha);
    let dn_arg = mul(m, neg_alpha, log_lo);
    let dn_factor = exp(m, dn_arg);
    let dn = mul(m, center, dn_factor);

    split_at_unit(m, alpha, up, dn, mid)
}

/// `factor * (constant + sum_i k_i * term_i)`.
fn weighted_sum(m: &mut Module, terms: &[(f64, NodeId)], constant: f64, factor: f64) -> NodeId {
    let mut acc = lit(m, constant);
    for &(k, term) in terms {
        let scaled = scale(m, term, k);
        acc = add(m, acc, scaled);
    }
    scale(m, acc, factor)
}

// -- particle-physics: kinematics ------------------------------------------

/// §09 `kallen`: $\lambda(x, y, z) = x^2 + y^2 + z^2 - 2xy - 2yz - 2zx$.
fn kallen(m: &mut Module, [x, y, z]: [NodeId; 3]) -> NodeId {
    let x2 = mul(m, x, x);
    let y2 = mul(m, y, y);
    let z2 = mul(m, z, z);
    let squares = add(m, x2, y2);
    let squares = add(m, squares, z2);
    let xy = mul(m, x, y);
    let yz = mul(m, y, z);
    let zx = mul(m, z, x);
    let cross = add(m, xy, yz);
    let cross = add(m, cross, zx);
    let cross2 = scale(m, cross, 2.0);
    sub(m, squares, cross2)
}

/// §09 `breakup_momentum`:
/// $p = \sqrt{(m - (m_a + m_b))(m + (m_a + m_b))}\sqrt{(m - (m_a - m_b))(m + (m_a - m_b))} / (2m)$.
/// §09 gives this factored form alongside $\sqrt{\lambda(m^2, m_a^2, m_b^2)}/(2m)$;
/// the factored one is emitted because it avoids the catastrophic cancellation the
/// Källén form suffers near threshold.
fn breakup_momentum(m: &mut Module, [mass, ma, mb]: [NodeId; 3]) -> NodeId {
    let s_plus = add(m, ma, mb);
    let s_minus = sub(m, ma, mb);
    let lo_p = sub(m, mass, s_plus);
    let hi_p = add(m, mass, s_plus);
    let prod_p = mul(m, lo_p, hi_p);
    let root_p = sqrt(m, prod_p);
    let lo_m = sub(m, mass, s_minus);
    let hi_m = add(m, mass, s_minus);
    let prod_m = mul(m, lo_m, hi_m);
    let root_m = sqrt(m, prod_m);
    let numer = mul(m, root_p, root_m);
    let denom = scale(m, mass, 2.0);
    divide(m, numer, denom)
}

/// §09 `blatt_weisskopf`: with $z = (dp)^2$, $F_\ell = \sqrt{z^\ell/\chi_\ell(z)}$.
///
/// Only a LITERAL $\ell$ lowers: the barrier polynomial is selected by $\ell$, and
/// FlatPPL has no control flow to select one at run time. §09 defines the function
/// for $0 \leq \ell \leq 7$; an $\ell$ outside that range is declined, as is a
/// computed one.
fn blatt_weisskopf(m: &mut Module, [l, p, d]: [NodeId; 3]) -> Option<NodeId> {
    let node = resolve_ref_one(m, l).0;
    let ell = match m.node(node) {
        Node::Lit(Scalar::Int(n)) => *n,
        _ => return None,
    };
    let chi_coeffs = *BARRIER_CHI.get(usize::try_from(ell).ok()?)?;

    let dp = mul(m, d, p);
    let z = mul(m, dp, dp);
    // chi(z) by Horner, descending.
    let mut chi = lit(m, *chi_coeffs.last().unwrap());
    for &c in chi_coeffs[..chi_coeffs.len() - 1].iter().rev() {
        let t = mul(m, z, chi);
        let c = lit(m, c);
        chi = add(m, c, t);
    }
    let mut z_pow = lit(m, 1.0);
    for _ in 0..ell {
        z_pow = mul(m, z_pow, z);
    }
    let ratio = divide(m, z_pow, chi);
    Some(sqrt(m, ratio))
}

// -- polynomials -----------------------------------------------------------

/// §09 `polynomials` members at a LITERAL degree, emitted as the unrolled
/// three-term recursion rather than precomputed coefficients: the recursions
/// carry only small integer literals, so no rational coefficient has to be
/// rounded into the emitted module.
///
/// * `legendre`: $P_0 = 1$, $P_1 = x$, $(k+1)P_{k+1} = (2k+1)xP_k - kP_{k-1}$.
/// * `hermite` (physicist's): $H_0 = 1$, $H_1 = 2x$, $H_{k+1} = 2xH_k - 2kH_{k-1}$.
/// * `laguerre`: $L_0 = 1$, $L_1 = 1 - x$, $(k+1)L_{k+1} = (2k+1-x)L_k - kL_{k-1}$.
/// * `chebyshev` (first kind): $T_0 = 1$, $T_1 = x$, $T_{k+1} = 2xT_k - T_{k-1}$.
fn orthogonal_polynomial(m: &mut Module, name: &str, degree: i64, x: NodeId) -> Option<NodeId> {
    let mut prev = lit(m, 1.0);
    if degree == 0 {
        return Some(prev);
    }
    let mut cur = match name {
        "legendre" | "chebyshev" => x,
        "hermite" => scale(m, x, 2.0),
        "laguerre" => {
            let one = lit(m, 1.0);
            sub(m, one, x)
        }
        _ => return None,
    };
    for k in 1..degree {
        let kf = k as f64;
        let next = match name {
            "legendre" => {
                let a = scale(m, x, 2.0 * kf + 1.0);
                let a = mul(m, a, cur);
                let b = scale(m, prev, kf);
                let t = sub(m, a, b);
                let denom = lit(m, kf + 1.0);
                divide(m, t, denom)
            }
            "hermite" => {
                let a = scale(m, x, 2.0);
                let a = mul(m, a, cur);
                let b = scale(m, prev, 2.0 * kf);
                sub(m, a, b)
            }
            "laguerre" => {
                let lead = lit(m, 2.0 * kf + 1.0);
                let lead = sub(m, lead, x);
                let a = mul(m, lead, cur);
                let b = scale(m, prev, kf);
                let t = sub(m, a, b);
                let denom = lit(m, kf + 1.0);
                divide(m, t, denom)
            }
            "chebyshev" => {
                let a = scale(m, x, 2.0);
                let a = mul(m, a, cur);
                sub(m, a, prev)
            }
            _ => return None,
        };
        prev = cur;
        cur = next;
    }
    Some(cur)
}

// -- distances -------------------------------------------------------------

/// §09 `cosine`: $1 - \frac{\mathbf{u}\cdot\mathbf{v}}{\|u\|_2\|v\|_2}$. The inner
/// product is `transpose(u) * v` — §07 lists transposed-vector–vector as a `mul`
/// domain, and `mul` on two plain vectors is not elementwise.
fn cosine(m: &mut Module, u: NodeId, v: NodeId) -> NodeId {
    let ut = build_call(m, "transpose", &[u]);
    let dot = mul(m, ut, v);
    let nu = build_call(m, "l2norm", &[u]);
    let nv = build_call(m, "l2norm", &[v]);
    let denom = mul(m, nu, nv);
    let ratio = divide(m, dot, denom);
    let one = lit(m, 1.0);
    sub(m, one, ratio)
}

// -- conformance support ---------------------------------------------------

/// The qualified name a residual `CallHead::User` call's callee spells, when that
/// callee is a module-namespace member ref. Lets `conformance::is_flatpdl` name the
/// member in its refusal instead of reporting the generic "residual user call" —
/// the surviving calls are exactly the §09 function members
/// [`lower_std_module_functions`] has no base-op form for.
///
/// Prefers the `standard_module` MODULE name, which needs the alias binding.
/// Root-based DCE runs between the lowering pass and the conformance check and can
/// drop that binding, so the fallback reads the alias and member straight off the
/// surviving ref node — DCE removes bindings, never nodes.
pub(crate) fn residual_std_member(m: &Module, call: NodeId) -> Option<String> {
    let Node::Call(c) = m.node(call) else {
        return None;
    };
    let CallHead::User(callee) = c.head else {
        return None;
    };
    if let Some((module, member)) = member_of_callee(m, callee) {
        return Some(format!("{module}.{member}"));
    }
    let ref_id = resolve_ref_one(m, callee).0;
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name: member,
    }) = *m.node(ref_id)
    else {
        return None;
    };
    // A `load_module` alias would also match this shape, but such a ref is grafted
    // to a local callee before the loop; one reaching here has no live alias
    // binding at all, so the spelling is all there is to report.
    m.binding_by_name(alias)
        .is_none()
        .then(|| format!("{}.{}", m.resolve(alias), m.resolve(member)))
}
