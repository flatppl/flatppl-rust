//! Scores of normalized §09 particle-physics densities.

use flatppl_core::graph::build_call;
use flatppl_core::{Module, NodeId};

use crate::{Emitter, Step, callbacks, real, rules::special_call, tensor};

// Evaluate child expressions before borrowing the graph builder.
macro_rules! op {
    ($graph:ident, $head:literal, $($arg:expr),+ $(,)?) => {{
        let args = [$($arg),+];
        $graph.call($head, &args)
    }};
}

mod voigt;

struct Graph<'a> {
    module: &'a mut Module,
    pointwise: bool,
}

impl Graph<'_> {
    fn call(&mut self, head: &str, args: &[NodeId]) -> NodeId {
        if self.pointwise {
            tensor::broadcast(self.module, head, args)
        } else {
            build_call(self.module, head, args)
        }
    }
}

pub(super) fn partial(emitter: &mut Emitter, step: &Step<'_>, slot: usize) -> Option<NodeId> {
    let args: Vec<_> = step.args.iter().map(|&arg| emitter.primal(arg)).collect();
    let pointwise = step.pointwise(&emitter.module);
    if let ("Argus log density", &[m, c, p, x]) = (step.op, args.as_slice())
        && pointwise
        && matches!(slot, 1 | 2)
    {
        let module = &mut emitter.module;
        let mut entries = Vec::new();
        let m_cell = callbacks::callback_input(module, "argus_endpoint", &mut entries);
        let c_cell = callbacks::callback_input(module, "argus_slope", &mut entries);
        let p_cell = callbacks::callback_input(module, "argus_power", &mut entries);
        let x_cell = callbacks::callback_input(module, "argus_observation", &mut entries);
        let cell = argus(
            &mut Graph {
                module,
                pointwise: false,
            },
            m_cell,
            c_cell,
            p_cell,
            x_cell,
            slot,
        );
        let f = callbacks::reify(module, cell, entries);
        return Some(build_call(module, "broadcast", &[f, m, c, p, x]));
    }
    let g = &mut Graph {
        module: &mut emitter.module,
        pointwise,
    };
    match (step.op, args.as_slice()) {
        ("Voigtian log density", &[mean, width, sigma, x]) => {
            Some(voigt::score(g, mean, width, sigma, x, slot))
        }

        ("Landau log density", &[loc, scale, x]) => Some(landau(g, loc, scale, x, slot)),

        ("Argus log density", &[m, c, p, x]) => Some(argus(g, m, c, p, x, slot)),
        ("BifurcatedNormal log density", &[m, sl, sr, x]) => {
            Some(bifurcated(g, m, [sl, sr], x, slot))
        }
        ("RelativisticBreitWigner log density", &[m, w, x]) => Some(breit_wigner(g, m, w, x, slot)),
        ("CrystalBall log density", &[m, s, a, n, x]) => Some(crystal(g, m, [s, a, n], x, slot)),
        ("DoubleSidedCrystalBall log density", &[m, sl, sr, al, ar, nl, nr, x]) => {
            Some(double_crystal(g, m, [[sl, al, nl], [sr, ar, nr]], x, slot))
        }
        _ => None,
    }
}

fn bifurcated(g: &mut Graph<'_>, m: NodeId, widths: [NodeId; 2], x: NodeId, slot: usize) -> NodeId {
    let [sl, sr] = widths;
    let left = op!(g, "lt", x, m);
    let width = op!(g, "ifelse", left, sl, sr);
    let t = op!(g, "divide", op!(g, "sub", x, m), width);
    let mean = op!(g, "divide", t, width);
    match slot {
        0 => mean,
        3 => op!(g, "neg", mean),
        1 | 2 => {
            let one = real(g.module, 1.0);
            let zero = real(g.module, 0.0);
            let local = op!(g, "mul", t, mean);
            let local = if slot == 1 {
                op!(g, "ifelse", left, local, zero)
            } else {
                op!(g, "ifelse", left, zero, local)
            };
            op!(
                g,
                "sub",
                local,
                op!(g, "divide", one, op!(g, "add", sl, sr))
            )
        }
        _ => unreachable!("bifurcated score coordinate"),
    }
}

fn breit_wigner(g: &mut Graph<'_>, m: NodeId, w: NodeId, x: NodeId, slot: usize) -> NodeId {
    let zero = real(g.module, 0.0);
    let one = real(g.module, 1.0);
    let two = real(g.module, 2.0);
    let valid = op!(g, "gt", x, zero);
    let x = op!(g, "ifelse", valid, x, m);
    let offset = op!(g, "mul", op!(g, "sub", x, m), op!(g, "add", x, m));
    let mw = op!(g, "mul", m, w);
    let r = hypot(g, offset, mw);
    let fit = op!(g, "divide", offset, r);
    let tail = op!(g, "divide", mw, r);
    let tail2 = op!(g, "mul", tail, tail);
    let four = real(g.module, 4.0);
    let score = match slot {
        2 => op!(
            g,
            "neg",
            op!(g, "mul", four, op!(g, "mul", fit, op!(g, "divide", x, r)))
        ),
        0 | 1 => {
            // Write the normalization using h=hypot(m,w), so gamma=m*h.
            // log(1/M) = const + 3/2 log(m) + log(w) + log(h)
            //            - 1/2 log(m+h).
            let h = hypot(g, m, w);
            let parameter = if slot == 0 { m } else { w };
            let dh = op!(g, "divide", parameter, h);
            let correction = if slot == 0 {
                op!(g, "add", one, dh)
            } else {
                dh
            };
            let correction = op!(
                g,
                "divide",
                correction,
                op!(g, "mul", two, op!(g, "add", m, h))
            );
            let log_h = op!(g, "divide", dh, h);
            let direct = real(g.module, if slot == 0 { 1.5 } else { 1.0 });
            let normalizer = op!(
                g,
                "sub",
                op!(g, "add", op!(g, "divide", direct, parameter), log_h),
                correction
            );
            let scale = op!(g, "divide", op!(g, "mul", two, tail2), parameter);
            let offset = if slot == 0 {
                op!(g, "mul", four, op!(g, "mul", fit, op!(g, "divide", m, r)))
            } else {
                zero
            };
            op!(g, "sub", op!(g, "add", normalizer, offset), scale)
        }
        _ => unreachable!("Breit-Wigner score coordinate"),
    };
    op!(g, "ifelse", valid, score, zero)
}

// The unnormalized side score at distance u. The tail is written as
// exp(-a²/2) * (1+a*(u-a)/n)^(-n), continuous with the Gaussian at u=a.
fn crystal_local(g: &mut Graph<'_>, u: NodeId, a: NodeId, n: NodeId, slot: usize) -> NodeId {
    let zero = real(g.module, 0.0);
    let one = real(g.module, 1.0);
    let tail = op!(g, "gt", u, a);
    let gap = op!(g, "max", zero, op!(g, "sub", u, a));
    let z = op!(g, "mul", op!(g, "divide", a, n), gap);
    let denominator = op!(g, "add", one, z);
    let local = match slot {
        0 => op!(g, "neg", op!(g, "divide", a, denominator)),
        1 => {
            // -a - (u-2a)/denominator; use gap to avoid forming 2a.
            op!(
                g,
                "sub",
                op!(g, "divide", op!(g, "sub", a, gap), denominator),
                a
            )
        }
        2 => op!(
            g,
            "sub",
            op!(g, "divide", z, denominator),
            op!(g, "log1p", z)
        ),
        _ => unreachable!("Crystal Ball side score coordinate"),
    };
    let core = if slot == 0 { op!(g, "neg", u) } else { zero };
    op!(g, "ifelse", tail, local, core)
}

// Half a Gaussian core plus the matching integrated power tail.
fn crystal_area(g: &mut Graph<'_>, a: NodeId, n: NodeId, slot: usize) -> NodeId {
    let half = real(g.module, -0.5);
    let one = real(g.module, 1.0);
    let e = op!(g, "exp", op!(g, "mul", half, op!(g, "mul", a, a)));
    let n1 = op!(g, "sub", n, one);
    let tail = op!(g, "mul", op!(g, "divide", e, a), op!(g, "divide", n, n1));
    match slot {
        0 => {
            let root2 = real(g.module, std::f64::consts::SQRT_2);
            let factor = real(g.module, (std::f64::consts::PI / 2.0).sqrt());
            let t = op!(g, "divide", a, root2);
            let erf = special_call(g.module, "erf", vec![t], g.pointwise);
            op!(g, "add", op!(g, "mul", factor, erf), tail)
        }
        1 => op!(
            g,
            "sub",
            e,
            op!(g, "mul", tail, op!(g, "add", a, op!(g, "divide", one, a)))
        ),
        2 => op!(g, "neg", op!(g, "divide", op!(g, "divide", tail, n), n1)),
        _ => unreachable!("Crystal Ball normalizer coordinate"),
    }
}

fn crystal(g: &mut Graph<'_>, m: NodeId, side: [NodeId; 3], x: NodeId, slot: usize) -> NodeId {
    let [s, a, n] = side;
    let u = op!(g, "divide", op!(g, "sub", m, x), s);
    match slot {
        0 | 4 => {
            let du = crystal_local(g, u, a, n, 0);
            let d = op!(g, "divide", du, s);
            if slot == 0 { d } else { op!(g, "neg", d) }
        }
        1 => {
            let du = crystal_local(g, u, a, n, 0);
            let one = real(g.module, 1.0);
            op!(
                g,
                "divide",
                op!(g, "sub", op!(g, "neg", op!(g, "mul", u, du)), one),
                s
            )
        }
        2 | 3 => {
            let area = crystal_area(g, a, n, 0);
            let half = real(g.module, (std::f64::consts::PI / 2.0).sqrt());
            let darea = crystal_area(g, a, n, slot - 1);
            let local = crystal_local(g, u, a, n, slot - 1);
            op!(
                g,
                "sub",
                local,
                op!(g, "divide", darea, op!(g, "add", area, half))
            )
        }
        _ => unreachable!("Crystal Ball score coordinate"),
    }
}

fn double_crystal(
    g: &mut Graph<'_>,
    m: NodeId,
    sides: [[NodeId; 3]; 2],
    x: NodeId,
    slot: usize,
) -> NodeId {
    let [[sl, al, nl], [sr, ar, nr]] = sides;
    let left = op!(g, "lt", x, m);
    let s = op!(g, "ifelse", left, sl, sr);
    let a = op!(g, "ifelse", left, al, ar);
    let n = op!(g, "ifelse", left, nl, nr);
    let u = op!(g, "divide", op!(g, "abs", op!(g, "sub", x, m)), s);
    if matches!(slot, 0 | 7) {
        let local = crystal_local(g, u, a, n, 0);
        let local = op!(g, "divide", local, s);
        let negative = op!(g, "neg", local);
        return if slot == 0 {
            op!(g, "ifelse", left, local, negative)
        } else {
            op!(g, "ifelse", left, negative, local)
        };
    }
    let hl = crystal_area(g, al, nl, 0);
    let hr = crystal_area(g, ar, nr, 0);
    let total = op!(g, "add", op!(g, "mul", sl, hl), op!(g, "mul", sr, hr));
    let zero = real(g.module, 0.0);
    let index = (slot - 1) % 2;
    let coordinate = (slot - 1) / 2;
    let [side_s, side_a, side_n] = sides[index];
    let (local, dtotal) = if coordinate == 0 {
        let du = crystal_local(g, u, a, n, 0);
        let local = op!(g, "neg", op!(g, "divide", op!(g, "mul", u, du), s));
        (local, [hl, hr][index])
    } else {
        let local = crystal_local(g, u, a, n, coordinate);
        let area = crystal_area(g, side_a, side_n, coordinate);
        (local, op!(g, "mul", side_s, area))
    };
    let local = if index == 0 {
        op!(g, "ifelse", left, local, zero)
    } else {
        op!(g, "ifelse", left, zero, local)
    };
    op!(g, "sub", local, op!(g, "divide", dtotal, total))
}

fn hypot(g: &mut Graph<'_>, a: NodeId, b: NodeId) -> NodeId {
    let a = op!(g, "abs", a);
    let b = op!(g, "abs", b);
    let hi = op!(g, "max", a, b);
    let lo = op!(g, "min", a, b);
    let t = op!(g, "divide", lo, hi);
    let one = real(g.module, 1.0);
    op!(
        g,
        "mul",
        hi,
        op!(g, "sqrt", op!(g, "add", one, op!(g, "mul", t, t)))
    )
}

fn argus(g: &mut Graph<'_>, m: NodeId, c: NodeId, p: NodeId, x: NodeId, slot: usize) -> NodeId {
    let zero = real(g.module, 0.0);
    let one = real(g.module, 1.0);
    let half = real(g.module, 0.5);
    let two = real(g.module, 2.0);
    let valid = op!(g, "land", op!(g, "gt", x, zero), op!(g, "lt", x, m));
    let x = op!(g, "ifelse", valid, x, op!(g, "mul", half, m));
    let ratio = op!(g, "divide", x, m);
    let q = op!(g, "mul", ratio, ratio);
    let u = op!(
        g,
        "mul",
        op!(g, "divide", op!(g, "sub", m, x), m),
        op!(g, "add", one, ratio)
    );
    let score = match slot {
        0 | 3 => {
            let kernel = op!(g, "add", op!(g, "divide", p, u), c);
            let slope = op!(g, "mul", two, op!(g, "mul", q, kernel));
            if slot == 0 {
                op!(g, "divide", op!(g, "sub", slope, two), m)
            } else {
                op!(g, "divide", op!(g, "sub", one, slope), x)
            }
        }
        1 | 2 => {
            let c = op!(g, "ifelse", valid, c, zero);
            let p = op!(g, "ifelse", valid, p, one);
            let (positive, expectation) = argus_moment(g, c, p, slot);
            if slot == 1 {
                op!(
                    g,
                    "ifelse",
                    positive,
                    op!(g, "sub", expectation, q),
                    op!(g, "sub", u, expectation)
                )
            } else {
                op!(g, "add", op!(g, "log", u), expectation)
            }
        }
        _ => unreachable!("Argus score coordinate"),
    };
    op!(g, "ifelse", valid, score, zero)
}

fn argus_moment(g: &mut Graph<'_>, c: NodeId, p: NodeId, slot: usize) -> (NodeId, NodeId) {
    use flatppl_core::{INTEGRATION_POINT, LOG_INTEGRAL};

    let zero = real(g.module, 0.0);
    let one = real(g.module, 1.0);
    let positive = op!(g, "ge", c, zero);
    let token = tensor::int(g.module, g.module.node_count() as i64);
    let t = build_call(g.module, INTEGRATION_POINT, &[token]);
    // u=exp(-t) gives the normalized log kernel (up to an additive
    // constant). For positive slopes integrate the deficit 1-u so the
    // observation score does not subtract two values close to one.
    let nt = op!(g, "neg", t);
    let u = op!(g, "exp", nt);
    let delta = op!(g, "expm1", nt);
    let centered = op!(g, "ifelse", positive, delta, u);
    let log_kernel = op!(
        g,
        "add",
        op!(g, "mul", c, centered),
        op!(g, "mul", op!(g, "add", p, one), nt)
    );
    let log_moment = if slot == 1 {
        op!(
            g,
            "ifelse",
            positive,
            op!(g, "log", op!(g, "neg", delta)),
            nt
        )
    } else {
        op!(g, "log", t)
    };
    let weighted = op!(g, "add", log_kernel, log_moment);
    let infinity = real(g.module, f64::INFINITY);
    let total = build_call(g.module, LOG_INTEGRAL, &[log_kernel, t, zero, infinity]);
    let moment = build_call(g.module, LOG_INTEGRAL, &[weighted, t, zero, infinity]);
    (positive, op!(g, "exp", op!(g, "sub", moment, total)))
}

fn landau(g: &mut Graph<'_>, loc: NodeId, scale: NodeId, x: NodeId, slot: usize) -> NodeId {
    use flatppl_core::landau::{C0, REGIONS};

    let zero = real(g.module, 0.0);
    let one = real(g.module, 1.0);
    let two = real(g.module, 2.0);
    let half = real(g.module, 0.5);
    let minus_five = real(g.module, -5.0);
    let twenty = real(g.module, 20.0);
    let z = op!(g, "divide", op!(g, "sub", x, loc), scale);
    let finite = op!(g, "isfinite", z);
    let z = op!(g, "ifelse", finite, z, zero);
    let left = op!(g, "min", z, minus_five);
    let exp_arg = op!(g, "add", left, one);
    let exp_derivative = op!(g, "exp", op!(g, "neg", exp_arg));
    let coefficient = real(g.module, 2.0 / (-4.0_f64).exp());
    let du = op!(g, "mul", coefficient, op!(g, "exp", exp_arg));
    let u = op!(g, "sub", du, one);
    let correction = chebyshev_derivative(g, u, C0);
    let mut score = op!(
        g,
        "add",
        op!(g, "sub", exp_derivative, half),
        op!(g, "mul", du, correction)
    );
    let right = op!(g, "ge", z, twenty);
    let right_z = op!(g, "max", z, twenty);
    let log_right = op!(g, "log", right_z);
    for region in REGIONS {
        let raw = if region.logarithmic { log_right } else { z };
        let lo = real(g.module, region.lower);
        let hi = real(g.module, region.upper);
        let within = op!(g, "land", op!(g, "ge", raw, lo), op!(g, "le", raw, hi));
        let within = if region.logarithmic {
            op!(g, "land", within, right)
        } else {
            within
        };
        let midpoint = real(g.module, 0.5 * (region.lower + region.upper));
        let factor = real(g.module, 2.0 / (region.upper - region.lower));
        let guarded = op!(g, "ifelse", within, raw, midpoint);
        let t = op!(g, "mul", factor, op!(g, "sub", guarded, midpoint));
        let slope = chebyshev_derivative(g, t, region.coefficients);
        let slope = op!(g, "mul", factor, slope);
        let slope = if region.logarithmic {
            op!(g, "divide", op!(g, "sub", slope, two), right_z)
        } else {
            slope
        };
        score = op!(g, "ifelse", within, slope, score);
    }
    // Differentiate the same Watson tail used by the primal approximation.
    let threshold = real(g.module, 1e8);
    let far = op!(g, "ge", z, threshold);
    let far_z = op!(g, "max", z, threshold);
    let inv = op!(g, "divide", one, far_z);
    let ln = op!(g, "log", far_z);
    let euler = std::f64::consts::EULER_GAMMA;
    let c1 = real(g.module, 2.0 * euler - 3.0);
    let c2 = real(g.module, euler - 11.0 / 6.0);
    let c3 = real(g.module, std::f64::consts::PI.powi(2) / 2.0 + 49.0 / 12.0);
    let three = real(g.module, 3.0);
    let six = real(g.module, 6.0);
    let p = op!(g, "add", op!(g, "mul", two, ln), c1);
    let offset = op!(g, "add", ln, c2);
    let q = op!(
        g,
        "sub",
        op!(g, "mul", three, op!(g, "mul", offset, offset)),
        c3
    );
    let r = op!(g, "mul", inv, op!(g, "add", p, op!(g, "mul", inv, q)));
    let dr = op!(
        g,
        "mul",
        inv,
        op!(
            g,
            "add",
            op!(g, "sub", two, p),
            op!(
                g,
                "mul",
                inv,
                op!(g, "sub", op!(g, "mul", six, offset), op!(g, "mul", two, q))
            )
        )
    );
    let far_score = op!(
        g,
        "mul",
        inv,
        op!(g, "sub", op!(g, "divide", dr, op!(g, "add", one, r)), two)
    );
    let score = op!(g, "ifelse", far, far_score, score);
    let score = match slot {
        0 => op!(g, "neg", op!(g, "divide", score, scale)),
        1 => op!(
            g,
            "neg",
            op!(
                g,
                "divide",
                op!(g, "add", one, op!(g, "mul", z, score)),
                scale
            )
        ),
        2 => op!(g, "divide", score, scale),
        _ => unreachable!("Landau score coordinate"),
    };
    op!(g, "ifelse", finite, score, zero)
}

fn chebyshev_derivative(g: &mut Graph<'_>, t: NodeId, coefficients: &[f64]) -> NodeId {
    // d/dt of a Chebyshev series is itself a Chebyshev series. Evaluate
    // its coefficients with the same Clenshaw recurrence as the primal.
    let degree = coefficients.len() - 1;
    let mut derivative = vec![0.; coefficients.len() + 1];
    for i in (0..degree).rev() {
        derivative[i] = derivative[i + 2] + 2.0 * (i + 1) as f64 * coefficients[i + 1];
    }
    derivative[0] *= 0.5;
    let two = real(g.module, 2.0);
    let twice = op!(g, "mul", two, t);
    let mut b1 = real(g.module, 0.0);
    let mut b2 = b1;
    for &c in derivative[1..degree].iter().rev() {
        let c = real(g.module, c);
        let b0 = op!(g, "add", c, op!(g, "sub", op!(g, "mul", twice, b1), b2));
        b2 = b1;
        b1 = b0;
    }
    let c = real(g.module, derivative[0]);
    op!(g, "add", c, op!(g, "sub", op!(g, "mul", t, b1), b2))
}
