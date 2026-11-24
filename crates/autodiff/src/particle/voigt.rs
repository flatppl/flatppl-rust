//! Analytic Voigt scores from the Faddeeva equation and wing expansion.

use flatppl_core::voigt::{FADDEEVA_A, FADDEEVA_L};

use super::{Graph, NodeId, real};

type Pair = [NodeId; 2];

pub(super) fn score(
    g: &mut Graph<'_>,
    mean: NodeId,
    width: NodeId,
    sigma: NodeId,
    x: NodeId,
    slot: usize,
) -> NodeId {
    let delta = op!(g, "sub", x, mean);
    let half = real(g.module, 0.5);
    let gamma = op!(g, "mul", half, width);
    let inv_root2 = real(g.module, std::f64::consts::FRAC_1_SQRT_2);
    let x = op!(g, "mul", inv_root2, op!(g, "divide", delta, sigma));
    let y = op!(g, "mul", inv_root2, op!(g, "divide", gamma, sigma));
    let eight = real(g.module, 8.0);
    let far = op!(g, "gt", op!(g, "max", op!(g, "abs", x), y), eight);
    let zero = real(g.module, 0.0);
    let one = real(g.module, 1.0);
    let xx = op!(g, "ifelse", far, zero, x);
    let yy = op!(g, "ifelse", far, one, y);
    let central = center(g, xx, yy, sigma, slot);
    let delta = op!(g, "ifelse", far, delta, eight);
    let gamma = op!(g, "ifelse", far, gamma, eight);
    let s = op!(g, "ifelse", far, sigma, one);
    let wing = wing(g, delta, gamma, s, slot);
    let result = op!(g, "ifelse", far, wing, central);
    match slot {
        0 => op!(g, "neg", result),
        1 => op!(g, "mul", half, result),
        _ => result,
    }
}

fn mul(g: &mut Graph<'_>, [a, b]: Pair, [c, d]: Pair) -> Pair {
    [
        op!(g, "sub", op!(g, "mul", a, c), op!(g, "mul", b, d)),
        op!(g, "add", op!(g, "mul", a, d), op!(g, "mul", b, c)),
    ]
}

fn scaled(g: &mut Graph<'_>, [x, y]: Pair, c: NodeId) -> Pair {
    [op!(g, "mul", c, x), op!(g, "mul", c, y)]
}

fn add(g: &mut Graph<'_>, [a, b]: Pair, [c, d]: Pair) -> Pair {
    [op!(g, "add", a, c), op!(g, "add", b, d)]
}

fn center(g: &mut Graph<'_>, x: NodeId, y: NodeId, s: NodeId, slot: usize) -> NodeId {
    let two = real(g.module, 2.0);
    let one = real(g.module, 1.0);
    let zero = real(g.module, 0.0);
    let minus_two = real(g.module, -2.0);
    let normal = real(g.module, 1.0 / std::f64::consts::PI.sqrt());
    let small = real(g.module, 0.01);
    let near_axis = op!(g, "lt", y, small);
    let w_rational = rational(g, x, y, normal);
    // Avoid cancellation in a very narrow Lorentzian wing by starting
    // the Taylor expansion at the exact real value exp(-x²) on the axis.
    let axis_y = op!(g, "ifelse", near_axis, y, zero);
    let wi = rational(g, x, zero, normal)[1];
    let mut prev = [op!(g, "exp", op!(g, "neg", op!(g, "mul", x, x))), wi];
    let ax = op!(g, "mul", minus_two, x);
    let source = op!(g, "mul", two, normal);
    let mut coefficient = scaled(g, prev, ax);
    coefficient[1] = op!(g, "add", coefficient[1], source);
    let step = [zero, axis_y];
    let mut power = step;
    let term = mul(g, coefficient, power);
    let mut series = add(g, prev, term);
    for k in 1..12 {
        let a = scaled(g, coefficient, ax);
        let b = scaled(g, prev, minus_two);
        let total = add(g, a, b);
        let inverse = real(g.module, 1.0 / (k + 1) as f64);
        let next = scaled(g, total, inverse);
        power = mul(g, power, step);
        let term = mul(g, next, power);
        series = add(g, series, term);
        prev = coefficient;
        coefficient = next;
    }
    let w = [
        op!(g, "ifelse", near_axis, series[0], w_rational[0]),
        op!(g, "ifelse", near_axis, series[1], w_rational[1]),
    ];
    // w'(z) = -2 z w(z) + 2i/sqrt(pi).
    let zw = mul(g, [x, y], w);
    let derivative = [
        op!(g, "mul", minus_two, zw[0]),
        op!(g, "sub", source, op!(g, "mul", two, zw[1])),
    ];
    let result = if slot == 2 {
        let zder = mul(g, [x, y], derivative);
        op!(
            g,
            "neg",
            op!(g, "add", one, op!(g, "divide", zder[0], w[0]))
        )
    } else {
        let v = if slot == 1 {
            op!(g, "neg", derivative[1])
        } else {
            derivative[0]
        };
        let factor = real(g.module, std::f64::consts::FRAC_1_SQRT_2);
        op!(g, "mul", factor, op!(g, "divide", v, w[0]))
    };
    op!(g, "divide", result, s)
}

fn rational(g: &mut Graph<'_>, x: NodeId, y: NodeId, c: NodeId) -> Pair {
    let l = real(g.module, FADDEEVA_L);
    let a = op!(g, "add", l, y);
    let norm = op!(g, "add", op!(g, "mul", a, a), op!(g, "mul", x, x));
    let inverse = [op!(g, "divide", a, norm), op!(g, "divide", x, norm)];
    let twice = real(g.module, 2.0 * FADDEEVA_L);
    let z = scaled(g, inverse, twice);
    let one = real(g.module, 1.0);
    let z = [op!(g, "sub", z[0], one), z[1]];
    let square = mul(g, inverse, inverse);
    let z4 = mul(g, z, z);
    let z4 = mul(g, z4, z4);
    let zero = real(g.module, 0.0);
    let mut polynomial = [zero, zero];
    // Group Horner into four-term blocks to keep source expression depth
    // bounded while retaining the coefficients used by primal lowering.
    let (chunks, []) = FADDEEVA_A.as_chunks::<4>() else {
        unreachable!("Faddeeva coefficients fill four-term blocks");
    };
    for chunk in chunks {
        let mut block = [zero, zero];
        for &coefficient in chunk {
            block = mul(g, block, z);
            let coefficient = real(g.module, coefficient);
            block[0] = op!(g, "add", block[0], coefficient);
        }
        polynomial = mul(g, polynomial, z4);
        polynomial = add(g, polynomial, block);
    }
    let poly = mul(g, polynomial, square);
    let two = real(g.module, 2.0);
    let poly = scaled(g, poly, two);
    let direct = scaled(g, inverse, c);
    add(g, poly, direct)
}

fn wing(g: &mut Graph<'_>, delta: NodeId, gamma: NodeId, sigma: NodeId, slot: usize) -> NodeId {
    let two = real(g.module, 2.0);
    let one = real(g.module, 1.0);
    let zero = real(g.module, 0.0);
    let root2 = real(g.module, std::f64::consts::SQRT_2);
    let c = real(g.module, -1.0 / std::f64::consts::PI.sqrt());
    let scale = op!(g, "max", op!(g, "abs", delta), gamma);
    let r = op!(g, "divide", delta, scale);
    let i = op!(g, "divide", gamma, scale);
    let norm = op!(g, "add", op!(g, "mul", r, r), op!(g, "mul", i, i));
    let q = [
        op!(g, "divide", r, norm),
        op!(g, "neg", op!(g, "divide", i, norm)),
    ];
    let a = op!(g, "mul", root2, op!(g, "divide", sigma, scale));
    let inv_z = scaled(g, q, a);
    let second = mul(g, inv_z, inv_z);
    let mut term = q;
    let mut total = q;
    let minus_one = real(g.module, -1.0);
    let mut derivative = if slot == 2 {
        [zero, zero]
    } else {
        scaled(g, q, minus_one)
    };
    for n in 1..=32 {
        term = mul(g, term, second);
        let factor = real(g.module, n as f64 - 0.5);
        term = scaled(g, term, factor);
        total = add(g, total, term);
        let slope = real(
            g.module,
            if slot == 2 {
                2.0 * n as f64
            } else {
                -2.0 * n as f64 - 1.0
            },
        );
        let contribution = scaled(g, term, slope);
        derivative = add(g, derivative, contribution);
    }
    let base = op!(g, "mul", c, total[1]);
    let numerator = if slot == 2 {
        op!(g, "divide", op!(g, "mul", c, derivative[1]), sigma)
    } else {
        let derivative = mul(g, derivative, q);
        let component = if slot == 1 {
            derivative[0]
        } else {
            derivative[1]
        };
        op!(g, "divide", op!(g, "mul", c, component), scale)
    };

    // Near the real axis a Gaussian wing can dominate for very small width.
    // On this branch x≤32 and y<0.01 before forming the elementary products.
    let small = real(g.module, 0.01);
    let thirty_two = real(g.module, 32.0);
    let gaussian_branch = op!(
        g,
        "land",
        op!(g, "lt", i, op!(g, "mul", a, small)),
        op!(g, "lt", op!(g, "abs", r), op!(g, "mul", a, thirty_two))
    );
    let r = op!(g, "ifelse", gaussian_branch, r, zero);
    let i = op!(g, "ifelse", gaussian_branch, i, zero);
    let a = op!(g, "ifelse", gaussian_branch, a, one);
    let x = op!(g, "divide", r, a);
    let y = op!(g, "divide", i, a);
    let x2 = op!(g, "mul", x, x);
    let y2 = op!(g, "mul", y, y);
    let exponent = op!(g, "sub", y2, x2);
    let phase = op!(g, "mul", two, op!(g, "mul", x, y));
    let exp = op!(
        g,
        "divide",
        op!(g, "ifelse", gaussian_branch, op!(g, "exp", exponent), zero),
        a
    );
    let cosine = op!(g, "cos", phase);
    let sine = op!(g, "sin", phase);
    let correction = op!(g, "mul", exp, cosine);
    let gaussian = match slot {
        2 => op!(
            g,
            "add",
            op!(
                g,
                "mul",
                op!(g, "sub", op!(g, "mul", two, op!(g, "neg", exponent)), one),
                cosine
            ),
            op!(g, "mul", op!(g, "mul", two, phase), sine)
        ),
        1 => op!(
            g,
            "mul",
            root2,
            op!(g, "sub", op!(g, "mul", y, cosine), op!(g, "mul", x, sine))
        ),
        _ => op!(
            g,
            "neg",
            op!(
                g,
                "mul",
                root2,
                op!(g, "add", op!(g, "mul", x, cosine), op!(g, "mul", y, sine))
            )
        ),
    };
    let corrected = op!(g, "divide", op!(g, "mul", exp, gaussian), sigma);
    op!(
        g,
        "divide",
        op!(g, "add", numerator, corrected),
        op!(g, "add", base, correction)
    )
}
