use crate::{
    emit::{broadcast, parameter_function},
    wiener::log_proposal_ratio,
};
use flatppl_convert::Builder;
use flatppl_core::{Axis, Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

const PROPOSALS: i64 = 64;

pub(crate) fn law(
    b: &mut Builder<'_>,
    drift: NodeId,
    separation: NodeId,
    nondecision: NodeId,
    bias: NodeId,
) -> NodeId {
    let rows = b.self_ref("N");
    let proposals = b.lit_int(PROPOSALS);
    let count = b.call("mul", &[proposals, rows]);
    let zero = b.lit_real(0.0);
    let one = b.lit_real(1.0);
    let normal = b.call("Normal", &[zero, one]);
    let unit_interval = b.call("interval", &[zero, one]);
    let uniform = b.call("Uniform", &[unit_interval]);
    let normal = b.call("iid", &[normal, count]);
    let uniform = b.call("iid", &[uniform, count]);
    let noise = b.call_fields(
        "joint",
        &[
            ("z", normal),
            ("side", uniform),
            ("root", uniform),
            ("accept", uniform),
        ],
    );
    let reals = b.call_head("reals");
    let vector = b.call("cartpow", &[reals, count]);
    let domain = b.call_fields(
        "cartprod",
        &[
            ("z", vector),
            ("side", vector),
            ("root", vector),
            ("accept", vector),
        ],
    );
    let noise_point = b.alloc_name("_wiener_noise");
    b.bind_set(&noise_point, domain);
    let predictors = b.call_fields(
        "record",
        &[
            ("drift", drift),
            ("bs", separation),
            ("ndt", nondecision),
            ("bias", bias),
        ],
    );
    let point = b.alloc_name("_wiener_predictors");
    b.bind(&point, predictors);
    let body = transform(b, rows, &noise_point, &point);
    let transform = parameter_function(b, "functionof", body, &[noise_point, point]);
    b.call("pushfwd", &[transform, noise, predictors])
}

fn transform(b: &mut Builder<'_>, rows: NodeId, noise: &str, point: &str) -> NodeId {
    let noise = b.self_ref(noise);
    let [z, side, root, accept] = ["z", "side", "root", "accept"].map(|name| field(b, noise, name));
    let point = b.self_ref(point);
    let proposals = b.lit_int(PROPOSALS);
    let zero = b.lit_real(0.0);
    let zeros = b.call("fill", &[zero, rows]);
    let [drift, separation, nondecision, bias] = ["drift", "bs", "ndt", "bias"].map(|name| {
        let value = field(b, point, name);
        // Constant auxiliaries are scalars. Give them the observation axis
        // before laying out one block of rows per proposal.
        let vector = broadcast(b, "add", &[value, zeros]);
        b.call("tile", &[vector, proposals])
    });
    let propose = function(b, ["z", "side", "root", "drift", "bs", "bias"], propose);
    let proposed = b.call(
        "broadcast",
        &[propose, z, side, root, drift, separation, bias],
    );
    let proposed = bind(b, "_wiener_trial", proposed);
    let [tau, distance, complement, decision, valid] =
        ["tau", "distance", "complement", "decision", "valid"].map(|name| field(b, proposed, name));
    let filter = function(
        b,
        [
            "tau",
            "distance",
            "complement",
            "valid",
            "accept",
            "bs",
            "ndt",
        ],
        filter,
    );
    let candidates = b.call(
        "broadcast",
        &[
            filter,
            tau,
            distance,
            complement,
            valid,
            accept,
            separation,
            nondecision,
        ],
    );
    let candidates = bind(b, "_wiener_filtered_trials", candidates);
    let time = field(b, candidates, "time");
    let accepted = field(b, candidates, "accepted");
    let [time, decision, accepted] = [time, decision, accepted].map(|value| {
        let blocks = b.call("partition", &[value, rows]);
        b.call("rowstack", &[blocks])
    });

    let lead = b.lit_int(1);
    let tail = b.lit_int(0);
    let index_seed = b.call("fill", &[zero, proposals]);
    let indices = b.call("indicesof", &[index_seed]);
    let indices = b.call("addaxes", &[indices, tail, lead]);
    let exhausted = b.lit_real((PROPOSALS + 1) as f64);
    let ranked = broadcast(b, "ifelse", &[accepted, indices, exhausted]);
    let first = reduce_proposals(b, "minimum", ranked);
    let first = bind(b, "_wiener_first_proposal", first);
    let first_row = b.call("addaxes", &[first, lead, tail]);
    let before = broadcast(b, "le", &[indices, first_row]);
    let after = broadcast(b, "ge", &[indices, first_row]);
    let chosen = broadcast(b, "land", &[before, after]);
    let ok = broadcast(b, "lt", &[first, exhausted]);
    let missing = b.lit_real(f64::NAN);
    let [time, decision] = [time, decision].map(|value| {
        let chosen = broadcast(b, "ifelse", &[chosen, value, zero]);
        reduce_proposals(b, "sum", chosen)
    });
    // A response that cannot be represented at the requested precision
    // fails after selection, rather than changing which proposal wins.
    let delay = field(b, point, "ndt");
    let positive_elapsed = broadcast(b, "gt", &[time, delay]);
    let finite_time = broadcast(b, "isfinite", &[time]);
    let ok = broadcast(b, "land", &[ok, positive_elapsed]);
    let ok = broadcast(b, "land", &[ok, finite_time]);
    let ok = bind(b, "_wiener_valid_draw", ok);

    // Keep the response shape in the query ABI even when partition/tile
    // leave the proposal reductions' inferred extents dynamic.
    let row_indices = b.call("indicesof", &[zeros]);
    let [time, decision] = [time, decision].map(|result| {
        let result = broadcast(b, "ifelse", &[ok, result, missing]);
        b.call("get", &[result, row_indices])
    });
    b.call_fields("record", &[("y", time), ("decision", decision)])
}

fn propose(b: &mut Builder<'_>, [z, side, root, drift, separation, bias]: [NodeId; 6]) -> NodeId {
    let [zero, quarter, half, one, minus_two] =
        [0.0, 0.25, 0.5, 1.0, -2.0].map(|value| b.lit_real(value));
    let scaled_drift = b.call("mul", &[separation, drift]);
    let valid = [
        b.call("isfinite", &[scaled_drift]),
        b.call("gt", &[separation, zero]),
        b.call("gt", &[bias, zero]),
        b.call("lt", &[bias, one]),
    ]
    .into_iter()
    .reduce(|left, right| b.call("land", &[left, right]))
    .unwrap();
    let bias = b.call("ifelse", &[valid, bias, half]);
    let reflected = b.call("sub", &[one, bias]);
    let scaled_drift = b.call("ifelse", &[valid, scaled_drift, zero]);
    let speed = b.call("abs", &[scaled_drift]);

    // Each proposal chooses a side in proportion to the mass of its one-sided
    // first-passage law. The two masses sum to at most two, giving at least
    // 1/2 acceptance and at most 2^-64 probability of exhausting the trials.
    let positive = b.call("ge", &[scaled_drift, zero]);
    let against_drift = b.call("ifelse", &[positive, bias, reflected]);
    let mass = b.call("mul", &[speed, against_drift]);
    let mass = b.call("mul", &[minus_two, mass]);
    let mass = b.call("exp", &[mass]);
    let upper_mass = b.call("ifelse", &[positive, one, mass]);
    let total_mass = b.call("add", &[one, mass]);
    let upper_probability = b.call("divide", &[upper_mass, total_mass]);
    let upper = b.call("lt", &[side, upper_probability]);
    let distance = b.call("ifelse", &[upper, reflected, bias]);
    let complement = b.call("ifelse", &[upper, bias, reflected]);

    // Michael, Schucany and Haas (1976), doi:10.1080/00031305.1976.10479147.
    // The inverse Gaussian's small root uses the rationalized expression,
    // which reduces to the Lévy draw at zero drift.
    let product = b.call("mul", &[distance, speed]);
    let z_squared = b.call("mul", &[z, z]);
    let half_squared = b.call("mul", &[half, z_squared]);
    let quarter_squared = b.call("mul", &[quarter, z_squared]);
    let radicand = b.call("add", &[product, quarter_squared]);
    let radical = b.call("sqrt", &[radicand]);
    let abs_z = b.call("abs", &[z]);
    let term = b.call("mul", &[abs_z, radical]);
    let denominator = b.call("add", &[product, half_squared]);
    let denominator = b.call("add", &[denominator, term]);
    let positive_denominator = b.call("gt", &[denominator, zero]);
    let finite_denominator = b.call("isfinite", &[denominator]);
    let valid_denominator = b.call("land", &[positive_denominator, finite_denominator]);
    let denominator = b.call("ifelse", &[valid_denominator, denominator, one]);
    let ratio = b.call("divide", &[product, denominator]);
    let small_threshold = b.call("add", &[one, ratio]);
    let small_threshold = b.call("divide", &[one, small_threshold]);
    let use_small = b.call("lt", &[root, small_threshold]);
    let small_time = b.call("divide", &[distance, denominator]);
    let small_time = b.call("mul", &[distance, small_time]);
    let positive_speed = b.call("gt", &[speed, zero]);
    let safe_speed = b.call("ifelse", &[positive_speed, speed, one]);
    let large_time = b.call("divide", &[denominator, safe_speed]);
    let large_time = b.call("divide", &[large_time, safe_speed]);
    let scaled_time = b.call("ifelse", &[use_small, small_time, large_time]);

    let valid = b.call("land", &[valid, valid_denominator]);
    let decision = b.call("ifelse", &[upper, one, zero]);
    b.call_fields(
        "record",
        &[
            ("tau", scaled_time),
            ("distance", distance),
            ("complement", complement),
            ("decision", decision),
            ("valid", valid),
        ],
    )
}

fn filter(
    b: &mut Builder<'_>,
    [
        scaled_time,
        distance,
        complement,
        valid,
        accept,
        separation,
        nondecision,
    ]: [NodeId; 7],
) -> NodeId {
    let zero = b.lit_real(0.0);
    let quarter = b.lit_real(0.25);
    let valid = [
        valid,
        b.call("gt", &[scaled_time, zero]),
        b.call("isfinite", &[scaled_time]),
        b.call("ge", &[nondecision, zero]),
        b.call("isfinite", &[nondecision]),
    ]
    .into_iter()
    .reduce(|left, right| b.call("land", &[left, right]))
    .unwrap();
    let scaled_time = b.call("ifelse", &[valid, scaled_time, quarter]);
    let log_ratio = log_proposal_ratio(b, scaled_time, distance, complement);
    let log_uniform = b.call("log", &[accept]);
    let accepted = b.call("lt", &[log_uniform, log_ratio]);

    let elapsed = b.call("mul", &[scaled_time, separation]);
    let elapsed = b.call("mul", &[elapsed, separation]);
    let time = b.call("add", &[nondecision, elapsed]);
    let accepted = b.call("land", &[valid, accepted]);
    b.call_fields("record", &[("time", time), ("accepted", accepted)])
}

fn function<const N: usize>(
    b: &mut Builder<'_>,
    names: [&str; N],
    body: fn(&mut Builder<'_>, [NodeId; N]) -> NodeId,
) -> NodeId {
    let refs = names.map(|name| Ref {
        ns: RefNs::Local,
        name: b.sym(&format!("_{name}_")),
    });
    let args = refs.map(|reference| b.m.alloc(Node::Ref(reference)));
    let result = body(b, args);
    let head = b.sym("functionof");
    let inputs = names
        .map(|name| b.sym(name))
        .into_iter()
        .zip(refs)
        .collect();
    b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([result]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs)),
    }))
}

fn reduce_proposals(b: &mut Builder<'_>, reduction: &str, matrix: NodeId) -> NodeId {
    let mut axis = |name| {
        let name = b.sym(name);
        b.m.alloc(Node::Axis(Axis {
            name,
            variance: None,
        }))
    };
    let proposal = axis("proposal");
    let row = axis("row");
    let cell = b.call("get", &[matrix, proposal, row]);
    let outputs = b.array(&[row]);
    let reduction = b.call_head(reduction);
    b.call("aggregate", &[reduction, outputs, cell])
}

fn bind(b: &mut Builder<'_>, name: &str, value: NodeId) -> NodeId {
    let name = b.alloc_name(name);
    b.bind(&name, value);
    b.self_ref(&name)
}

fn field(b: &mut Builder<'_>, record: NodeId, name: &str) -> NodeId {
    let name = b.str_lit(name);
    b.call("get", &[record, name])
}
