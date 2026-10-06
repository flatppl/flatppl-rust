//! Portable definitions of the `pyhf_helpers` standard module.
//! Each expansion uses ordinary FlatPDL operations. Backends that retain a
//! helper call never construct this fallback body.

use super::*;
use flatppl_core::{Axis, ScalarType};

/// Share only the closed reference callable, never its arguments or results.
/// Declared Real formals make this independent of the first call's shape/type.
/// The cache belongs to one lowering pass over one module arena.
pub(super) fn lower_call(
    m: &mut Module,
    args: &[NodeId],
    member: &str,
    broadcast: bool,
    functions: &mut HashMap<String, NodeId>,
) -> Option<NodeId> {
    // Direct collection calls bind whole arrays. Explicit broadcast binds
    // one array-valued cell per application and retains its outer axes.
    if !matches!(member, "normsys_factor" | "histosys_shift") {
        return if broadcast {
            lower_member_broadcast(m, args, "pyhf_helpers", member)
        } else {
            lower_function(m, args, member)
        };
    }
    let function = if let Some(&function) = functions.get(member) {
        function
    } else {
        let scope = m.node_count();
        let names =
            flatppl_infer::builtin_catalogue().module_param_names("pyhf_helpers", member)?;
        let mut inputs = Vec::with_capacity(names.len());
        let mut cells = Vec::with_capacity(names.len());
        for (index, label) in names.iter().enumerate() {
            let label = m.intern(label);
            let reference = Ref {
                ns: RefNs::Local,
                name: m.intern(&format!("_pyhf_{scope}_{index}_")),
            };
            inputs.push((label, reference));
            let cell = m.alloc(Node::Ref(reference));
            m.set_type(cell, Type::Scalar(ScalarType::Real));
            cells.push(cell);
        }
        let body = lower_function(m, &cells, member)?;
        let head = CallHead::Builtin(m.intern("functionof"));
        let function = m.alloc(Node::Call(Call {
            head,
            args: vec![body].into(),
            named: vec![].into(),
            inputs: Some(Inputs::Spec(inputs.into())),
        }));
        functions.insert(member.to_owned(), function);
        function
    };
    let mut mapped_args = Vec::with_capacity(args.len() + 1);
    mapped_args.push(function);
    mapped_args.extend_from_slice(args);
    Some(build_call(m, "broadcast", &mapped_args))
}

/// Resolve the ordinary call forms before creating the closed reference body.
pub(super) fn ordered_args(m: &mut Module, call: &Call, member: &str) -> Option<Vec<NodeId>> {
    let names = flatppl_infer::builtin_catalogue().module_param_names("pyhf_helpers", member)?;
    let offset =
        usize::from(matches!(call.head, CallHead::Builtin(s) if m.resolve(s) == "broadcast"));
    let names: Vec<_> = names.iter().map(|name| m.intern(name)).collect();
    crate::kernel::bind_call_args(m, &names, &call.args[offset..], &call.named, false)
}

pub(super) fn lower_function(m: &mut Module, args: &[NodeId], member: &str) -> Option<NodeId> {
    match member {
        // normsys_factor(lo, hi, alpha) = hep.interp_poly6_exp(lo, 1.0, hi, alpha)
        "normsys_factor" => {
            let [lo, hi, alpha] = arity::<3>(args)?;
            let one = lit(m, 1.0);
            Some(interp_poly6_exp(m, [lo, one, hi, alpha]))
        }
        // Center the reference at zero: do not add then subtract a large nominal.
        "histosys_shift" => {
            let [lo, nominal, hi, alpha] = arity::<4>(args)?;
            let lo = sub(m, lo, nominal);
            let hi = sub(m, hi, nominal);
            let zero = lit(m, 0.0);
            Some(interp_poly6_lin(m, [lo, zero, hi, alpha]))
        }
        "sample_yields" => {
            let [nominal, shifts, factors] = arity::<3>(args)?;
            let [sample, modifier, bin] = axes(m);
            let sums = aggregate(m, "sum", shifts, &[sample, modifier, bin], &[sample, bin]);
            let products = aggregate(m, "prod", factors, &[sample, modifier, bin], &[sample, bin]);
            let add = m.intern("add");
            let add = m.alloc(Node::Const(add));
            let shifted = build_call(m, "broadcast", &[add, nominal, sums]);
            let mul = m.intern("mul");
            let mul = m.alloc(Node::Const(mul));
            Some(build_call(m, "broadcast", &[mul, shifted, products]))
        }
        "expected_counts" => {
            let [samples] = arity::<1>(args)?;
            let [sample, _, bin] = axes(m);
            Some(aggregate(m, "sum", samples, &[sample, bin], &[bin]))
        }
        _ => None,
    }
}

/// Fresh labels keep model axes separate from caller axes.
fn axes(m: &mut Module) -> [NodeId; 3] {
    let scope = m.node_count();
    ["sample", "modifier", "bin"].map(|label| {
        let name = m.intern(&format!("pyhf_{scope}_{label}"));
        m.alloc(Node::Axis(Axis {
            name,
            variance: None,
        }))
    })
}

fn aggregate(
    m: &mut Module,
    reducer: &str,
    input: NodeId,
    axes: &[NodeId],
    output: &[NodeId],
) -> NodeId {
    let mut indices = vec![input];
    indices.extend_from_slice(axes);
    let body = build_call(m, "get", &indices);
    let one = lit(m, 1.0);
    let body = divide(m, body, one);
    let output = build_call(m, "vector", output);
    let reducer = m.intern(reducer);
    let reducer = m.alloc(Node::Const(reducer));
    build_call(m, "aggregate", &[reducer, output, body])
}
