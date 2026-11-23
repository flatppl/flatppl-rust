//! Backward scan demand for a declared query. All recurrence inputs remain live.

use flatppl_core::query::promoted_input_names;

use super::*;
use crate::RulePolicy;

#[derive(Clone, Default)]
pub struct ScanOutputDemand {
    terminal_reads: HashMap<NodeId, NodeId>,
    histories: HashMap<NodeId, Vec<Symbol>>,
    history_maps: HashMap<NodeId, (NodeId, Symbol)>,
}

impl ScanOutputDemand {
    pub fn terminal_scan(&self, reader: NodeId) -> Option<NodeId> {
        self.terminal_reads.get(&reader).copied()
    }

    pub fn histories(&self, scan: NodeId) -> Option<&[Symbol]> {
        self.histories.get(&scan).map(Vec::as_slice)
    }

    pub fn history_map(&self, reader: NodeId) -> Option<(NodeId, Symbol)> {
        self.history_maps.get(&reader).copied()
    }
}

enum Read {
    Terminal,
    History(Symbol),
    Map(Symbol),
}

fn resolved(module: &Module, mut node: NodeId, abi: &HashSet<NodeId>) -> Option<NodeId> {
    for _ in 0..MAX_PROOF_DEPTH {
        if abi.contains(&node) {
            return None;
        }
        let (next, _) = resolve_ref_one(module, node);
        if next == node {
            return Some(node);
        }
        node = next;
    }
    None
}

fn positional(module: &Module, node: NodeId) -> Option<(&str, &[NodeId])> {
    let (head, call) = builtin(module, node)?;
    (call.named.is_empty() && call.inputs.is_none()).then_some((head, &call.args))
}

fn scan(module: &Module, node: NodeId) -> bool {
    matches!(
        positional(module, node),
        Some(("scan", [_, _, _] | [_, _, _, _]))
    )
}

fn scalar_columns(module: &Module, scan: NodeId) -> Option<&[(Symbol, Type)]> {
    let Type::Table {
        columns,
        nrows: Dim::Static(1..),
        ..
    } = module.type_of(scan)?
    else {
        return None;
    };
    columns
        .iter()
        .all(|(_, ty)| {
            matches!(
                ty,
                Type::Scalar(ScalarType::Real | ScalarType::Integer | ScalarType::Boolean)
            )
        })
        .then_some(columns.as_ref())
}

fn read(module: &Module, reader: NodeId, abi: &HashSet<NodeId>) -> Option<(NodeId, Read)> {
    let (head, [a, b]) = positional(module, reader)? else {
        return None;
    };
    let container = match head {
        "get" | "get0" | "get_field" => *a,
        "broadcast" => *b,
        _ => return None,
    };
    let scan_id = resolved(module, container, abi)?;
    if !scan(module, scan_id) {
        return None;
    }
    let kind = match head {
        "get" | "get0" | "get_field" => match module.node(resolved(module, *b, abi)?) {
            Node::Lit(Scalar::Str(name))
                if matches!(module.node(*b), Node::Lit(Scalar::Str(_))) =>
            {
                let (symbol, _) = scalar_columns(module, scan_id)?
                    .iter()
                    .find(|(symbol, _)| module.resolve(*symbol) == &**name)?;
                Read::History(*symbol)
            }
            Node::Lit(Scalar::Int(index)) if head != "get_field" => {
                let length = match module.type_of(scan_id)? {
                    Type::Array { shape, .. } => shape.first()?,
                    Type::TVector { len, .. } => len,
                    _ => return None,
                };
                let Dim::Static(length @ 1..) = *length else {
                    return None;
                };
                if *index != i64::from(length) - i64::from(head == "get0")
                    || i32::try_from(*index).is_err()
                    || !tensor_value(module.type_of(reader)?)
                {
                    return None;
                }
                Read::Terminal
            }
            _ => return None,
        },
        "broadcast" => {
            let function = resolved(module, *a, abi)?;
            let ("functionof", call) = builtin(module, function)? else {
                return None;
            };
            let Some(Inputs::Spec(entries)) = &call.inputs else {
                return None;
            };
            let ([(_, input)], [body]) = (entries.as_ref(), call.args.as_ref()) else {
                return None;
            };
            let ("get" | "get_field", [object, key]) = positional(module, *body)? else {
                return None;
            };
            let Node::Lit(Scalar::Str(name)) = module.node(*key) else {
                return None;
            };
            if input.ns != RefNs::Local
                || [body, object, key]
                    .into_iter()
                    .any(|node| abi.contains(node))
                || !matches!(module.node(*object), Node::Ref(r) if r == input)
            {
                return None;
            }
            let (symbol, _) = scalar_columns(module, scan_id)?
                .iter()
                .find(|(symbol, _)| module.resolve(*symbol) == &**name)?;
            Read::Map(*symbol)
        }
        _ => return None,
    };
    Some((scan_id, kind))
}

fn tensor_value(mut ty: &Type) -> bool {
    for _ in 0..MAX_PROOF_DEPTH {
        ty = match ty {
            Type::Scalar(ScalarType::Real | ScalarType::Integer | ScalarType::Boolean) => {
                return true;
            }
            Type::Array { shape, elem } if shape.iter().all(|d| matches!(d, Dim::Static(_))) => {
                elem
            }
            Type::TVector {
                len: Dim::Static(_),
                elem,
            } => elem,
            _ => return false,
        };
    }
    false
}

/// Find scan outputs that can be lowered without materializing every history.
/// ABI bindings cut traversal. Unrecognized users and callable captures keep
/// the full value, and mixed terminal/history consumers share the full scan.
pub fn scan_output_demand(module: &Module) -> ScanOutputDemand {
    scan_output_demand_with_rules(module, &RulePolicy::default())
}

/// Find scan output demand using the caller's egglog rules.
pub fn scan_output_demand_with_rules(module: &Module, rules: &RulePolicy) -> ScanOutputDemand {
    if module.node_count() > MAX_DEMAND_NODES || !rules.enables_ruleset("scan-demand") {
        return ScanOutputDemand::default();
    }
    let Some((_, output)) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "outputs")
    else {
        return ScanOutputDemand::default();
    };
    let abi = promoted_input_names(module)
        .into_iter()
        .filter_map(|name| module.binding_by_name(name))
        .map(|binding| module.binding(binding).rhs)
        .collect::<HashSet<_>>();
    let mut result = ScanOutputDemand::default();
    let mut whole = HashSet::new();
    let mut terminal = HashSet::new();
    let mut seen = HashSet::new();
    let mut seen_dependencies = HashSet::new();
    let mut pending = vec![(output.rhs, false)];
    while let Some((node, scoped)) = pending.pop() {
        if abi.contains(&node) || !seen.insert((node, scoped)) {
            continue;
        }
        if !scoped && let Some((scan, kind)) = read(module, node, &abi) {
            match kind {
                Read::Terminal => {
                    terminal.insert(scan);
                    result.terminal_reads.insert(node, scan);
                }
                Read::History(field) | Read::Map(field) => {
                    let fields = result.histories.entry(scan).or_default();
                    if !fields.contains(&field) {
                        fields.push(field);
                    }
                    if matches!(kind, Read::Map(_)) {
                        result.history_maps.insert(node, (scan, field));
                    }
                }
            }
            if seen_dependencies.insert(scan) {
                dependencies(module, scan, |child| pending.push((child, false)));
            }
            continue;
        }
        if scan(module, node) {
            whole.insert(node);
        }
        let scoped =
            scoped || matches!(module.node(node), Node::Call(call) if call.inputs.is_some());
        dependencies(module, node, |child| pending.push((child, scoped)));
    }
    let candidates = terminal
        .iter()
        .copied()
        .chain(result.histories.keys().copied())
        .collect::<HashSet<_>>();
    let summaries = candidates
        .iter()
        .map(|&scan| {
            [
                scan.index() as i64,
                i64::from(whole.contains(&scan)),
                i64::from(terminal.contains(&scan)),
                result.histories.get(&scan).map_or(0, |fields| fields.len()) as i64,
                scalar_columns(module, scan).map_or(0, |columns| columns.len()) as i64,
            ]
        })
        .collect::<Vec<_>>();
    // Skip the catalogue when every candidate escapes or keeps every field.
    // The rules still select the individual plans when there is any saving.
    if !summaries
        .iter()
        .any(|&[_, escapes, terminal, fields, total]| {
            escapes == 0 && (terminal > 0 || fields < total)
        })
    {
        return ScanOutputDemand::default();
    }
    let facts = summaries
        .iter()
        .map(|summary| ("scan-use-summary", summary.as_slice()))
        .collect::<Vec<_>>();
    let Some(plans) = crate::engine::integer_plan_rows(
        "scan_demand.egg",
        include_str!("rules/scan_demand.egg"),
        "scan-demand",
        &facts,
        "scan-output-plan",
        rules,
    ) else {
        return ScanOutputDemand::default();
    };
    let plans = plans
        .iter()
        .filter_map(|plan| match plan.as_slice() {
            [scan, choice] => Some((NodeId::from_usize((*scan).try_into().ok()?), *choice)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    result
        .terminal_reads
        .retain(|_, scan| plans.get(scan) == Some(&0));
    result
        .histories
        .retain(|scan, _| plans.get(scan) == Some(&1));
    result
        .history_maps
        .retain(|_, (scan, _)| result.histories.contains_key(scan));
    result
}
