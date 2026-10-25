//! Per-binding metadata for hosts: kind, domain, shape, source text and edges.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    BindingId, CallHead, Dim, Inputs, Module, Node, NodeId, Ref, RefNs, Symbol, Type, ValueSet,
};
use serde::Serialize;

/// The role of a binding, read from the call on its right-hand side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingKind {
    /// `elementof(set)`: a value the query supplies.
    Parameter,
    /// `external(set)` or `load_data(...)`: data fixed before compiling.
    Observation,
    /// `x ~ measure`: a draw from a measure.
    Draw,
    /// Anything else, including values bound by `Context::set`.
    Derived,
}

/// A value set as JSON, written from the inferred [`ValueSet`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "set", rename_all = "lowercase")]
pub enum Domain {
    Reals,
    PosReals,
    NonNegReals,
    UnitInterval,
    Integers,
    PosIntegers,
    NonNegIntegers,
    Booleans,
    Complexes,
    RngStates,
    Anything,
    Interval {
        lo: Bound,
        hi: Bound,
    },
    StdSimplex {
        n: Option<u32>,
    },
    Ordered {
        n: Option<u32>,
    },
    ZeroSum {
        n: Option<u32>,
    },
    /// Nested per-axis powers flatten into one shape. `None` is a dynamic extent.
    CartPow {
        of: Box<Domain>,
        shape: Vec<Option<u32>>,
    },
    CartProd {
        of: Vec<Domain>,
    },
    Record {
        fields: Vec<DomainField>,
    },
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DomainField {
    pub name: String,
    pub set: Domain,
}

/// An interval bound. JSON has no infinity, so the infinite bounds are strings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bound(pub f64);

impl Serialize for Bound {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            f64::INFINITY => serializer.serialize_str("inf"),
            f64::NEG_INFINITY => serializer.serialize_str("-inf"),
            value => serializer.serialize_f64(value),
        }
    }
}

impl Domain {
    pub(crate) fn of(module: &Module, set: &ValueSet) -> Self {
        match set {
            ValueSet::Reals => Self::Reals,
            ValueSet::PosReals => Self::PosReals,
            ValueSet::NonNegReals => Self::NonNegReals,
            ValueSet::UnitInterval => Self::UnitInterval,
            ValueSet::Integers => Self::Integers,
            ValueSet::PosIntegers => Self::PosIntegers,
            ValueSet::NonNegIntegers => Self::NonNegIntegers,
            ValueSet::Booleans => Self::Booleans,
            ValueSet::Complexes => Self::Complexes,
            ValueSet::RngStates => Self::RngStates,
            ValueSet::Anything => Self::Anything,
            ValueSet::Interval(lo, hi) => Self::Interval {
                lo: Bound(*lo),
                hi: Bound(*hi),
            },
            ValueSet::StdSimplex(n) => Self::StdSimplex { n: extent(*n) },
            ValueSet::Ordered(n) => Self::Ordered { n: extent(*n) },
            ValueSet::ZeroSum(n) => Self::ZeroSum { n: extent(*n) },
            ValueSet::CartPow(elem, n) => {
                let mut shape = vec![extent(*n)];
                let mut elem = elem.as_ref();
                while let ValueSet::CartPow(inner, n) = elem {
                    shape.push(extent(*n));
                    elem = inner;
                }
                Self::CartPow {
                    of: Box::new(Self::of(module, elem)),
                    shape,
                }
            }
            ValueSet::CartProd(parts) => Self::CartProd {
                of: parts.iter().map(|s| Self::of(module, s)).collect(),
            },
            ValueSet::RecordSet(fields) => Self::Record {
                fields: fields
                    .iter()
                    .map(|(name, set)| DomainField {
                        name: module.resolve(*name).into(),
                        set: Self::of(module, set),
                    })
                    .collect(),
            },
            ValueSet::Deferred | ValueSet::Unknown => Self::Unknown,
        }
    }
}

/// One input of a kernel or likelihood interface.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InputInfo {
    /// The interface name, which a boundary keyword can rename.
    pub name: String,
    /// The support of the value bound to the input.
    pub domain: Domain,
}

/// The interface inputs of a kernel or likelihood binding, in interface
/// order, each with the support of the value its boundary binds. `None` for
/// any other binding.
pub(crate) fn inputs(module: &Module, rhs: NodeId) -> Option<Vec<InputInfo>> {
    let names = match module.type_of(rhs)? {
        Type::Kernel { inputs, .. } | Type::Likelihood { inputs, .. } => inputs,
        _ => return None,
    };
    let mut bound = HashMap::new();
    collect_inputs(module, rhs, &mut bound, &mut HashSet::new());
    Some(
        names
            .iter()
            .map(|name| InputInfo {
                name: module.resolve(*name).into(),
                domain: Domain::of(module, bound.get(name).unwrap_or(&ValueSet::Unknown)),
            })
            .collect(),
    )
}

/// Every reification boundary that `node` composes, as interface name to the
/// bound value's set. A likelihood or a combined kernel reaches them through
/// its argument kernels; the first binding of a name wins.
fn collect_inputs(
    module: &Module,
    node: NodeId,
    bound: &mut HashMap<Symbol, ValueSet>,
    seen: &mut HashSet<BindingId>,
) {
    match module.node(node) {
        Node::Ref(r) if r.ns == RefNs::SelfMod => {
            if let Some(binding) = module.binding_by_name(r.name)
                && seen.insert(binding)
            {
                collect_inputs(module, module.binding(binding).rhs, bound, seen);
            }
        }
        Node::Call(call) => {
            let entries = match &call.inputs {
                Some(Inputs::Spec(entries)) => Some(entries.as_ref()),
                Some(Inputs::Auto) => module.auto_inputs_of(node),
                None => None,
            };
            if let Some(entries) = entries {
                for (name, source) in entries {
                    bound
                        .entry(*name)
                        .or_insert_with(|| bound_set(module, source));
                }
                return;
            }
            for &arg in call.args.iter() {
                collect_inputs(module, arg, bound, seen);
            }
            for named in call.named.iter() {
                collect_inputs(module, named.value, bound, seen);
            }
        }
        _ => {}
    }
}

/// The set of the binding a boundary entry names; a placeholder has none.
fn bound_set(module: &Module, source: &Ref) -> ValueSet {
    if source.ns != RefNs::SelfMod {
        return ValueSet::Unknown;
    }
    module
        .binding_by_name(source.name)
        .and_then(|binding| module.valueset_of(module.binding(binding).rhs))
        .cloned()
        .unwrap_or(ValueSet::Unknown)
}

fn extent(dim: Dim) -> Option<u32> {
    match dim {
        Dim::Static(n) => Some(n),
        Dim::Dynamic => None,
    }
}

pub(crate) fn kind(module: &Module, rhs: NodeId) -> BindingKind {
    let Node::Call(call) = module.node(rhs) else {
        return BindingKind::Derived;
    };
    let CallHead::Builtin(head) = call.head else {
        return BindingKind::Derived;
    };
    match module.resolve(head) {
        "elementof" => BindingKind::Parameter,
        "external" | "load_data" => BindingKind::Observation,
        "draw" => BindingKind::Draw,
        _ => BindingKind::Derived,
    }
}

/// The array extents of a value: `[]` for a scalar, the outer shape for an
/// array, and `None` for records, tuples, tables and non-values.
pub(crate) fn shape(ty: &Type) -> Option<Vec<Option<u32>>> {
    match ty {
        Type::Scalar(_) => Some(Vec::new()),
        Type::Array { shape, .. } => Some(shape.iter().map(|&d| extent(d)).collect()),
        Type::TVector { len, .. } => Some(vec![extent(*len)]),
        _ => None,
    }
}

/// The number of scalar elements in a value, or `None` when an extent is
/// dynamic or the type is not a value.
pub(crate) fn size(ty: &Type) -> Option<u64> {
    let dims = |dims: &[Dim]| {
        dims.iter().try_fold(1_u64, |n, &d| match d {
            Dim::Static(d) => n.checked_mul(d.into()),
            Dim::Dynamic => None,
        })
    };
    match ty {
        Type::Scalar(_) => Some(1),
        Type::Array { shape, elem } => dims(shape)?.checked_mul(size(elem)?),
        Type::TVector { len, elem } => dims(&[*len])?.checked_mul(size(elem)?),
        Type::Record(fields) => total(fields.iter().map(|(_, t)| t)),
        Type::Tuple(items) => total(items.iter()),
        Type::Table { columns, nrows } => {
            dims(&[*nrows])?.checked_mul(total(columns.iter().map(|(_, t)| t))?)
        }
        _ => None,
    }
}

fn total<'a>(mut types: impl Iterator<Item = &'a Type>) -> Option<u64> {
    types.try_fold(0_u64, |n, t| n.checked_add(size(t)?))
}

/// The public bindings that `binding` references directly, in source order.
/// Private and synthetic intermediates are followed through to the public
/// bindings they reference.
pub(crate) fn dependencies(module: &Module, binding: BindingId) -> Vec<String> {
    let own = module.binding(binding).name;
    let mut found = HashSet::new();
    let mut visited = HashSet::from([own]);
    let mut pending = vec![module.binding(binding).rhs];
    while let Some(rhs) = pending.pop() {
        let mut names: HashSet<Symbol> = HashSet::new();
        module.collect_referenced_names(rhs, &mut names);
        for name in names {
            if !visited.insert(name) {
                continue;
            }
            match module.binding_by_name(name).map(|id| module.binding(id)) {
                Some(b) if b.public => {
                    found.insert(name);
                }
                Some(b) => pending.push(b.rhs),
                None => {}
            }
        }
    }
    module
        .public_bindings()
        .filter(|(_, b)| found.contains(&b.name))
        .map(|(_, b)| module.resolve(b.name).to_string())
        .collect()
}

/// Every binding that `roots` reach, transitively and including private ones.
pub(crate) fn reachable(module: &Module, roots: &[Symbol]) -> HashSet<BindingId> {
    let mut keep = HashSet::new();
    let mut work: Vec<_> = roots
        .iter()
        .filter_map(|&r| module.binding_by_name(r))
        .collect();
    keep.extend(work.iter().copied());
    while let Some(id) = work.pop() {
        let mut names = HashSet::new();
        module.collect_referenced_names(module.binding(id).rhs, &mut names);
        for name in names {
            if let Some(dep) = module.binding_by_name(name)
                && keep.insert(dep)
            {
                work.push(dep);
            }
        }
    }
    keep
}

/// A byte offset as a 1-based line and 1-based column in Unicode characters,
/// with the text of that line.
pub(crate) fn locate(text: &str, offset: u32) -> Option<(u32, u32, String)> {
    let offset = offset as usize;
    if offset > text.len() || !text.is_char_boundary(offset) {
        return None;
    }
    let start = text[..offset].rfind('\n').map_or(0, |i| i + 1);
    let end = text[offset..].find('\n').map_or(text.len(), |i| offset + i);
    let line = text[..start].matches('\n').count() + 1;
    let column = text[start..offset].chars().count() + 1;
    let excerpt = text[start..end].trim_end_matches('\r').to_string();
    Some((
        u32::try_from(line).ok()?,
        u32::try_from(column).ok()?,
        excerpt,
    ))
}
