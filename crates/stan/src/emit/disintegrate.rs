//! `disintegrate(selector, J)` and `restrict(J, x)` along observed leaves
//! (spec §06 "Structural disintegration", "Measure restriction").
//!
//! For `J = lawof(record(f = v, …))` whose selected fields are leaves (no
//! other field's law reads one, and every draw a field reads is a field), the
//! marginal is the joint law of the other fields and the kernel is the
//! selected fields' laws given them: μ(A × B) = ∫_A κ(a, B) dν(a). By the
//! selector form of §06 "Measure restriction", `restrict(J, x)` is
//! `bayesupdate(likelihoodof(kernel, x), marginal)` with the split along the
//! fields of x, so both reach the same split.

use super::Emitter;
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, Symbol, Type};
use std::collections::HashSet;

/// A record joint split along observed leaves.
pub(super) struct Split {
    /// The selected fields with their values: the kernel's variates.
    pub selected: Vec<(Symbol, NodeId)>,
    /// The other fields: the marginal's variates, the kernel's inputs.
    pub complement: Vec<(Symbol, NodeId)>,
}

impl<'m> Emitter<'m> {
    /// `get(disintegrate(sel, J), part)` (the tuple `kernel, marginal`):
    /// the `get` node, the selector and J. A refusal sits at the `get`,
    /// whose binding is the name the user wrote (`K, prior = …`), not the
    /// tuple's hidden binding.
    pub(super) fn disintegrate_part(
        &self,
        id: NodeId,
        part: i64,
    ) -> Option<(NodeId, NodeId, NodeId)> {
        let id = self.resolve(id);
        if self.head(id) != Some("get") {
            return None;
        }
        let [d, k] = self.call(id)?.args[..] else {
            return None;
        };
        let d = self.resolve(d);
        if self.head(d) != Some("disintegrate") {
            return None;
        }
        if !matches!(self.m.node(self.resolve(k)), Node::Lit(Scalar::Int(i)) if *i == part) {
            return None;
        }
        let [sel, joint] = self.call(d)?.args[..] else {
            return None;
        };
        Some((id, sel, joint))
    }

    /// The field names a selector `["a", "b"]` names.
    fn selector_names(&self, at: NodeId, sel: NodeId) -> Result<Vec<String>> {
        let sel = self.resolve(sel);
        if matches!(self.m.node(sel), Node::Lit(Scalar::Str(_))) {
            return Err(Refusal::not_yet(
                Some(at),
                "a bare-string `disintegrate` selector (a bare kernel variate)",
            ));
        }
        let items = match self.head(sel) {
            Some("vector") => self.call(sel).unwrap().args.to_vec(),
            _ => {
                return Err(Refusal::not_yet(
                    Some(at),
                    "a `disintegrate` selector that is not a list of field names",
                ));
            }
        };
        let mut names: Vec<String> = Vec::new();
        for i in items {
            match self.m.node(self.resolve(i)) {
                // The kernel variate is `record(…)` of the selected fields (§06,
                // selectors as in `get`), and a record holds each field once (§03).
                Node::Lit(Scalar::Str(s)) if names.iter().any(|n| *n == **s) => {
                    return Err(Refusal::new(
                        RefusalKind::IllFormedModule,
                        Some(at),
                        format!("`disintegrate` selector field `{s}`"),
                        format!(
                            "the `disintegrate` selector names the field `{s}` twice; the kernel's \
                             variate is a record, which holds each field once (§03)"
                        ),
                    ));
                }
                Node::Lit(Scalar::Str(s)) => names.push(s.to_string()),
                _ => {
                    return Err(Refusal::not_yet(
                        Some(at),
                        "a positional `disintegrate` selector",
                    ));
                }
            }
        }
        Ok(names)
    }

    /// `disintegrate(sel, J)` split along the selected fields; the strict
    /// structural test refuses anything else.
    pub(super) fn leaf_split(&self, at: NodeId, sel: NodeId, joint: NodeId) -> Result<Split> {
        let rec = self.record_joint(at, joint, "disintegrate")?;
        let names = self.selector_names(at, sel)?;
        self.split_record(at, rec, &names, "disintegrate")
    }

    /// `restrict(J, x)`: the split along the fields of x, and x.
    pub(super) fn restrict_split(&self, r: NodeId) -> Result<(Split, NodeId)> {
        let [joint, x] = self.call(r).unwrap().args[..] else {
            return Err(Refusal::not_yet(Some(r), "`restrict` with this arity"));
        };
        let point = self.resolve(x);
        if matches!(self.m.type_of(point), Some(Type::Table { .. })) {
            return Err(Refusal::not_yet(Some(r), "a `restrict` to a table point"));
        }
        if self.head(point) != Some("record") {
            return Err(Refusal::not_yet(
                Some(r),
                "a `restrict` to a point that is not a record",
            ));
        }
        let names: Vec<String> = self
            .call(point)
            .unwrap()
            .named
            .iter()
            .map(|na| self.name(na.name).to_string())
            .collect();
        let rec = self.record_joint(r, joint, "restrict")?;
        let split = self.split_record(r, rec, &names, "restrict")?;
        // §06 "Measure restriction": the point is a variate of the selected
        // fields, as `likelihoodof` data is of its kernel.
        for na in self.call(point).unwrap().named.iter() {
            let Some(&(_, v)) = split.selected.iter().find(|(f, _)| *f == na.name) else {
                continue;
            };
            // A point with no usable type (`mean(M)` is `%deferred`) is left
            // to the path that writes it, whose refusal is accurate.
            let (Some(want), Some(have)) = (self.m.type_of(v), self.m.type_of(na.value)) else {
                continue;
            };
            let (Some(want_table), Some(have_table)) = (holds_table(want), holds_table(have))
            else {
                continue;
            };
            let f = self.name(na.name);
            if want_table || have_table {
                return Err(Refusal::not_yet(
                    Some(na.value),
                    format!("a `restrict` point field `{f}` that is a table"),
                ));
            }
            if shapes_differ(want, have) {
                return Err(Refusal::new(
                    RefusalKind::IllFormedModule,
                    Some(na.value),
                    format!("`restrict` point field `{f}`"),
                    format!(
                        "the `restrict` point sets `{f}` to {}, but the field `{f}` of the joint \
                         has variates of type {}; a `restrict` point is a variate of the fields \
                         it names (§06 \"Measure restriction\")",
                        show_type(self.m, have),
                        show_type(self.m, want)
                    ),
                ));
            }
        }
        Ok((split, x))
    }

    /// The record of `J = lawof(record(…))`. `op` is the construct the user
    /// wrote, for the refusal text.
    fn record_joint(&self, at: NodeId, joint: NodeId, op: &str) -> Result<NodeId> {
        let j = self.resolve(joint);
        match self.head(j) {
            Some("lawof") => {
                let rec = self.resolve(self.call(j).unwrap().args[0]);
                if self.head(rec) == Some("record") && !self.call(rec).unwrap().named.is_empty() {
                    return Ok(rec);
                }
                Err(Refusal::not_yet(
                    Some(at),
                    format!("a `{op}` of a `lawof` that is not of a named record"),
                ))
            }
            Some("jointchain") => Err(Refusal::not_yet(
                Some(at),
                format!("a `{op}` of a `jointchain`"),
            )),
            _ => Err(Refusal::not_yet(
                Some(at),
                format!("a `{op}` of a measure that is not `lawof(record(…))`"),
            )),
        }
    }

    /// The strict leaf test: each field is its own draw, every draw a field's
    /// law reads is a field, and no other field's law reads a selected one.
    fn split_record(&self, at: NodeId, rec: NodeId, names: &[String], op: &str) -> Result<Split> {
        let picked = |f: Symbol| names.iter().any(|n| n == self.name(f));
        let fields: Vec<(Symbol, NodeId)> = self
            .call(rec)
            .unwrap()
            .named
            .iter()
            .map(|na| (na.name, na.value))
            .collect();
        if let Some(n) = names
            .iter()
            .find(|n| !fields.iter().any(|(f, _)| self.name(*f) == n.as_str()))
        {
            let what = if op == "restrict" {
                "point"
            } else {
                "selector"
            };
            return Err(Refusal::not_yet(
                Some(at),
                format!("a `{op}` {what} field `{n}` that is not a field of the joint"),
            ));
        }
        // Each field's draw, and the draws its law reads.
        let mut draws: Vec<(Symbol, Symbol, HashSet<Symbol>)> = Vec::new();
        for &(f, v) in &fields {
            let Ok((Some(draw), law)) = self.output_law(v) else {
                return Err(Refusal::not_yet(
                    Some(at),
                    format!(
                        "a `{op}` of a joint whose field `{}` is not a draw",
                        self.name(f)
                    ),
                ));
            };
            if draws.iter().any(|d| d.1 == draw) {
                return Err(self.singular_joint(at, draw));
            }
            draws.push((f, draw, self.draws_read(law)));
        }
        let field_draws: HashSet<Symbol> = draws.iter().map(|d| d.1).collect();
        for (_, _, read) in &draws {
            if let Some(latent) = read.iter().find(|d| !field_draws.contains(d)) {
                return Err(Refusal::not_yet(
                    Some(at),
                    format!(
                        "a `{op}` of a joint with the latent draw `{}` (the marginal integrates it out)",
                        self.name(*latent)
                    ),
                ));
            }
        }
        let selected: HashSet<Symbol> = draws.iter().filter(|d| picked(d.0)).map(|d| d.1).collect();
        for (f, _, read) in draws.iter().filter(|d| !picked(d.0)) {
            if read.iter().any(|d| selected.contains(d)) {
                let text = if op == "restrict" {
                    format!(
                        "a `restrict` to fields that the field `{}` depends on (the complement form)",
                        self.name(*f)
                    )
                } else {
                    format!(
                        "a `disintegrate` along fields that the field `{}` depends on (the marginal integrates them out)",
                        self.name(*f)
                    )
                };
                return Err(Refusal::not_yet(Some(at), text));
            }
        }
        let (selected, complement) = fields.into_iter().partition(|(f, _)| picked(*f));
        Ok(Split {
            selected,
            complement,
        })
    }

    /// The draws `law` reads, through bindings and across reification
    /// boundaries as the free-input walk sees them, but not the draws a
    /// reification absorbs (§04 "Phase of the reified law").
    fn draws_read(&self, law: NodeId) -> HashSet<Symbol> {
        self.reached_unabsorbed(law, &HashSet::new())
            .into_iter()
            .filter(|s| {
                self.m
                    .binding_by_name(*s)
                    .is_some_and(|b| self.head(self.m.binding(b).rhs) == Some("draw"))
            })
            .collect()
    }
}

/// A type as a refusal text spells it: `real`, `array of real [3]`.
fn show_type(m: &Module, t: &Type) -> String {
    match t {
        Type::Scalar(k) => k.to_string(),
        Type::Array { shape, elem } => {
            let dims: Vec<String> = shape.iter().map(Dim::to_string).collect();
            format!("array of {} [{}]", show_type(m, elem), dims.join(", "))
        }
        Type::TVector { len, elem } => format!("row vector of {} [{len}]", show_type(m, elem)),
        Type::Record(fields) => {
            let fs: Vec<String> = fields
                .iter()
                .map(|(n, t)| format!("{}: {}", m.resolve(*n), show_type(m, t)))
                .collect();
            format!("record({})", fs.join(", "))
        }
        _ => "a value of another kind".to_string(),
    }
}

/// For a known value type: whether it holds a table. `None` when the type is
/// not known (`%deferred`, `%any`, a type variable) or is no value type.
fn holds_table(t: &Type) -> Option<bool> {
    match t {
        Type::Scalar(_) => Some(false),
        Type::Array { elem, .. } | Type::TVector { elem, .. } => holds_table(elem),
        Type::Record(fields) => {
            let mut out = false;
            for (_, f) in fields.iter() {
                out |= holds_table(f)?;
            }
            Some(out)
        }
        Type::Table { .. } => Some(true),
        _ => None,
    }
}

/// Shape and structure differ: rank, a static extent, record field names.
/// Scalar kinds are not compared, and an unknown extent matches, as
/// inference compares `likelihoodof` data with its kernel's variates.
fn shapes_differ(want: &Type, have: &Type) -> bool {
    use Type::*;
    let extent = |a: &Dim, b: &Dim| matches!((a, b), (Dim::Static(p), Dim::Static(q)) if p != q);
    match (want, have) {
        (Scalar(_), Scalar(_)) => false,
        (Array { shape: a, elem: x }, Array { shape: b, elem: y }) => {
            a.len() != b.len()
                || a.iter().zip(b.iter()).any(|(p, q)| extent(p, q))
                || shapes_differ(x, y)
        }
        (TVector { len: a, elem: x }, TVector { len: b, elem: y }) => {
            extent(a, b) || shapes_differ(x, y)
        }
        (Record(a), Record(b)) => {
            a.len() != b.len()
                || a.iter().any(|(p, t)| match b.iter().find(|(q, _)| q == p) {
                    Some((_, u)) => shapes_differ(t, u),
                    None => true,
                })
        }
        (
            Scalar(_) | Array { .. } | TVector { .. } | Record(_),
            Scalar(_) | Array { .. } | TVector { .. } | Record(_),
        ) => true,
        _ => false,
    }
}
