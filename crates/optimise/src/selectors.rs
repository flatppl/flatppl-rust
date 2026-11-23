//! Bounds, uniqueness and use counts for fixed gather indices.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::Dtype;

use super::engine::{Fact, FactArg, Term};

#[derive(Default)]
pub(super) struct Selectors {
    literals: HashMap<usize, Rc<Vec<i64>>>,
    synthetic: HashMap<usize, Rc<Vec<i64>>>,
}

impl Selectors {
    pub(super) fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub(super) fn insert(&mut self, term: usize, rows: Vec<i64>) {
        self.literals.insert(term, Rc::new(rows));
    }

    pub(super) fn synthetic(&self, key: usize) -> Option<&[i64]> {
        self.synthetic.get(&key).map(|rows| rows.as_slice())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Base {
    One,
    Zero,
}

impl Base {
    fn head(self) -> &'static str {
        match self {
            Self::One => "get",
            Self::Zero => "get0",
        }
    }

    fn offset(self) -> i64 {
        match self {
            Self::One => 1,
            Self::Zero => 0,
        }
    }
}

/// Keep one copy of the authored integers even when a selector is used in both bases.
#[derive(Clone)]
struct Rows {
    values: Rc<Vec<i64>>,
    base: Base,
}

impl Rows {
    fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.values.iter().map(|&row| row - self.base.offset())
    }
}

impl PartialEq for Rows {
    fn eq(&self, other: &Self) -> bool {
        self.values.len() == other.values.len() && self.iter().eq(other.iter())
    }
}

impl Eq for Rows {}

impl Hash for Rows {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.values.len().hash(state);
        for row in self.iter() {
            row.hash(state);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Index {
    base: Base,
    term: usize,
}

struct Entry {
    rows: Rows,
    representative: Index,
    lo: i64,
    hi: i64,
    unique: bool,
    identity: bool,
}

#[derive(Default)]
struct Catalogue {
    by_rows: HashMap<Rows, usize>,
    entries: Vec<Entry>,
}

impl Catalogue {
    fn intern(&mut self, rows: Rows, index: Index, facts: &mut Vec<Fact>) -> usize {
        use FactArg::{Bool, Expr, Int, Str};

        let id = *self.by_rows.entry(rows.clone()).or_insert_with(|| {
            let id = self.entries.len();
            let mut seen = HashSet::new();
            let mut lo = i64::MAX;
            let mut hi = 0;
            let mut identity = true;
            for (position, row) in rows.iter().enumerate() {
                lo = lo.min(row);
                hi = hi.max(row);
                identity &= position as i64 == row;
                seen.insert(row);
            }
            let unique = seen.len() == rows.values.len();
            self.entries.push(Entry {
                rows,
                representative: index,
                lo,
                hi,
                unique,
                identity,
            });
            facts.push(Fact {
                name: "selector-representative",
                args: vec![Int(id as i64), Str(index.base.head()), Expr(index.term)],
            });
            id
        });
        let entry = &self.entries[id];
        facts.push(Fact {
            name: "fixed-selector",
            args: vec![
                Str(index.base.head()),
                Expr(index.term),
                Int(entry.rows.values.len() as i64),
                Int(entry.lo),
                Int(entry.hi),
                Int(id as i64),
                Bool(entry.unique),
                Bool(entry.identity),
            ],
        });
        id
    }
}

#[derive(Clone, Copy)]
struct Gather {
    source: usize,
    index: Index,
    rows: usize,
}

fn gather_args(term: &Term) -> Option<(usize, Index)> {
    let Term::Call(head, args) = term else {
        return None;
    };
    let &[source, term] = args.as_slice() else {
        return None;
    };
    let base = match head.as_str() {
        "get" => Base::One,
        "get0" => Base::Zero,
        _ => return None,
    };
    Some((source, Index { base, term }))
}

fn neg_arg(term: &Term) -> Option<usize> {
    if let Term::Call(head, args) = term
        && head == "neg"
        && let &[arg] = args.as_slice()
    {
        return Some(arg);
    }
    None
}

fn gather_fact(name: &'static str, source: usize, index: Index, facts: &mut Vec<Fact>) {
    facts.push(Fact {
        name,
        args: vec![
            FactArg::Str(index.base.head()),
            FactArg::Expr(source),
            FactArg::Expr(index.term),
        ],
    });
}

/// Seed proofs from projected literal selectors and use counts from the full query DAG.
pub(super) fn seed(
    selectors: &mut Selectors,
    terms: &mut Vec<Term>,
    facts: &mut Vec<Fact>,
    uses: &[usize],
    next_leaf: usize,
    dtype: Dtype,
) {
    if selectors.literals.is_empty() {
        return;
    }
    let widths: HashMap<_, _> = facts
        .iter()
        .filter_map(|fact| match (fact.name, fact.args.as_slice()) {
            ("real-vector", [FactArg::Expr(term), FactArg::Int(width)]) => Some((*term, *width)),
            _ => None,
        })
        .collect();
    let raw_max = match dtype {
        Dtype::F32 => i64::from(i32::MAX),
        Dtype::F64 => i64::MAX,
    };
    let exclusive = |id| uses.get(id) == Some(&1);
    let mut catalogue = Catalogue::default();
    let mut original = HashMap::new();
    let mut gathers = vec![None; terms.len()];
    for (term_id, term) in terms.iter().enumerate() {
        let Some((source, index)) = gather_args(term) else {
            continue;
        };
        let (Some(&width), Some(&selected)) = (widths.get(&source), widths.get(&term_id)) else {
            continue;
        };
        let Some(rows) = *original.entry((index.base, index.term)).or_insert_with(|| {
            let values = selectors.literals.get(&index.term)?;
            if values.is_empty()
                || values
                    .iter()
                    .any(|&v| v < index.base.offset() || v > raw_max)
            {
                return None;
            }
            let rows = Rows {
                values: Rc::clone(values),
                base: index.base,
            };
            Some(catalogue.intern(rows, index, facts))
        }) else {
            continue;
        };
        let entry = &catalogue.entries[rows];
        if entry.hi >= width || entry.rows.values.len() as i64 != selected {
            continue;
        }
        gathers[term_id] = Some(Gather {
            source,
            index,
            rows,
        });
        if entry.unique && exclusive(source) {
            if entry.identity || entry.representative != index {
                gather_fact("exclusive-gather-source", source, index, facts);
            }
            if let Some(x) = neg_arg(&terms[source])
                && exclusive(x)
                && widths.get(&x) == Some(&width)
            {
                gather_fact("commutable-negation", x, index, facts);
            }
        }
    }
    for (term_id, term) in terms.iter().enumerate() {
        if let Some(subset) = neg_arg(term)
            && let Some(gather) = gathers[subset]
            && catalogue.entries[gather.rows].unique
            && exclusive(subset)
            && exclusive(gather.source)
            && widths.get(&term_id) == widths.get(&subset)
        {
            gather_fact("commutable-negation", gather.source, gather.index, facts);
        }
    }

    // Reuse computed pairs across the query. Bound composition work and
    // additional storage by the original literal payload.
    let mut budget: usize = selectors.literals.values().map(|rows| rows.len()).sum();
    let mut compositions = HashMap::new();
    for outer in gathers.iter().flatten() {
        let Some(inner) = gathers[outer.source] else {
            continue;
        };
        if !exclusive(outer.source) || !exclusive(inner.source) {
            continue;
        }
        let Some(rows) = *compositions
            .entry((inner.rows, outer.rows))
            .or_insert_with(|| {
                let inner = &catalogue.entries[inner.rows];
                let outer = &catalogue.entries[outer.rows];
                if !outer.unique {
                    return None;
                }
                budget = budget.checked_sub(outer.rows.values.len())?;
                let mut seen = HashSet::new();
                let values = outer
                    .rows
                    .iter()
                    .map(|position| {
                        let row =
                            inner.rows.values.get(position as usize)? - inner.rows.base.offset();
                        seen.insert(row).then_some(row)
                    })
                    .collect::<Option<Vec<_>>>()?;
                let rows = Rows {
                    values: Rc::new(values),
                    base: Base::Zero,
                };
                if let Some(&id) = catalogue.by_rows.get(&rows) {
                    return Some(id);
                }
                let key = next_leaf + selectors.synthetic.len();
                let term = terms.len();
                terms.push(Term::Leaf(key));
                selectors.synthetic.insert(key, Rc::clone(&rows.values));
                Some(catalogue.intern(
                    rows,
                    Index {
                        base: Base::Zero,
                        term,
                    },
                    facts,
                ))
            })
        else {
            continue;
        };
        use FactArg::{Expr, Int, Str};
        facts.push(Fact {
            name: "composed-selector",
            args: vec![
                Expr(inner.source),
                Str(inner.index.base.head()),
                Expr(inner.index.term),
                Str(outer.index.base.head()),
                Expr(outer.index.term),
                Int(rows as i64),
            ],
        });
        let composed = &catalogue.entries[rows];
        let index = composed.representative;
        if composed.identity {
            gather_fact("exclusive-gather-source", inner.source, index, facts);
        }
        if let Some(x) = neg_arg(&terms[inner.source])
            && exclusive(x)
            && widths.get(&x) == widths.get(&inner.source)
        {
            gather_fact("commutable-negation", x, index, facts);
        }
    }
}
