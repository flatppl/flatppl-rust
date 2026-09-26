//! Interpolate scalar normalization effects together without changing products.

use std::collections::{BTreeMap, BTreeSet};

use flatppl_core::{Node, NodeId, Ref, RefNs, Symbol};

use crate::builder::Builder;
use crate::histfactory::Multiplier;

pub(crate) const PRODUCT_THRESHOLD: usize = 96;

#[derive(Clone, Copy)]
pub(crate) struct Interpolation {
    pub function: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub param: Symbol,
}

type Key = (&'static str, Symbol, u64, u64);

impl Interpolation {
    fn key(self) -> Key {
        (
            self.function,
            self.param,
            self.lo.to_bits(),
            self.hi.to_bits(),
        )
    }

    fn inputs(self, b: &mut Builder) -> [NodeId; 3] {
        let alpha = b.m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: self.param,
        }));
        [b.lit_real(self.lo), b.lit_real(self.hi), alpha]
    }
}

/// Module-local reuse keeps shared nuisances out of repeated tensor lanes.
#[derive(Default)]
pub(crate) struct Factors {
    values: BTreeMap<Key, NodeId>,
}

impl Factors {
    /// Preserve occurrence and product order; batch only new distinct factors.
    pub(crate) fn multipliers(
        &mut self,
        b: &mut Builder,
        channel: &str,
        samples: Vec<Vec<Multiplier>>,
    ) -> Vec<Vec<NodeId>> {
        let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut pending = BTreeSet::new();
        for factor in samples.iter().flatten() {
            if let Multiplier::Interpolated(interpolation) = factor {
                let key = interpolation.key();
                if !self.values.contains_key(&key) && pending.insert(key) {
                    groups
                        .entry(interpolation.function)
                        .or_default()
                        .push(*interpolation);
                }
            }
        }

        let one = b.lit_real(1.0);
        for (function, rows) in groups {
            let inputs: Vec<_> = rows.iter().map(|row| row.inputs(b)).collect();
            if inputs.len() == 1 {
                let [lo, hi, alpha] = inputs[0];
                let factor = b.module_user_call("hepphys", function, &[lo, one, hi, alpha]);
                self.values.insert(rows[0].key(), factor);
                continue;
            }
            // Native HS3 can mix interpolation kinds, so each has its own batch.
            let lo = b.array(&inputs.iter().map(|row| row[0]).collect::<Vec<_>>());
            let hi = b.array(&inputs.iter().map(|row| row[1]).collect::<Vec<_>>());
            let alpha = b.array(&inputs.iter().map(|row| row[2]).collect::<Vec<_>>());
            let head = b.module_call("hepphys", function);
            let factors = b.call("broadcast", &[head, lo, one, hi, alpha]);
            let name = b.bind_unique_doc(
                &format!("{channel}_normsys_{function}"),
                factors,
                "Distinct normalization factors.",
            );
            let factors = b.self_ref(&name);
            for (offset, row) in rows.into_iter().enumerate() {
                let offset = b.lit_int(offset as i64 + 1);
                let factor = b.call("get", &[factors, offset]);
                self.values.insert(row.key(), factor);
            }
        }

        samples
            .into_iter()
            .map(|factors| {
                factors
                    .into_iter()
                    .map(|factor| match factor {
                        Multiplier::Value(value) => value,
                        Multiplier::Interpolated(interpolation) => {
                            self.values[&interpolation.key()]
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Reduce equal-length scalar runs together, retaining factor order and uses.
/// Small or unpaired runs keep the caller's original multiplication chain.
pub(crate) fn products(
    b: &mut Builder,
    channel: &str,
    runs: impl IntoIterator<Item = Vec<NodeId>>,
) -> BTreeMap<Vec<NodeId>, NodeId> {
    let mut groups: BTreeMap<usize, BTreeSet<Vec<NodeId>>> = BTreeMap::new();
    for run in runs
        .into_iter()
        .filter(|run| run.len() >= PRODUCT_THRESHOLD)
    {
        groups.entry(run.len()).or_default().insert(run);
    }
    let mut products = BTreeMap::new();
    for (length, rows) in groups {
        if rows.len() < 2 {
            continue;
        }
        let arrays: Vec<_> = rows.iter().map(|row| b.array(row)).collect();
        let arrays = b.array(&arrays);
        let prod = b.call_head("prod");
        let values = b.call("broadcast", &[prod, arrays]);
        let name = b.bind_unique_doc(
            &format!("{channel}_scalar_products_{length}"),
            values,
            "Products of equal-length scalar modifier runs.",
        );
        let values = b.self_ref(&name);
        for (index, row) in rows.into_iter().enumerate() {
            let index = b.lit_int(index as i64 + 1);
            products.insert(row, b.call("get", &[values, index]));
        }
    }
    products
}
