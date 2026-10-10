//! Interpolate scalar normalization effects together without changing products.

use std::collections::{BTreeMap, BTreeSet};

use flatppl_core::{Node, NodeId, Ref, RefNs, Symbol};

use crate::builder::Builder;
use crate::histfactory::{INTERP_NORMSYS_DEFAULT, Multiplier};

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
    /// Batched factor vectors as (binding name, length).
    vectors: Vec<(String, usize)>,
    /// Vector index and zero-based offset of each batched factor.
    slots: BTreeMap<NodeId, (usize, usize)>,
    /// Binding of each vector with a trailing unit lane, once a run needs padding.
    padded: BTreeMap<usize, String>,
}

impl Factors {
    /// Preserve occurrence and product order; batch only new distinct factors.
    pub(crate) fn multipliers(
        &mut self,
        b: &mut Builder,
        channel: &str,
        samples: Vec<Vec<Multiplier>>,
        pyhf_helpers: Option<&str>,
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
            let helper = pyhf_helpers.filter(|_| function == INTERP_NORMSYS_DEFAULT);
            let inputs: Vec<_> = rows.iter().map(|row| row.inputs(b)).collect();
            if inputs.len() == 1 {
                let [lo, hi, alpha] = inputs[0];
                let factor = match helper {
                    Some(alias) => b.module_user_call(alias, "normsys_factor", &[lo, hi, alpha]),
                    None => b.module_user_call("hepphys", function, &[lo, one, hi, alpha]),
                };
                self.values.insert(rows[0].key(), factor);
                continue;
            }
            // Native HS3 can mix interpolation kinds, so each has its own batch.
            let lo = b.array(&inputs.iter().map(|row| row[0]).collect::<Vec<_>>());
            let hi = b.array(&inputs.iter().map(|row| row[1]).collect::<Vec<_>>());
            let alpha = b.array(&inputs.iter().map(|row| row[2]).collect::<Vec<_>>());
            let factors = match helper {
                Some(alias) => {
                    let head = b.module_call(alias, "normsys_factor");
                    b.call("broadcast", &[head, lo, hi, alpha])
                }
                None => {
                    let head = b.module_call("hepphys", function);
                    b.call("broadcast", &[head, lo, one, hi, alpha])
                }
            };
            let name = b.bind_unique_doc(
                &format!("{channel}_normsys_{function}"),
                factors,
                "Distinct normalization factors.",
            );
            let factors = b.self_ref(&name);
            let vector = self.vectors.len();
            self.vectors.push((name, rows.len()));
            for (offset, row) in rows.into_iter().enumerate() {
                let index = b.lit_int(offset as i64 + 1);
                let factor = b.call("get", &[factors, index]);
                self.values.insert(row.key(), factor);
                self.slots.insert(factor, (vector, offset));
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

    /// Replace the batched factors of each scalar run by one gathered product.
    ///
    /// A scalar `get` per factor and an ordered multiply chain emit one tiny
    /// contraction per factor, which dominates the lowered graph and its
    /// derivative. Each run gathers its factor indices from the channel factor
    /// vector, and one `prod` reduces all runs of that vector. Shorter runs pad
    /// with the index of a trailing unit lane, so the work follows the run
    /// lengths. §07 defines `prod` as the product, with no order.
    /// A repeated factor stays in the chain, as the index set has no multiplicity.
    /// Factors that an earlier channel batched reduce over that vector.
    /// The product takes the place of the run's first gathered factor, and the
    /// other factors keep their order around it.
    pub(crate) fn gathered_products(
        &mut self,
        b: &mut Builder,
        channel: &str,
        runs: impl IntoIterator<Item = Vec<NodeId>>,
    ) -> BTreeMap<Vec<NodeId>, Vec<NodeId>> {
        let runs: BTreeSet<Vec<NodeId>> = runs.into_iter().collect();
        let slots = &self.slots;
        let masks_of = |run: &[NodeId]| {
            let mut masks: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
            for factor in run {
                if let Some(&(vector, offset)) = slots.get(factor) {
                    masks.entry(vector).or_default().insert(offset);
                }
            }
            masks.retain(|_, offsets| offsets.len() >= 2);
            masks
        };
        let mut rows: BTreeMap<usize, BTreeSet<BTreeSet<usize>>> = BTreeMap::new();
        for run in &runs {
            for (vector, offsets) in masks_of(run) {
                rows.entry(vector).or_default().insert(offsets);
            }
        }
        let mut products = BTreeMap::new();
        for (vector, masks) in rows {
            let (name, length) = self.vectors[vector].clone();
            let width = masks.iter().map(BTreeSet::len).max().unwrap_or(0);
            let source = if masks.iter().all(|offsets| offsets.len() == width) {
                name
            } else {
                self.padded
                    .entry(vector)
                    .or_insert_with(|| {
                        let factors = b.self_ref(&name);
                        let one = b.lit_real(1.0);
                        let unit = b.array(&[one]);
                        let padded = b.call("cat", &[factors, unit]);
                        b.bind_unique_doc(
                            &format!("{name}_padded"),
                            padded,
                            "Normalization factors with a trailing unit lane for padded gathers.",
                        )
                    })
                    .clone()
            };
            let arrays: Vec<_> = masks
                .iter()
                .map(|offsets| {
                    let indices: Vec<_> = offsets
                        .iter()
                        .map(|&offset| offset + 1)
                        .chain(std::iter::repeat(length + 1))
                        .take(width)
                        .map(|index| b.lit_int(index as i64))
                        .collect();
                    let indices = b.array(&indices);
                    let factors = b.self_ref(&source);
                    b.call("get", &[factors, indices])
                })
                .collect();
            let arrays = b.array(&arrays);
            let prod = b.call_head("prod");
            let values = b.call("broadcast", &[prod, arrays]);
            let bound = b.bind_unique_doc(
                &format!("{channel}_normsys_products"),
                values,
                "Products of gathered normalization factors, one per modifier run.",
            );
            let values = b.self_ref(&bound);
            for (index, offsets) in masks.into_iter().enumerate() {
                let index = b.lit_int(index as i64 + 1);
                products.insert((vector, offsets), b.call("get", &[values, index]));
            }
        }
        runs.into_iter()
            .map(|run| {
                let masks = masks_of(&run);
                let mut seen = BTreeSet::new();
                let mut placed = BTreeSet::new();
                let mut reduced = Vec::new();
                for &factor in &run {
                    match slots.get(&factor) {
                        Some(&(vector, _))
                            if masks.contains_key(&vector) && seen.insert(factor) =>
                        {
                            if placed.insert(vector) {
                                reduced.push(products[&(vector, masks[&vector].clone())]);
                            }
                        }
                        _ => reduced.push(factor),
                    }
                }
                (run, reduced)
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
