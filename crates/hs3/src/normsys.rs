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
    /// Interpolation behind each batched factor.
    batched: BTreeMap<NodeId, Interpolation>,
    /// One run-ordered lane tensor per interpolation function, for the whole model.
    lanes: BTreeMap<&'static str, Lanes>,
}

/// Rows are contiguous blocks of one width, padded with unit lanes, so each
/// row is a slice and the reverse pass needs no scatter. Single factors follow
/// the rows.
#[derive(Default)]
struct Lanes {
    /// Bindings of the lanes, the row products and the single factors.
    names: Option<[String; 3]>,
    rows: Vec<Vec<Interpolation>>,
    singles: Vec<Interpolation>,
    index: BTreeMap<Vec<Key>, usize>,
}

impl Factors {
    /// Preserve occurrence and product order; batch only new distinct factors.
    pub(crate) fn multipliers(
        &mut self,
        b: &mut Builder,
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
            // A lone factor of its kind stays a scalar call. Batched factors are
            // scalar calls too, which `gathered_products` replaces by lanes of
            // the model-wide tensor.
            let batch = rows.len() > 1;
            for row in rows {
                let [lo, hi, alpha] = row.inputs(b);
                let factor = match helper {
                    Some(alias) => b.module_user_call(alias, "normsys_factor", &[lo, hi, alpha]),
                    None => b.module_user_call("hepphys", function, &[lo, one, hi, alpha]),
                };
                self.values.insert(row.key(), factor);
                if batch {
                    self.batched.insert(factor, row);
                }
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

    /// Replace the batched factors of each scalar run by one product row of a
    /// model-wide lane tensor.
    ///
    /// Per-channel vectors with gathered rows emit one gather per row, and the
    /// reverse pass one scatter per row, each its own GPU kernel. Here every run
    /// owns a contiguous row of interpolated lanes, so rows are slices and the
    /// whole model shares one interpolation and one `prod`. A unit lane is
    /// `normsys(1, 1, alpha) = 1` for every alpha, the identity of `prod` (§07), and a repeated
    /// factor repeats its lane. The product takes the place of the run's first
    /// batched factor, and the other factors keep their order around it.
    pub(crate) fn gathered_products(
        &mut self,
        b: &mut Builder,
        runs: impl IntoIterator<Item = Vec<NodeId>>,
    ) -> BTreeMap<Vec<NodeId>, Vec<NodeId>> {
        let runs: BTreeSet<Vec<NodeId>> = runs.into_iter().collect();
        runs.into_iter()
            .map(|run| {
                let row: Vec<_> = run
                    .iter()
                    .filter_map(|f| self.batched.get(f).copied())
                    .collect();
                let mut reduced = Vec::new();
                let mut placed = false;
                for &factor in &run {
                    if !self.batched.contains_key(&factor) {
                        reduced.push(factor);
                    } else if !placed {
                        placed = true;
                        reduced.push(self.row(b, row.clone()));
                    }
                }
                (run, reduced)
            })
            .collect()
    }

    fn row(&mut self, b: &mut Builder, row: Vec<Interpolation>) -> NodeId {
        let lanes = self.lanes.entry(row[0].function).or_default();
        let [_, products_name, singles_name] = lanes
            .names
            .get_or_insert_with(|| {
                ["lanes", "products", "singles"]
                    .map(|part| b.alloc_name(&format!("normsys_{}_{part}", row[0].function)))
            })
            .clone();
        let key: Vec<_> = row.iter().map(|r| r.key()).collect();
        let (source, index) = if let [single] = row[..] {
            let next = lanes.singles.len();
            let index = *lanes.index.entry(key).or_insert(next);
            if index == next {
                lanes.singles.push(single);
            }
            (singles_name, index)
        } else {
            let next = lanes.rows.len();
            let index = *lanes.index.entry(key).or_insert(next);
            if index == next {
                lanes.rows.push(row);
            }
            (products_name, index)
        };
        let values = b.self_ref(&source);
        let index = b.lit_int(index as i64 + 1);
        b.call("get", &[values, index])
    }

    /// Bind the lane tensors once every channel has requested its rows.
    pub(crate) fn finish(&self, b: &mut Builder, pyhf_helpers: Option<&str>) {
        for (&function, lanes) in &self.lanes {
            let Some([lanes_name, products_name, singles_name]) = &lanes.names else {
                continue;
            };
            let mut order: Vec<usize> = (0..lanes.rows.len()).collect();
            order.sort_by_key(|&row| std::cmp::Reverse(lanes.rows[row].len()));
            let lengths: Vec<_> = order.iter().map(|&row| lanes.rows[row].len()).collect();
            let mut inputs = Vec::new();
            // (first lane, width, rows) per class, rows in decreasing length.
            let mut classes = Vec::new();
            let mut start = 0;
            for end in width_classes(&lengths) {
                let members = &order[start..end];
                let width = tree_width(lanes.rows[members[0]].len());
                classes.push((inputs.len(), width, members));
                for &member in members {
                    let row = &lanes.rows[member];
                    inputs.extend(row.iter().map(|r| r.inputs(b)));
                    // The row's own parameter keeps alpha one input gather; a
                    // literal lane would split it into one concatenate per lane.
                    let unit = Interpolation {
                        lo: 1.0,
                        hi: 1.0,
                        ..row[0]
                    };
                    (row.len()..width).for_each(|_| inputs.push(unit.inputs(b)));
                }
                start = end;
            }
            let inputs_before_singles = inputs.len();
            inputs.extend(lanes.singles.iter().map(|r| r.inputs(b)));
            let [lo, hi, alpha] =
                [0, 1, 2].map(|i| b.array(&inputs.iter().map(|row| row[i]).collect::<Vec<_>>()));
            let factors = match pyhf_helpers.filter(|_| function == INTERP_NORMSYS_DEFAULT) {
                Some(alias) => {
                    let head = b.module_call(alias, "normsys_factor");
                    b.call("broadcast", &[head, lo, hi, alpha])
                }
                None => {
                    let head = b.module_call("hepphys", function);
                    let one = b.lit_real(1.0);
                    b.call("broadcast", &[head, lo, one, hi, alpha])
                }
            };
            b.bind_doc(
                lanes_name,
                factors,
                &["Normalization factors, one padded row per modifier run, then single factors."],
            );
            let mut places = vec![None; lanes.rows.len()];
            for (first, width, members) in classes {
                let rows: Vec<_> = (0..members.len())
                    .map(|k| {
                        let lanes = first + k * width;
                        let indices: Vec<_> = (lanes + 1..=lanes + width)
                            .map(|i| b.lit_int(i as i64))
                            .collect();
                        let indices = b.array(&indices);
                        let values = b.self_ref(lanes_name);
                        b.call("get", &[values, indices])
                    })
                    .collect();
                let rows = b.array(&rows);
                let prod = b.call_head("prod");
                let products = b.call("broadcast", &[prod, rows]);
                let class = b.bind_unique_doc(
                    &format!("{products_name}_{width}"),
                    products,
                    "Products of normalization factor rows of one width class.",
                );
                for (k, &member) in members.iter().enumerate() {
                    places[member] = Some((class.clone(), k));
                }
            }
            if !places.is_empty() {
                let products: Vec<_> = places
                    .into_iter()
                    .map(|place| {
                        let (class, k) = place.expect("every row has a class");
                        let values = b.self_ref(&class);
                        let index = b.lit_int(k as i64 + 1);
                        b.call("get", &[values, index])
                    })
                    .collect();
                let products = b.array(&products);
                b.bind_doc(
                    products_name,
                    products,
                    &["Products of normalization factor rows, one per modifier run."],
                );
            }
            if !lanes.singles.is_empty() {
                let start = inputs_before_singles;
                let indices: Vec<_> = (start + 1..=start + lanes.singles.len())
                    .map(|i| b.lit_int(i as i64))
                    .collect();
                let indices = b.array(&indices);
                let values = b.self_ref(lanes_name);
                let singles = b.call("get", &[values, indices]);
                b.bind_doc(
                    singles_name,
                    singles,
                    &["Normalization factors that no run multiplies with another."],
                );
            }
        }
    }
}

/// Padding share above which a lane tensor splits into another width class.
/// Padding costs lanes on CPU and each class costs kernel launches on GPU.
const MAX_PADDING: f64 = 0.15;

/// Smallest width of the form 2^k or 3 * 2^k that holds `length` lanes.
/// The pairwise product tree then halves without an odd tail, except once at
/// width 3, and each odd tail costs a slice and a concatenate that split GPU
/// fusions.
fn tree_width(length: usize) -> usize {
    let power = length.next_power_of_two();
    if power / 4 * 3 >= length {
        power / 4 * 3
    } else {
        power
    }
}

/// Fewest contiguous classes over `lengths`, sorted in decreasing order, whose
/// rows padded to the tree width of each class's first length keep the lanes
/// beyond each row's own tree width under [`MAX_PADDING`]. One row per class
/// meets the bound, so the search ends.
/// Returns the end of each class.
fn width_classes(lengths: &[usize]) -> Vec<usize> {
    // Tree rounding alone can exceed the bound, as one row per class shows.
    let floor: usize = lengths.iter().map(|&length| tree_width(length)).sum();
    // Fewest lanes covering the first i rows with the current class count.
    let mut best: Vec<Option<(usize, Vec<usize>)>> = vec![None; lengths.len() + 1];
    best[0] = Some((0, Vec::new()));
    loop {
        if let Some((lanes, ends)) = &best[lengths.len()]
            && (lanes - floor) as f64 <= MAX_PADDING * *lanes as f64
        {
            return ends.clone();
        }
        best = (0..=lengths.len())
            .map(|i| {
                (0..i)
                    .filter_map(|j| {
                        let (lanes, ends) = best[j].as_ref()?;
                        Some((lanes + tree_width(lengths[j]) * (i - j), ends))
                    })
                    .min_by_key(|(lanes, _)| *lanes)
                    .map(|(lanes, ends)| (lanes, [ends.as_slice(), &[i]].concat()))
            })
            .collect();
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
