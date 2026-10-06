//! Tensorize additive histogram modifiers before the compiler sees scalar calls.
//! Rows are actual modifier occurrences, not a padded sample-by-modifier grid.

use std::collections::BTreeMap;

use flatppl_core::NodeId;

use crate::builder::Builder;
use crate::histfactory::{INTERP_HISTOSYS_DEFAULT, Interpolation};

/// Interpolate a channel's templates together, then reduce each sample's rows.
/// Every shift uses the original nominal. Multiplicative modifiers and auxiliary
/// constraints remain the caller's responsibility.
pub(crate) fn shifted_nominals(
    b: &mut Builder,
    channel: &str,
    nominals: &[String],
    modifiers: &[Vec<Interpolation>],
    pyhf_helpers: Option<&str>,
) -> Vec<NodeId> {
    let mut results: Vec<_> = nominals.iter().map(|name| b.self_ref(name)).collect();
    let count: usize = modifiers.iter().map(Vec::len).sum();
    if count < 2 {
        for (sample, rows) in modifiers.iter().enumerate() {
            if let Some(row) = rows.first() {
                results[sample] =
                    match pyhf_helpers.filter(|_| row.function == INTERP_HISTOSYS_DEFAULT) {
                        Some(alias) => {
                            let head = b.module_call(alias, "histosys_shift");
                            let shift = b.call(
                                "broadcast",
                                &[head, row.lo, results[sample], row.hi, row.alpha],
                            );
                            let add = b.call_head("add");
                            b.call("broadcast", &[add, results[sample], shift])
                        }
                        None => row.apply(b, results[sample]),
                    };
            }
        }
        return results;
    }

    // pyhf uses one interpolation kind. Native HS3 can mix kinds, whose
    // additive contributions still sum against the same original nominal.
    let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for (sample, rows) in modifiers.iter().enumerate() {
        for row in rows {
            groups.entry(row.function).or_default().push((sample, row));
        }
    }
    for (function, rows) in groups {
        let lo: Vec<_> = rows.iter().map(|(_, row)| row.lo).collect();
        let hi: Vec<_> = rows.iter().map(|(_, row)| row.hi).collect();
        let nominal: Vec<_> = rows
            .iter()
            .map(|(sample, _)| b.self_ref(&nominals[*sample]))
            .collect();
        let alpha: Vec<_> = rows.iter().map(|(_, row)| row.alpha).collect();
        let lo = b.array(&lo);
        let lo = b.call("rowstack", &[lo]);
        let hi = b.array(&hi);
        let hi = b.call("rowstack", &[hi]);
        let nominal = b.array(&nominal);
        let nominal = b.call("rowstack", &[nominal]);
        let alpha = b.array(&alpha);
        let zero = b.lit_int(0);
        let one = b.lit_int(1);
        let alpha = b.call("addaxes", &[alpha, zero, one]);
        let shifts = match pyhf_helpers.filter(|_| function == INTERP_HISTOSYS_DEFAULT) {
            Some(alias) => {
                let head = b.module_call(alias, "histosys_shift");
                b.call("broadcast", &[head, lo, nominal, hi, alpha])
            }
            None => {
                let head = b.module_call("hepphys", function);
                let interpolated = b.call("broadcast", &[head, lo, nominal, hi, alpha]);
                let sub = b.call_head("sub");
                b.call("broadcast", &[sub, interpolated, nominal])
            }
        };
        let name = b.bind_unique_doc(
            &format!("{channel}_histosys_shifts"),
            shifts,
            "Additive histosys shifts: modifier rows in sample order, bin columns.",
        );
        let shifts = b.self_ref(&name);

        let mut offset = 0;
        for sample_rows in rows.chunk_by(|a, b| a.0 == b.0) {
            let sample = sample_rows[0].0;
            let indices: Vec<_> = (offset..offset + sample_rows.len())
                .map(|index| b.lit_int(index as i64 + 1))
                .collect();
            let indices = b.array(&indices);
            let all = b.call_head("all");
            let selected = b.call("get", &[shifts, indices, all]);
            let delta = b.column_sums(selected);
            let add = b.call_head("add");
            results[sample] = b.call("broadcast", &[add, results[sample], delta]);
            offset += sample_rows.len();
        }
    }
    results
}
