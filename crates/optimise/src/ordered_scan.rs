//! Size-bounded grouping for a verified scalar GPU scan.

use crate::engine::integer_plan_rows;

/// Choose an ordered block width for an already eligible GPU scalar scan.
/// Callback size includes its distinct syntax nodes, before repeated lowering.
pub fn ordered_scan_width(length: u64, body_nodes: usize) -> u64 {
    const ORIGINAL: u64 = 4;
    if length < 128 || !(1..=48).contains(&body_nodes) {
        return ORIGINAL;
    }
    let Ok(length) = i64::try_from(length) else {
        return ORIGINAL;
    };
    let Some(plans) = integer_plan_rows(
        "ordered_scan.egg",
        include_str!("rules/ordered_scan.egg"),
        "ordered-scan",
        &[("ordered-scalar-scan", &[length, body_nodes as i64])],
        "ordered-block-choice",
    ) else {
        return ORIGINAL;
    };
    plans
        .iter()
        .filter_map(|plan| match plan.as_slice() {
            [width] => u64::try_from(*width).ok(),
            _ => None,
        })
        .max()
        .unwrap_or(ORIGINAL)
}
