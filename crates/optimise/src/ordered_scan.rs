//! Size-bounded grouping for a verified scalar GPU scan.

use crate::{RulePolicy, engine::integer_plan_rows};

const RULES: &str = include_str!("rules/ordered_scan.egg");

/// Choose an ordered block width for an already eligible GPU scalar scan.
/// Callback size includes its distinct syntax nodes, before repeated lowering.
pub fn ordered_scan_width(length: u64, body_nodes: usize) -> u64 {
    ordered_scan_width_with_rules(length, body_nodes, &RulePolicy::default())
}

/// Choose a block width using the caller's egglog rules.
pub fn ordered_scan_width_with_rules(length: u64, body_nodes: usize, rules: &RulePolicy) -> u64 {
    const ORIGINAL: u64 = 4;
    if length < 128 || !(1..=48).contains(&body_nodes) {
        return ORIGINAL;
    }
    let Ok(length) = i64::try_from(length) else {
        return ORIGINAL;
    };
    let Some(plans) = integer_plan_rows(
        "ordered_scan.egg",
        RULES,
        "ordered-scan",
        &[("ordered-scalar-scan", &[length, body_nodes as i64])],
        "ordered-block-choice",
        rules,
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

/// Use a counter-derived item for a fixed integer sequence in an ordered GPU scan.
/// The caller supplies endpoints of a verified unit-stride sequence.
pub fn plan_ordered_scan_index(
    first: i64,
    last: i64,
    length: u64,
    width: u64,
) -> Option<(i64, i64)> {
    plan_ordered_scan_index_with_rules(first, last, length, width, &RulePolicy::default())
}

/// Plan a counter-derived scan item using the caller's egglog rules.
pub fn plan_ordered_scan_index_with_rules(
    first: i64,
    last: i64,
    length: u64,
    width: u64,
    rules: &RulePolicy,
) -> Option<(i64, i64)> {
    last.checked_sub(first)?;
    let length = i64::try_from(length).ok()?;
    let width = i64::try_from(width).ok()?;
    integer_plan_rows(
        "ordered_scan.egg",
        RULES,
        "ordered-scan-index",
        &[
            ("ordered-scan-range", &[length, first, last]),
            ("ordered-scan-width", &[length, width]),
        ],
        "ordered-index-choice",
        rules,
    )?
    .iter()
    .find_map(|plan| match plan.as_slice() {
        [first, last] => Some((*first, *last)),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::plan_ordered_scan_index;

    #[test]
    fn range_plan_requires_an_ordered_block_and_matching_interval() {
        assert_eq!(plan_ordered_scan_index(2, 500, 499, 4), Some((2, 500)));
        assert_eq!(plan_ordered_scan_index(-2, 32, 35, 4), Some((-2, 32)));
        assert_eq!(plan_ordered_scan_index(2, 500, 499, 1), None);
        assert_eq!(plan_ordered_scan_index(2, 500, 500, 4), None);
        assert_eq!(plan_ordered_scan_index(1, 2, 2, 4), None);
        assert_eq!(plan_ordered_scan_index(i64::MIN, i64::MAX, 35, 4), None);
    }
}
