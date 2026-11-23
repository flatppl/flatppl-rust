//! Reuse the simplex verdict for repeats of a fixed probability-row index.

use std::collections::HashSet;

use crate::{RulePolicy, engine::integer_plan_rows};

const RULES: &str = include_str!("rules/categorical_counts.egg");

/// The rows needed for the same parameter validity check as the original gather.
pub enum SimplexRowCheck {
    All,
    Select(Vec<i64>),
}

/// Accept only an in-range fixed selector supplied by the caller after binding
/// runtime inputs. `index` reads an already-normalized zero-based row.
pub fn plan_simplex_rows(
    row_extent: u64,
    observations: u64,
    index: impl FnMut(usize) -> Option<i64>,
) -> Option<SimplexRowCheck> {
    plan_simplex_rows_with_rules(row_extent, observations, index, &RulePolicy::default())
}

/// Select the required simplex row checks using the caller's egglog rules.
pub fn plan_simplex_rows_with_rules(
    row_extent: u64,
    observations: u64,
    index: impl FnMut(usize) -> Option<i64>,
    rules: &RulePolicy,
) -> Option<SimplexRowCheck> {
    plan(row_extent, observations, index, RULES, rules)
}

fn plan(
    row_extent: u64,
    observations: u64,
    mut index: impl FnMut(usize) -> Option<i64>,
    program: &str,
    rules: &RulePolicy,
) -> Option<SimplexRowCheck> {
    if !(2..=65_536).contains(&observations)
        || row_extent == 0
        || !rules.enables_ruleset("simplex-row-checks")
    {
        return None;
    }
    let row_extent = i64::try_from(row_extent).ok()?;
    let mut seen = HashSet::new();
    let mut distinct = Vec::new();
    for at in 0..observations as usize {
        let row = index(at)?;
        if !(0..row_extent).contains(&row) {
            return None;
        }
        if seen.insert(row) {
            distinct.push(row);
        }
    }
    if distinct.len() as u64 == observations {
        return None;
    }
    let occupied = distinct.len() as i64;
    let identity = occupied == row_extent && distinct.iter().copied().eq(0..row_extent);
    let partition = [
        row_extent,
        observations as i64,
        occupied,
        i64::from(identity),
    ];
    let plans = integer_plan_rows(
        "categorical_counts.egg",
        program,
        "simplex-row-checks",
        &[("fixed-row-partition", &partition)],
        "check-occupied-rows",
        rules,
    )?;
    match plans.as_slice() {
        [plan] if identity && plan.as_slice() == [occupied, 1] => Some(SimplexRowCheck::All),
        [plan] if plan.as_slice() == [occupied, 0] => Some(SimplexRowCheck::Select(distinct)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_controls_both_profitability_thresholds() {
        let small = |at| Some([2, 0][at % 2]);
        assert!(plan_simplex_rows(3, 16, small).is_none());
        let lower_floor = RULES.replace("(>= selected 64)", "(>= selected 8)");
        assert!(matches!(
            plan(3, 16, small, &lower_floor, &RulePolicy::default()),
            Some(SimplexRowCheck::Select(rows)) if rows == [2, 0]
        ));

        let modest_reuse = |at| Some((at % 24) as i64);
        assert!(plan_simplex_rows(24, 64, modest_reuse).is_none());
        let lower_ratio = RULES.replace("(* 3 occupied)", "(* 2 occupied)");
        assert!(matches!(
            plan(24, 64, modest_reuse, &lower_ratio, &RulePolicy::default()),
            Some(SimplexRowCheck::All)
        ));
    }
}
