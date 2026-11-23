//! Opt-out policy for the embedded Egglog catalogues.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

use egglog::ast::Command;
use egglog::{EGraph, Error};

const PROGRAMS: &[(&str, &str)] = &[
    (
        "categorical_counts.egg",
        include_str!("rules/categorical_counts.egg"),
    ),
    ("concat.egg", include_str!("rules/concat.egg")),
    ("conjugacy.egg", include_str!("rules/conjugacy.egg")),
    ("fixed_poisson.egg", include_str!("rules/fixed_poisson.egg")),
    ("gaussian.egg", include_str!("rules/gaussian.egg")),
    ("hoisting.egg", include_str!("rules/hoisting.egg")),
    ("mathematical.egg", include_str!("rules/mathematical.egg")),
    (
        "matrix_projection.egg",
        include_str!("rules/matrix_projection.egg"),
    ),
    (
        "measure_algebra.egg",
        include_str!("rules/measure_algebra.egg"),
    ),
    ("ordered_scan.egg", include_str!("rules/ordered_scan.egg")),
    ("portable.egg", include_str!("rules/portable.egg")),
    ("scan_demand.egg", include_str!("rules/scan_demand.egg")),
    ("schema.egg", include_str!("rules/schema.egg")),
    ("sharing.egg", include_str!("rules/sharing.egg")),
    (
        "simplification.egg",
        include_str!("rules/simplification.egg"),
    ),
    ("stan.egg", include_str!("rules/stan.egg")),
    ("statistical.egg", include_str!("rules/statistical.egg")),
];

/// An authored catalogue rule and its ruleset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleInfo {
    pub name: &'static str,
    pub ruleset: &'static str,
}

struct Registry {
    rules: Vec<RuleInfo>,
    rulesets: Vec<&'static str>,
}

#[derive(Default)]
struct CatalogueNames {
    rules: Vec<(String, usize)>,
    rulesets: Vec<String>,
}

fn parse_catalogue_names() -> CatalogueNames {
    let mut graph = EGraph::default().with_num_threads(1);
    let mut names = CatalogueNames::default();
    for &(filename, source) in PROGRAMS {
        let commands = graph
            .parse_program(Some(filename.to_owned()), source)
            .unwrap_or_else(|error| {
                panic!("invalid embedded Egglog catalogue {filename}: {error}")
            });
        for command in commands {
            if let Command::AddRuleset(_, name) = command {
                assert!(
                    !names.rulesets.contains(&name),
                    "duplicate ruleset in {filename}"
                );
                names.rulesets.push(name);
            } else if let Some((name, ruleset)) = rule_identity(&command) {
                assert!(!name.is_empty(), "unnamed rule in {filename}");
                let ruleset = names
                    .rulesets
                    .iter()
                    .position(|declared| declared == ruleset)
                    .expect("declared Egglog ruleset");
                names.rules.push((name.to_owned(), ruleset));
            }
        }
    }
    names
}

fn registry() -> &'static Registry {
    static NAMES: OnceLock<CatalogueNames> = OnceLock::new();
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let names = NAMES.get_or_init(parse_catalogue_names);
        let mut rules: Vec<_> = names
            .rules
            .iter()
            .map(|(name, ruleset)| RuleInfo {
                name,
                ruleset: &names.rulesets[*ruleset],
            })
            .collect();
        rules.sort_unstable_by_key(|rule| rule.name);
        assert!(
            rules.windows(2).all(|pair| pair[0].name != pair[1].name),
            "duplicate Egglog rule identity"
        );
        let mut rulesets: Vec<_> = names.rulesets.iter().map(String::as_str).collect();
        rulesets.sort_unstable();
        Registry { rules, rulesets }
    })
}

/// The declared rules for every profile and standalone emitter planner.
pub fn available_rules() -> &'static [RuleInfo] {
    &registry().rules
}

/// The declared categories, in sorted order.
pub fn available_rulesets() -> &'static [&'static str] {
    &registry().rulesets
}

fn find_rule(name: &str) -> Option<&'static RuleInfo> {
    let rules = available_rules();
    let index = rules.binary_search_by_key(&name, |rule| rule.name).ok()?;
    Some(&rules[index])
}

/// A validated rule, ruleset or complete Egglog opt-out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleSelector(Selector);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Selector {
    All,
    Rule(&'static str),
    Ruleset(&'static str),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnknownRuleSelector(String);

impl fmt::Display for UnknownRuleSelector {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(&self.0)
    }
}

impl std::error::Error for UnknownRuleSelector {}

impl RuleSelector {
    pub fn rule(name: &str) -> Result<Self, UnknownRuleSelector> {
        find_rule(name)
            .map(|rule| Self(Selector::Rule(rule.name)))
            .ok_or_else(|| UnknownRuleSelector(format!("unknown Egglog rule {name:?}")))
    }

    pub fn ruleset(name: &str) -> Result<Self, UnknownRuleSelector> {
        let sets = available_rulesets();
        sets.binary_search(&name)
            .map(|index| Self(Selector::Ruleset(sets[index])))
            .map_err(|_| UnknownRuleSelector(format!("unknown Egglog ruleset {name:?}")))
    }
}

impl FromStr for RuleSelector {
    type Err = UnknownRuleSelector;

    fn from_str(selector: &str) -> Result<Self, Self::Err> {
        match selector.split_once(':') {
            None if selector == "all" => Ok(Self(Selector::All)),
            Some(("rule", name)) => Self::rule(name),
            Some(("ruleset", name)) => Self::ruleset(name),
            _ => Err(UnknownRuleSelector(format!(
                "invalid Egglog selector {selector:?} (expected all, rule:<id>, or ruleset:<id>)"
            ))),
        }
    }
}

impl fmt::Display for RuleSelector {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Selector::All => fmt.write_str("all"),
            Selector::Rule(name) => write!(fmt, "rule:{name}"),
            Selector::Ruleset(name) => write!(fmt, "ruleset:{name}"),
        }
    }
}

/// An emission-local opt-out policy. Its default keeps all eligible rules.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RulePolicy(Policy);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
enum Policy {
    #[default]
    Default,
    Filtered(Arc<Disabled>),
    AllDisabled,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct Disabled {
    rules: BTreeSet<&'static str>,
    rulesets: BTreeSet<&'static str>,
}

impl RulePolicy {
    /// Validate IDs against the full catalogue, independent of target.
    pub fn new(
        disabled_rules: &[&str],
        disabled_rulesets: &[&str],
    ) -> Result<Self, UnknownRuleSelector> {
        disabled_rules
            .iter()
            .map(|name| RuleSelector::rule(name))
            .chain(
                disabled_rulesets
                    .iter()
                    .map(|name| RuleSelector::ruleset(name)),
            )
            .collect::<Result<Vec<_>, _>>()
            .map(Self::from_selectors)
    }

    /// Collect validated selectors. The complete opt-out supersedes individual selections.
    pub fn from_selectors(selectors: impl IntoIterator<Item = RuleSelector>) -> Self {
        let mut disabled = Disabled::default();
        for selector in selectors {
            match selector.0 {
                Selector::All => return Self::all_disabled(),
                Selector::Rule(name) => {
                    disabled.rules.insert(name);
                }
                Selector::Ruleset(name) => {
                    disabled.rulesets.insert(name);
                }
            }
        }
        if disabled.rules.is_empty() && disabled.rulesets.is_empty() {
            Self::default()
        } else {
            Self(Policy::Filtered(Arc::new(disabled)))
        }
    }

    pub const fn all_disabled() -> Self {
        Self(Policy::AllDisabled)
    }

    pub const fn is_all_disabled(&self) -> bool {
        matches!(self.0, Policy::AllDisabled)
    }

    pub const fn is_default(&self) -> bool {
        matches!(self.0, Policy::Default)
    }

    /// Whether an authored rule is disabled directly or by its ruleset.
    pub fn disables_rule(&self, name: &str) -> bool {
        match &self.0 {
            Policy::Default => false,
            Policy::Filtered(disabled) => {
                disabled.rules.contains(name)
                    || (!disabled.rulesets.is_empty()
                        && find_rule(name)
                            .is_some_and(|rule| disabled.rulesets.contains(rule.ruleset)))
            }
            Policy::AllDisabled => true,
        }
    }

    fn disables_rule_in_set(&self, name: &str, ruleset: &str) -> bool {
        match &self.0 {
            Policy::Default => false,
            Policy::Filtered(disabled) => {
                disabled.rules.contains(name) || disabled.rulesets.contains(ruleset)
            }
            Policy::AllDisabled => true,
        }
    }

    pub fn disables_ruleset(&self, name: &str) -> bool {
        match &self.0 {
            Policy::Default => false,
            Policy::Filtered(disabled) => disabled.rulesets.contains(name),
            Policy::AllDisabled => true,
        }
    }

    pub fn enables_ruleset(&self, name: &str) -> bool {
        !self.disables_ruleset(name)
    }
}

fn rule_identity(command: &Command) -> Option<(&str, &str)> {
    match command {
        Command::Rule { rule } => Some((&rule.name, &rule.ruleset)),
        Command::Rewrite(ruleset, rewrite, _) | Command::BiRewrite(ruleset, rewrite) => {
            Some((&rewrite.name, ruleset))
        }
        _ => None,
    }
}

pub(super) fn load_program(
    graph: &mut EGraph,
    filename: String,
    source: &str,
    policy: &RulePolicy,
) -> Result<(), Error> {
    if policy.is_default() {
        graph.parse_and_run_program(Some(filename), source)?;
    } else {
        let mut commands = graph.parse_program(Some(filename), source)?;
        // Relations and ruleset declarations remain available to the host
        // and to schedules that combine the active sets.
        commands.retain(|command| {
            rule_identity(command)
                .is_none_or(|(name, ruleset)| !policy.disables_rule_in_set(name, ruleset))
        });
        graph.run_program(commands)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::integer_plan_rows;

    #[test]
    fn policy_recognizes_authored_rules_and_sets() {
        let selectors = ["rule:portable.identity", "ruleset:measure-algebra"];
        let rules = RulePolicy::from_selectors(selectors.map(|selector| selector.parse().unwrap()));
        assert!(rules.disables_rule("portable.identity"));
        assert!(!rules.disables_rule("portable.double-negation"));
        assert!(rules.disables_ruleset("measure-algebra"));
        assert!(rules.disables_rule("measure-algebra.intersect-density"));
        assert!(rules.enables_ruleset("portable"));
        assert!("rule:identity".parse::<RuleSelector>().is_err());
        assert!("ruleset:query-portable".parse::<RuleSelector>().is_err());
        assert!(RulePolicy::from_selectors(["all".parse().unwrap()]).is_all_disabled());
        assert_eq!(RulePolicy::new(&[], &[]).unwrap(), RulePolicy::default());
    }

    #[test]
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn every_rule_file_contributes_to_policy_discovery() {
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rules");
        let actual: BTreeSet<_> = std::fs::read_dir(folder)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|file| file.extension().is_some_and(|extension| extension == "egg"))
            .map(|file| file.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let registered: BTreeSet<_> = PROGRAMS
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .collect();
        assert_eq!(actual, registered);
    }

    #[test]
    fn standalone_plans_respect_individual_rules_and_entire_sets() {
        fn plan(policy: &RulePolicy) -> Option<Vec<Vec<i64>>> {
            integer_plan_rows(
                "ordered_scan.egg",
                include_str!("rules/ordered_scan.egg"),
                "ordered-scan-index",
                &[
                    ("ordered-scan-range", &[499, 2, 500]),
                    ("ordered-scan-width", &[499, 4]),
                ],
                "ordered-index-choice",
                policy,
            )
        }
        let default = RulePolicy::default();
        let expected = Some(vec![vec![2, 500]]);
        assert_eq!(plan(&default), expected);
        assert_eq!(
            plan(&RulePolicy::new(&["ordered-scan-index.choose"], &[]).unwrap())
                .unwrap_or_default(),
            Vec::<Vec<i64>>::new()
        );
        assert_eq!(
            plan(&RulePolicy::new(&[], &["ordered-scan-index"]).unwrap()),
            None
        );
        assert_eq!(
            plan(&RulePolicy::new(&["ordered-scan.choose-width"], &[]).unwrap()),
            expected
        );
        assert_eq!(plan(&RulePolicy::all_disabled()), None);
        assert_eq!(plan(&default), expected);
    }
}
