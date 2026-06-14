use flatppl_core::Module;
use flatppl_lint::{Config, RuleId, Severity};

#[test]
fn rule_id_round_trips_kebab_name() {
    assert_eq!(RuleId::UnusedBinding.to_string(), "unused-binding");
    assert_eq!(
        "unused-binding".parse::<RuleId>().unwrap(),
        RuleId::UnusedBinding
    );
    assert!("no-such-rule".parse::<RuleId>().is_err());
}

#[test]
fn config_default_levels_match_spec() {
    let cfg = Config::default();
    assert_eq!(cfg.level(RuleId::UnresolvedName), Severity::Deny);
    assert_eq!(cfg.level(RuleId::InferenceCycle), Severity::Deny);
    assert_eq!(cfg.level(RuleId::InferenceGap), Severity::Warn);
    assert_eq!(cfg.level(RuleId::NotCanonical), Severity::Warn);
    assert_eq!(cfg.level(RuleId::UnusedBinding), Severity::Warn);
    assert_eq!(cfg.level(RuleId::ShadowsBuiltin), Severity::Warn);
    assert_eq!(cfg.level(RuleId::MissingDoc), Severity::Allow);
}

#[test]
fn config_override_wins_over_default() {
    let mut cfg = Config::default();
    cfg.set(RuleId::MissingDoc, Severity::Deny);
    assert_eq!(cfg.level(RuleId::MissingDoc), Severity::Deny);
}

/// Parse surface FlatPPL into a module (panics on parse error — test inputs are
/// known-good).
fn parse(src: &str) -> Module {
    flatppl_syntax::parse(src).expect("parse")
}

/// The rules that fired, by id, for a default config.
fn fired(src: &str) -> Vec<RuleId> {
    let mut module = parse(src);
    let cfg = Config::default();
    let mut ids: Vec<RuleId> = flatppl_lint::lint(&mut module, &cfg)
        .into_iter()
        .map(|d| d.rule)
        .collect();
    ids.sort_by_key(|r| r.name());
    ids
}

#[test]
fn flags_a_private_binding_no_one_references() {
    // `_helper` is private (leading `_`), referenced by nothing.
    let src = "_helper = 1.0\nx ~ Normal(mu = 0.0, sigma = 1.0)\n";
    assert!(fired(src).contains(&RuleId::UnusedBinding));
}

#[test]
fn does_not_flag_a_referenced_private_binding() {
    let src = "_mu = 0.0\nx ~ Normal(mu = _mu, sigma = 1.0)\n";
    assert!(!fired(src).contains(&RuleId::UnusedBinding));
}

#[test]
fn does_not_flag_public_bindings() {
    // `mu` is public (no leading `_`); a public binding is an interface point,
    // never "unused".
    let src = "mu = 0.0\n";
    assert!(!fired(src).contains(&RuleId::UnusedBinding));
}

#[test]
fn flags_a_binding_that_shadows_a_builtin() {
    // `sum` is a built-in reduction; binding it shadows the built-in.
    let src = "sum = 1.0\n";
    assert!(fired(src).contains(&RuleId::ShadowsBuiltin));
}

/// The rules that fire when `missing-doc` is promoted to a warning (its default
/// is `allow`, i.e. suppressed).
fn fired_with_doc_enabled(src: &str) -> Vec<RuleId> {
    let mut module = parse(src);
    let mut cfg = Config::default();
    cfg.set(RuleId::MissingDoc, Severity::Warn);
    let mut ids: Vec<RuleId> = flatppl_lint::lint(&mut module, &cfg)
        .into_iter()
        .map(|d| d.rule)
        .collect();
    ids.sort_by_key(|r| r.name());
    ids
}

#[test]
fn does_not_flag_an_ordinary_name() {
    let src = "my_param = 1.0\n";
    assert!(!fired(src).contains(&RuleId::ShadowsBuiltin));
}

#[test]
fn missing_doc_is_silent_by_default() {
    let src = "mu = 0.0\n";
    assert!(!fired(src).contains(&RuleId::MissingDoc));
}

#[test]
fn missing_doc_fires_for_undocumented_public_binding_when_enabled() {
    let src = "mu = 0.0\n";
    assert!(fired_with_doc_enabled(src).contains(&RuleId::MissingDoc));
}

#[test]
fn missing_doc_does_not_fire_for_private_binding() {
    let src = "_mu = 0.0\n_x ~ Normal(mu = _mu, sigma = 1.0)\n";
    assert!(!fired_with_doc_enabled(src).contains(&RuleId::MissingDoc));
}
