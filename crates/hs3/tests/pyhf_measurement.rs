//! The `pyhf_measurement_<name>` settings records against pyhf 0.7.6.
//!
//! The expected values are pyhf's own: `Workspace.model(measurement_name=...)`
//! then `config.par_order` and each paramset's `suggested_init`,
//! `suggested_bounds`, `suggested_fixed`, `width()` and `auxdata`.
use flatppl_hs3::{Auxiliary, Observations, PyhfOptions};
use flatppl_syntax::{Syntax, print_with};

/// Two channels and two measurements. The fixture covers a name shared by
/// normsys and histosys, staterror sorting, lumi, a shapesys bin with no
/// uncertainty, a zero-error staterror bin, and every override field.
const WORKSPACE: &str = include_str!("fixtures/pyhf_measurements.json");

fn convert(options: &PyhfOptions) -> String {
    print_with(
        &flatppl_hs3::read_pyhf_with(WORKSPACE, options).unwrap(),
        Syntax::Minimal,
    )
}

fn select(name: &str) -> PyhfOptions {
    PyhfOptions {
        measurement: Some(name.into()),
        ..PyhfOptions::default()
    }
}

fn line<'a>(text: &'a str, binding: &str) -> &'a str {
    let prefix = format!("{binding} = ");
    text.lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{binding}` in:\n{text}"))
}

const CHANNELS: &str = "channels = record(\
    b = record(name = \"b\", bins = 2, observed = \"b_observed\", expected = \"b_expected\"), \
    a = record(name = \"a\", bins = 1, observed = \"a_observed\", expected = \"a_expected\"))";

/// Parameters that both measurements leave at pyhf's defaults.
const SHARED: [&str; 4] = [
    "a_norm = record(pyhf = \"a_norm\", init = 1.0, lo = 0.0, hi = 10.0, fixed = false)",
    "k = record(pyhf = \"k\", init = 1.0, lo = 0.0, hi = 10.0, fixed = false)",
    "stat_a = record(pyhf = \"stat_a\", init = [1.0], lo = [0.0000000001], hi = [10.0], \
     fixed = [false], width = [0.25], auxdata = [1.0])",
    "stat_b = record(pyhf = \"stat_b\", init = [1.0, 1.0], lo = [0.0000000001, 0.0000000001], \
     hi = [10.0, 10.0], fixed = [false, true], width = [0.1, 1.0], auxdata = [1.0, 1.0])",
];

fn expected(head: &str, z: &str, lumi: &str, mu: &str, sf: &str, sh: &str) -> String {
    let [a_norm, k, stat_a, stat_b] = SHARED;
    // pyhf's `par_order`: histosys, lumi, normfactor, normsys, shapefactor,
    // shapesys, staterror, each by first (channel, sample) in sorted order.
    let parameters = [z, lumi, a_norm, k, mu, sf, sh, stat_a, stat_b].join(", ");
    format!("{head}, parameters = record({parameters}), {CHANNELS})")
}

#[test]
fn records_follow_pyhf_order_defaults_and_overrides() {
    let text = convert(&PyhfOptions::default());
    assert_eq!(
        line(&text, "pyhf_measurement_first"),
        expected(
            "pyhf_measurement_first = record(name = \"first\", selected = true, poi = \"mu\"",
            "z = record(pyhf = \"z\", init = 0.0, lo = -5.0, hi = 5.0, fixed = false, width = 1.0, auxdata = 0.0)",
            "lumi = record(pyhf = \"lumi\", init = 1.0, lo = 0.9, hi = 1.1, fixed = false, width = 0.02, auxdata = 1.0)",
            "mu = record(pyhf = \"mu\", init = 1.0, lo = 0.0, hi = 10.0, fixed = false)",
            "sf = record(pyhf = \"sf\", init = [1.0], lo = [0.0], hi = [10.0], fixed = [false])",
            "sh = record(pyhf = \"sh\", init = [1.0, 1.0], lo = [0.0000000001, 0.0000000001], \
             hi = [10.0, 10.0], fixed = [false, true], width = [0.2, 1.0], auxdata = [25.0, 1.0])",
        )
    );
    // The measurement name needs sanitizing. The overrides set inits, bounds
    // and a scalar `fixed` on a vector parameter, and `factors` moves the
    // shapesys width but not its auxdata.
    assert_eq!(
        line(&text, "pyhf_measurement_second_set"),
        expected(
            "pyhf_measurement_second_set = record(name = \"second set\", selected = false, poi = \"k\"",
            "z = record(pyhf = \"z\", init = 0.0, lo = -5.0, hi = 5.0, fixed = false, width = 1.0, auxdata = 0.5)",
            "lumi = record(pyhf = \"lumi\", init = 1.0, lo = 0.5, hi = 1.5, fixed = false, width = 0.05, auxdata = 1.0)",
            "mu = record(pyhf = \"mu\", init = 2.0, lo = -1.0, hi = 5.0, fixed = true)",
            "sf = record(pyhf = \"sf\", init = [1.0], lo = [0.0], hi = [10.0], fixed = [true])",
            "sh = record(pyhf = \"sh\", init = [1.0, 1.0], lo = [0.0000000001, 0.0000000001], \
             hi = [10.0, 10.0], fixed = [false, true], width = [0.1, 1.0], auxdata = [25.0, 1.0])",
        )
    );
}

#[test]
fn measurement_option_selects_the_likelihood_and_the_record() {
    let first = convert(&PyhfOptions::default());
    let second = convert(&select("second set"));
    assert!(first.contains("lumi_constraint = functionof(Normal(mu = lumi, sigma = 0.02))"));
    assert!(second.contains("lumi_constraint = functionof(Normal(mu = lumi, sigma = 0.05))"));
    assert!(second.contains("z_constraint_likelihood = likelihoodof(z_constraint, 0.5)"));
    // `factors` alone moves the rate, and the centre stays the default
    // (nominal/sigma)^2, as pyhf's `config.auxdata` and the settings record say.
    assert!(second.contains("sh_tau = [100.0, 1.0]"));
    assert!(second.contains("sh_constraint_likelihood = likelihoodof(sh_constraint, [25.0, 1.0])"));
    assert!(second.contains("pyhf_measurement_first = record(name = \"first\", selected = false,"));
    assert!(
        second.contains(
            "pyhf_measurement_second_set = record(name = \"second set\", selected = true,"
        )
    );
    assert_eq!(convert(&select("first")), first);
}

#[test]
fn measurement_option_refuses_an_unknown_name_and_a_model() {
    let error = flatppl_hs3::read_pyhf_with(WORKSPACE, &select("third")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("the workspace has no measurement named `third`"),
        "{error}"
    );
    let model = include_str!("fixtures/pyhf_model.json");
    let error = flatppl_hs3::read_pyhf_with(model, &select("first")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("a pyhf model has no measurements to select"),
        "{error}"
    );
}

#[test]
fn external_observations_leave_the_data_to_the_loader() {
    let text = convert(&PyhfOptions {
        observations: Observations::External,
        ..PyhfOptions::default()
    });
    assert!(
        text.contains("b_observed = external(cartpow(nonnegreals, 2))"),
        "{text}"
    );
    assert!(
        text.contains("a_observed = external(cartpow(nonnegreals, 1))"),
        "{text}"
    );
    // Loaded counts need not be integers, so the observation model is the
    // continued Poisson that bound non-integer counts also get.
    let bound = convert(&PyhfOptions::default())
        .replace("hepphys.ContinuedPoisson", "Poisson")
        .replace("Per-bin Poisson", "Per-bin continued-Poisson")
        .replace("broadcast(Poisson,", "broadcast(hepphys.ContinuedPoisson,");
    let strip = |text: &str| {
        text.lines()
            .filter(|line| !line.contains("_observed = "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&text), strip(&bound));
}

#[test]
fn a_wrong_length_override_is_refused() {
    let mut source: serde_json::Value = serde_json::from_str(WORKSPACE).unwrap();
    source["measurements"][0]["config"]["parameters"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"name": "sh", "inits": [1.0]}));
    let error =
        flatppl_hs3::read_pyhf_with(&source.to_string(), &PyhfOptions::default()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("configuration sets 1 `inits` value(s) on parameter `sh`, which has 2"),
        "{error}"
    );
}

fn external_centres() -> PyhfOptions {
    PyhfOptions {
        auxiliary: Auxiliary::External,
        ..PyhfOptions::default()
    }
}

#[test]
fn external_centres_follow_pyhf_auxdata_order() {
    let text = convert(&external_centres());
    // pyhf: `config.auxdata_order` is ['z', 'lumi', 'sh', 'stat_a', 'stat_b'],
    // the constrained parameters of `par_order`, for both measurements.
    assert_eq!(
        line(&text, "pyhf_auxdata"),
        "pyhf_auxdata = external(cartprod(z = reals, lumi = reals, \
         sh = cartpow(nonnegreals, 2), stat_a = cartpow(reals, 1), stat_b = cartpow(reals, 2)))"
    );
    // Every constraint reads its centre, including the zero-error staterror bin.
    for param in ["z", "sh", "lumi", "stat_a", "stat_b"] {
        assert!(
            line(&text, &format!("{param}_constraint_likelihood"))
                .ends_with(&format!("get(pyhf_auxdata, \"{param}\"))")),
            "{param}"
        );
    }
    assert!(text.contains(
        "cat([get(pyhf_auxdata, \"z\")], [get(pyhf_auxdata, \"lumi\")], \
         get(pyhf_auxdata, \"stat_b\"), get(pyhf_auxdata, \"stat_a\")))"
    ));
    // The nominal centres stay in the settings record.
    assert!(text.contains("auxdata = [25.0, 1.0]"));
    // Only the centres change.
    let strip = |text: &str| {
        text.lines()
            .filter(|line| {
                !line.contains("pyhf_auxdata")
                    && !line.contains("constraint_likelihood = ")
                    && !line.contains("constraints_likelihood = ")
                    && !line.contains("Auxiliary-measurement centres")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&text), strip(&convert(&PyhfOptions::default())));
}

#[test]
fn external_centres_avoid_a_parameter_name() {
    let source = WORKSPACE.replace("\"k\"", "\"pyhf_auxdata\"");
    let text = print_with(
        &flatppl_hs3::read_pyhf_with(&source, &external_centres()).unwrap(),
        Syntax::Minimal,
    );
    assert!(text.contains("pyhf_auxdata = elementof(reals)"), "{text}");
    assert!(
        text.contains("pyhf_auxdata_2 = external(cartprod(z = reals,"),
        "{text}"
    );
    assert!(text.contains("likelihoodof(z_constraint, get(pyhf_auxdata_2, \"z\"))"));
}

#[test]
fn a_model_without_constraints_has_no_centres() {
    let model = r#"{"channels": [{"name": "c", "samples": [
        {"name": "s", "data": [5.0], "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null}]}]}]}"#;
    let text = print_with(
        &flatppl_hs3::read_pyhf_with(model, &external_centres()).unwrap(),
        Syntax::Minimal,
    );
    assert!(!text.contains("pyhf_auxdata"), "{text}");
}
