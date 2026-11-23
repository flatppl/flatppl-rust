use egglog::EGraph;

const BASE: &str = concat!(
    include_str!("../src/rules/schema.egg"),
    "\n",
    include_str!("../src/rules/portable.egg"),
    "\n",
    include_str!("../src/rules/gaussian.egg"),
    "\n",
    include_str!("../src/rules/conjugacy.egg"),
    "\n(unstable-combined-ruleset query-portable analysis portable gaussian conjugacy)\n",
);

const STABLEHLO: &str = concat!(
    include_str!("../src/rules/mathematical.egg"),
    "\n",
    include_str!("../src/rules/statistical.egg"),
    "\n",
    include_str!("../src/rules/simplification.egg"),
    "\n",
    include_str!("../src/rules/sharing.egg"),
    "\n",
    include_str!("../src/rules/hoisting.egg"),
    "\n(unstable-combined-ruleset query-gpu query-portable mathematical selector-facts sharing statistical simplification)\n",
    "(unstable-combined-ruleset query-cpu query-gpu hoisting)\n",
);

fn check_cases(graph: &EGraph, name: &str, program: &str, schedule: &str) {
    // Use the production schedule so positive and negative assertions cover
    // interactions among rules, including rules from other categories.
    let mut frame = graph.clone();
    let run = format!("(run-schedule (saturate (run {schedule})))");
    let mut fixture = String::new();
    let mut ran = false;
    for line in program.lines() {
        if line.starts_with("(run ") || line.starts_with("(run-schedule ") {
            if !ran {
                fixture.push_str(&run);
                fixture.push('\n');
                ran = true;
            }
        } else {
            fixture.push_str(line);
            fixture.push('\n');
        }
    }
    assert!(ran, "{name} contains no schedule to replace");
    frame
        .parse_and_run_program(Some(name.into()), &fixture)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
}

#[test]
fn portable_catalogue() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, BASE).unwrap();
    for (name, fixture) in [
        ("portable", include_str!("rules/portable_cases.egg")),
        ("gaussian", include_str!("rules/gaussian_cases.egg")),
        ("conjugacy", include_str!("rules/conjugacy_cases.egg")),
    ] {
        check_cases(&graph, name, fixture, "query-portable");
    }
}

#[test]
fn stablehlo_catalogue() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, BASE).unwrap();
    graph.parse_and_run_program(None, STABLEHLO).unwrap();
    for (name, fixture) in [
        ("mathematical", include_str!("rules/mathematical_cases.egg")),
        ("statistical", include_str!("rules/statistical_cases.egg")),
        (
            "simplification",
            include_str!("rules/simplification_cases.egg"),
        ),
        ("selection", include_str!("rules/selection_cases.egg")),
    ] {
        check_cases(&graph, name, fixture, "query-cpu");
    }
}
