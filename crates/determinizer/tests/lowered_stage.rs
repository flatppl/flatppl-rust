//! The lowered stage projects a determinized module and rebuilds it unchanged.

use flatppl_determinizer::{LoweringOptions, RulePolicy, determinize_with_options};

fn lower(source: &str) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(source).unwrap_or_else(|e| panic!("{source}\n{e:?}"));
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        !diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error),
        "{source}\n{diagnostics:?}"
    );
    determinize_with_options(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        None,
        &LoweringOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{source}\n{e:?}"))
}

/// Rebuild every projected call and compare the printed modules.
fn round_trip(lowered: &flatppl_core::Module) -> usize {
    let (rebuilt, calls) =
        flatppl_optimise::lowered_roundtrip(lowered, None).expect("the lowered stage projects");
    assert_eq!(
        flatppl_syntax::print(&rebuilt),
        flatppl_syntax::print(lowered)
    );
    calls
}

const DESIGN: &str = "x = rowstack([[1.0, 0.5], [0.2, -0.3], [0.7, 1.1]])\n";

#[test]
fn lowered_laws_and_their_links_rebuild_unchanged() {
    let models = [
        // A link law in a broadcast: density, record and link heads.
        format!(
            "{DESIGN}b = elementof(cartpow(reals, 2))\ninputs = b\n\
             outputs = logdensityof(Bernoulli.(invlogit.(x * b)), [1, 0, 1])\n"
        ),
        format!(
            "{DESIGN}a = elementof(reals)\nb = elementof(cartpow(reals, 2))\ninputs = (a, b)\n\
             outputs = logdensityof(Binomial.(5, invlogit.(a .+ x * b)), [2, 0, 5])\n"
        ),
        format!(
            "{DESIGN}a = elementof(reals)\nb = elementof(cartpow(reals, 2))\ninputs = (a, b)\n\
             outputs = logdensityof(Poisson.(exp.(a .+ x * b)), [0, 2, 3])\n"
        ),
        // Scalar laws with a link and with a softmax.
        "eta = elementof(reals)\ninputs = eta\n\
         outputs = logdensityof(Bernoulli(invlogit(eta)), 1)\n"
            .to_owned(),
        "v = elementof(cartpow(reals, 3))\ninputs = v\n\
         outputs = logdensityof(Categorical(softmax(v)), 2)\n"
            .to_owned(),
        // A §09 law with four record fields.
        format!(
            "{DESIGN}reg = standard_module(\"regression\", \"0.1\")\n\
             a = elementof(reals)\nb = elementof(cartpow(reals, 2))\ns = elementof(posreals)\n\
             inputs = (a, b, s)\n\
             outputs = logdensityof(reg.NormalGLM(x, a, b, s), [0.5, 1.0, -0.2])\n"
        ),
        // A joint likelihood over a record of draws.
        "mu = elementof(reals)\ns = elementof(posreals)\ninputs = (mu, s)\n\
         outputs = logdensityof(lawof(record(a = draw(Normal(mu, s)), b = draw(Exponential(s)))), \
         record(a = 0.3, b = 1.2))\n"
            .to_owned(),
    ];
    let mut calls = 0;
    for source in &models {
        let lowered = lower(source);
        let rebuilt = round_trip(&lowered);
        assert!(
            rebuilt > 0,
            "nothing projected:\n{}",
            flatppl_syntax::print(&lowered)
        );
        calls += rebuilt;
    }
    assert!(calls >= models.len() * 2, "{calls}");
}

#[test]
fn the_lowered_stage_does_not_run_without_a_ruleset_to_match() {
    let lowered = lower(&format!(
        "{DESIGN}b = elementof(cartpow(reals, 2))\ninputs = b\n\
         outputs = logdensityof(Bernoulli.(invlogit.(x * b)), [1, 0, 1])\n"
    ));
    assert!(flatppl_optimise::optimise_lowered(&lowered, None, &RulePolicy::default()).is_none());
    // Forced, the run finds no cheaper form and keeps the module.
    assert!(
        flatppl_optimise::optimise_lowered_forced(&lowered, None, &RulePolicy::default())
            .is_none_or(|out| flatppl_syntax::print(&out) == flatppl_syntax::print(&lowered))
    );
}

/// Round trip and forced-run time over printed lowered modules, one `.flatppl`
/// or `.out` file each. `FLATPPL_LOWERED_CORPUS` lists their directories,
/// separated by `:`.
#[test]
#[ignore = "reads a corpus of lowered texts named by FLATPPL_LOWERED_CORPUS"]
fn a_corpus_of_lowered_modules_rebuilds_unchanged() {
    let dirs = std::env::var("FLATPPL_LOWERED_CORPUS")
        .expect("FLATPPL_LOWERED_CORPUS names the corpus directories");
    let mut files = 0;
    let mut projected = 0;
    let mut mismatches = vec![];
    let mut unreadable = vec![];
    let mut times = vec![];
    for dir in dirs.split(':').filter(|d| !d.is_empty()) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|ext| ext == "flatppl" || ext == "out")
            })
            .collect();
        entries.sort();
        for path in entries {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            let Ok(mut module) = flatppl_syntax::parse(&text) else {
                unreadable.push(path.display().to_string());
                continue;
            };
            let _ = flatppl_infer::infer(&mut module);
            files += 1;
            match flatppl_optimise::lowered_roundtrip(&module, None) {
                Some((rebuilt, calls)) => {
                    if flatppl_syntax::print(&rebuilt) != flatppl_syntax::print(&module) {
                        mismatches.push(path.display().to_string());
                    }
                    if calls > 0 {
                        projected += 1;
                    }
                }
                None => unreadable.push(format!("{} (no projection)", path.display())),
            }
            let start = std::time::Instant::now();
            let _ =
                flatppl_optimise::optimise_lowered_forced(&module, None, &RulePolicy::default());
            times.push((start.elapsed(), path.display().to_string()));
        }
    }
    times.sort();
    let total: std::time::Duration = times.iter().map(|(t, _)| *t).sum();
    eprintln!(
        "lowered corpus: {files} modules, {projected} with projected calls, {} mismatches, \
         {} unreadable or not projected",
        mismatches.len(),
        unreadable.len()
    );
    eprintln!("forced lowered stage: total {total:?}");
    for (time, path) in times.iter().rev().take(5) {
        eprintln!("  {time:?} {path}");
    }
    for path in &unreadable {
        eprintln!("  not projected: {path}");
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}
