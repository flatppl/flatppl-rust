use std::process::Command;

fn flatppl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
}

#[test]
fn determinize_lowers_a_gaussian_to_stdout() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("g.flatppl");
    std::fs::write(
        &input,
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("builtin_logdensityof"),
        "emitted FlatPDL:\n{stdout}"
    );
}

/// `densityof` has no dedicated `builtin_*` primitive (§07 lists six
/// `builtin_*`, only `builtin_logdensityof` for density); it lowers as
/// `exp(<the logdensityof lowering>)` (§06). Self-contained CLI-corpus
/// regression: a `densityof` query over a record-of-draws prior must lower
/// (exit 0) to FlatPDL containing both `exp(` and `builtin_logdensityof`,
/// mirroring `determinize_lowers_a_gaussian_to_stdout` above but for the
/// plain-density query form.
#[test]
fn determinize_lowers_a_densityof_query_to_stdout() {
    let dir =
        std::env::temp_dir().join(format!("flatppl-det-cli-densityof-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("d.flatppl");
    std::fs::write(
        &input,
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\nd = densityof(lawof(record(a = a)), record(a = 0.5))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("builtin_logdensityof"),
        "emitted FlatPDL:\n{stdout}"
    );
    assert!(
        stdout.contains("exp("),
        "densityof must lower to exp(<logdensity>); emitted FlatPDL:\n{stdout}"
    );
}

/// `determinize` resolves a `load_module` cross-module measure ref: the CLI
/// must assemble the `ModuleBundle` (same cache-only resolver as `infer`) and
/// pass it to `determinize_with`, or this refuses on an unresolved module ref
/// instead of lowering. Mirrors `crates/determinizer/tests/crossmodule_golden.rs`'s
/// `cross_module_likelihood_lowers`, but through the real CLI binary and
/// filesystem-resolved `load_module` (not an in-process `ModuleBundle`).
#[test]
fn determinize_resolves_cross_module_load_module() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-xmod-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("helpers.flatppl"),
        "flatppl_compat = \"0.1\"\n\
         center = elementof(reals)\n\
         obs_kernel = functionof(Normal(mu = center, sigma = 1.0), center = center)\n",
    )
    .unwrap();
    let input = dir.join("model.flatppl");
    std::fs::write(
        &input,
        "flatppl_compat = \"0.1\"\n\
         a = elementof(reals)\n\
         helpers = load_module(\"helpers.flatppl\", center = a)\n\
         input_data = 2.5\n\
         L = likelihoodof(helpers.obs_kernel, input_data)\n\
         lp = logdensityof(L, record(a = 0.0))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("builtin_logdensityof"),
        "emitted FlatPDL did not resolve the cross-module kernel:\n{stdout}"
    );
}

/// `determinize` resolves a `load_module` cross-module `%autoinputs`
/// (keyword/auto-traced) kernel APPLICATION: the submodule's boundary-less
/// `k = functionof(Normal(mu = center, sigma = 1.0))` auto-traces its input
/// `center`, and the host scores the keyword application `logdensityof(m.k(center
/// = 0.0), 0.5)`. Mirrors `determinize_resolves_cross_module_load_module`, but the
/// grafted kernel is keyword-only (§04): the CLI must graft, re-infer (repopulating
/// `auto_inputs_of` on the grafted node), and β-reduce the keyword bind — or this
/// refuses on the auto-traced boundary instead of lowering.
#[test]
fn determinize_resolves_cross_module_autoinputs_kernel() {
    let dir =
        std::env::temp_dir().join(format!("flatppl-det-cli-autoinputs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("helpers.flatppl"),
        "flatppl_compat = \"0.1\"\n\
         center = elementof(reals)\n\
         k = functionof(Normal(mu = center, sigma = 1.0))\n",
    )
    .unwrap();
    let input = dir.join("model.flatppl");
    std::fs::write(
        &input,
        "flatppl_compat = \"0.1\"\n\
         m = load_module(\"helpers.flatppl\")\n\
         lp = logdensityof(m.k(center = 0.0), 0.5)\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("builtin_logdensityof"),
        "emitted FlatPDL did not resolve the cross-module %autoinputs kernel application:\n{stdout}"
    );
}

/// Expected outcome for a fixture in the checked-in query corpus below.
enum Expect {
    /// `determinize` exits 0 and the emitted FlatPDL contains `builtin_logdensityof`.
    Lowers,
    /// `determinize` exits 3 (refuse) and stderr contains `determinize: refuse`.
    RefusesExit3,
}

/// Regression corpus over the checked-in `fixtures/flatppl/queries/*.flatppl`
/// query modules: each `load_module`s a real base fixture and queries a
/// cross-module handle (reified kernel, keyword/record joint prior, or
/// positional-constructor `normalize`), plus one documented spec-correct
/// refuse. Keeps the controller-verified "these lower on the real checked-in
/// base fixtures" claim durable — a regression here means a query form that
/// used to lower (or refuse) now behaves differently.
#[test]
fn fixture_query_corpus_lowers_or_documented_refuse() {
    let corpus_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/flatppl/queries");
    let cases: &[(&str, Expect)] = &[
        ("bayesian_inference_1_likelihood.flatppl", Expect::Lowers),
        ("bayesian_inference_1_prior.flatppl", Expect::Lowers),
        ("eight_schools_prior.flatppl", Expect::Lowers),
        ("pushfwd_exp_lognormal.flatppl", Expect::Lowers),
        ("pushfwd_arbitrary_f_refuses.flatppl", Expect::RefusesExit3),
        ("nested_crossmodule.flatppl", Expect::Lowers),
        ("xmodule_kernel_application.flatppl", Expect::Lowers),
        ("bayesian_inference_1_posterior.flatppl", Expect::Lowers),
        ("bayesian_inference_2_posterior.flatppl", Expect::Lowers),
        ("eight_schools_posterior.flatppl", Expect::Lowers),
        ("bayesian_inference_3_posterior.flatppl", Expect::Lowers),
        ("bayesian_inference_4_posterior.flatppl", Expect::Lowers),
    ];
    for (filename, expect) in cases {
        let path = corpus_dir.join(filename);
        let out = flatppl().arg("determinize").arg(&path).output().unwrap();
        match expect {
            Expect::Lowers => {
                assert!(
                    out.status.success(),
                    "{filename}: expected exit 0; stderr: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let stdout = String::from_utf8_lossy(&out.stdout);
                assert!(
                    stdout.contains("builtin_logdensityof"),
                    "{filename}: expected `builtin_logdensityof` in stdout:\n{stdout}"
                );
            }
            Expect::RefusesExit3 => {
                assert_eq!(
                    out.status.code(),
                    Some(3),
                    "{filename}: expected refuse exit 3; stderr: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let stderr = String::from_utf8_lossy(&out.stderr);
                assert!(
                    stderr.contains("determinize: refuse"),
                    "{filename}: expected `determinize: refuse` in stderr:\n{stderr}"
                );
            }
        }
    }
}

/// Pass 4 Task A review Fix 2: `--keep <name>` (repeatable) runs root-based DCE
/// (Buffy #263 Pass 4-A) through the real CLI binary. Without `--keep`, the
/// unreachable `dead1 = 42.0` binding survives (zeroed-or-as-is, current
/// keep-all behavior); with `--keep __score__`, only `__score__` (and its
/// transitive deps — here none, since it lowers to a self-contained
/// `builtin_logdensityof` call) survives, and the unreachable `dead1` binding
/// is dropped entirely from the emitted FlatPPL.
#[test]
fn determinize_keep_flag_drops_unreachable_binding() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-keep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("m.flatppl");
    std::fs::write(
        &input,
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         dead1 = 42.0\n\
         __score__ = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    )
    .unwrap();

    // No `--keep`: today's keep-all behavior — the unreachable binding survives.
    let out_all = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out_all.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_all.stderr)
    );
    let stdout_all = String::from_utf8_lossy(&out_all.stdout);
    assert!(
        stdout_all.contains("dead1"),
        "control: no --keep must retain the unreachable binding:\n{stdout_all}"
    );

    // `--keep __score__`: root-based DCE drops `dead1`, keeps the requested root.
    let out_keep = flatppl()
        .arg("determinize")
        .arg("--keep")
        .arg("__score__")
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        out_keep.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out_keep.stderr)
    );
    let stdout_keep = String::from_utf8_lossy(&out_keep.stdout);
    assert!(
        stdout_keep.contains("__score__") && stdout_keep.contains("builtin_logdensityof"),
        "--keep __score__ must keep the requested root:\n{stdout_keep}"
    );
    assert!(
        !stdout_keep.contains("dead1"),
        "--keep __score__ must drop the unreachable binding:\n{stdout_keep}"
    );
}

#[test]
fn determinize_refuses_with_exit_3() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("r.flatppl");
    // A continuous-latent kchain marginal where the latent feeds the likelihood
    // SCALE (non-conjugate — no closed-form marginal) → refuse. (A latent feeding
    // the Normal mean is the Normal–Normal conjugate case, which now lowers.)
    std::fs::write(
        &input,
        "z = draw(Normal(mu = 0.0, sigma = 1.0))\nk = kernelof(record(y = draw(Normal(mu = 1.0, sigma = z))), z = z)\npp = kchain(lawof(record(z = z)), k)\nlp = logdensityof(pp, record(y = 0.5))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(3),
        "refuse must exit 3; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("determinize: refuse"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
