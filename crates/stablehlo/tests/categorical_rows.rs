use flatppl_infer::Severity;
use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target, emit};

fn query(base: i64, selected_rows: [i64; 2], runtime_selector: bool) -> String {
    let rows = [selected_rows[0]; 64]
        .into_iter()
        .chain([selected_rows[1]; 64])
        .collect::<Vec<_>>();
    let inputs = if runtime_selector { "(p, docs)" } else { "p" };
    let constructor = if base == 0 {
        "Categorical0"
    } else {
        "Categorical"
    };
    let get = if base == 0 { "get0" } else { "get" };
    format!(
        "p = external(cartpow(stdsimplex(3), 4))\n\
         docs = {rows:?}\n\
         y = fill({base}, 128)\n\
         inputs = {inputs}\n\
         outputs = sum(builtin_logdensityof.({constructor}, broadcast(record, p = {get}(p, docs)), y))\n"
    )
}

fn lower(source: &str, opts: &EmitOptions) -> String {
    let mut m = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    emit(&m, Mode::LogDensity, opts).unwrap()
}

#[test]
fn selected_simplex_verdicts_keep_first_occurrence_order() {
    for (base, rows) in [(0, [2, 0]), (1, [3, 1])] {
        let source = query(base, rows, false);
        let enabled = lower(
            &source,
            &EmitOptions {
                dtype: Dtype::F64,
                ..Default::default()
            },
        );
        let disabled = lower(
            &source,
            &EmitOptions {
                dtype: Dtype::F64,
                sufficient_statistics: false,
                ..Default::default()
            },
        );
        assert!(
            enabled.contains("dense<[2, 0]> : tensor<2xi64>"),
            "{enabled}"
        );
        assert!(
            enabled
                .lines()
                .any(|line| line.contains("stablehlo.log") && line.ends_with("tensor<128xf64>"))
        );
        assert!(enabled.contains("flatppl.runtime_status = 1"));
        assert!(!disabled.contains("tensor<2xi64>"));
    }
}

#[test]
fn live_rows_and_other_check_profiles_keep_the_original_validation() {
    let live = query(0, [2, 0], true);
    let original = lower(
        &live,
        &EmitOptions {
            sufficient_statistics: false,
            ..Default::default()
        },
    );
    assert_eq!(lower(&live, &EmitOptions::default()), original);
    let references = query(0, [2, 0], false);
    for (target, check_parameters) in [(Target::Gpu, true), (Target::Cpu, false)] {
        let opts = EmitOptions {
            target,
            check_parameters,
            ..Default::default()
        };
        let original = lower(
            &references,
            &EmitOptions {
                sufficient_statistics: false,
                ..opts
            },
        );
        assert_eq!(lower(&references, &opts), original);
    }
}
