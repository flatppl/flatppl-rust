use flatppl_infer::Severity;
use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target, emit};

fn query(extra_input: &str, extra_output: &str, last_count: i64) -> String {
    let sites = [vec![1; 45], vec![2; 45], vec![3; 6]].concat();
    let observed = [vec![0; 45], vec![2; 45], vec![last_count; 6]].concat();
    format!(
        "\
log_rate = external(cartpow(reals, 3))
sites = {sites:?}
obs = {observed:?}
inputs = (log_rate{extra_input})
rate = exp.(log_rate[sites])
cells = builtin_logdensityof.(Poisson, broadcast(record, rate = rate), obs)
outputs = (sum(cells){extra_output})
"
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
fn grouped_poisson_preserves_a_lazy_native_fallback() {
    let source = query("", "", 64);
    for dtype in [Dtype::F32, Dtype::F64] {
        let mlir = lower(
            &source,
            &EmitOptions {
                dtype,
                ..Default::default()
            },
        );
        let width = match dtype {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
        };
        assert!(mlir.contains("stablehlo.case"));
        assert!(mlir.contains(&format!("tensor<96x{width}>")));
        assert!(mlir.contains(&format!("tensor<6x{width}>")));
        assert!(mlir.contains(&format!("tensor<2x{width}>")));
    }
}

#[test]
fn grouped_poisson_checks_only_used_sites_of_a_wide_source() {
    let source = query("", "", 64).replace("cartpow(reals, 3)", "cartpow(reals, 32768)");
    let mlir = lower(&source, &EmitOptions::default());
    assert!(mlir.contains("stablehlo.case"));
    let guard = mlir
        .lines()
        .find(|line| line.contains("stablehlo.abs"))
        .unwrap();
    assert!(guard.contains("tensor<3xf32>"), "{guard}");
    assert!(!guard.contains("%arg0"), "{guard}");
}

#[test]
fn grouped_poisson_excludes_runtime_rows_shared_cells_and_other_profiles() {
    for source in [
        query(", sites", "", 64),
        query("", ", cells", 64),
        query("", "", 256),
    ] {
        assert!(!lower(&source, &EmitOptions::default()).contains("stablehlo.case"));
    }
    let source = query("", "", 64);
    for opts in [
        EmitOptions {
            target: Target::Gpu,
            ..Default::default()
        },
        EmitOptions {
            sufficient_statistics: false,
            ..Default::default()
        },
    ] {
        assert!(!lower(&source, &opts).contains("stablehlo.case"));
    }
}
