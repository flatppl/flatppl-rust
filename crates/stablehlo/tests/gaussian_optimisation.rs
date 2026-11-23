use flatppl_stablehlo::{Dtype, EmitError, EmitOptions, Mode, emit};

fn emit_query(source: &str, options: &EmitOptions) -> Result<String, EmitError> {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    emit(&model, Mode::LogDensity, options)
}

#[test]
fn fixed_normal_cdf_endpoints_emit_the_exact_literals() {
    let input = "theta = external(reals)\ninputs = theta\n";
    let cdf = "builtin_touniform(Normal, record(mu = 2.5, sigma = 1.5),";
    let source = format!("{input}outputs = [theta, {cdf} -inf), {cdf} inf)]\n");
    let expected = format!("{input}outputs = [theta, 0.0, 1.0]\n");
    for dtype in [Dtype::F32, Dtype::F64] {
        let options = EmitOptions {
            dtype,
            ..Default::default()
        };
        assert_eq!(
            emit_query(&source, &options).unwrap(),
            emit_query(&expected, &options).unwrap()
        );
    }
}

#[test]
fn normal_cdf_retains_promoted_scale_runtime_status() {
    let source = "\
        _positive = 1.5\n\
        theta = _positive\n\
        inputs = theta\n\
        outputs = builtin_touniform(Normal, record(mu = 2.5, sigma = theta), -inf)\n";
    let options = EmitOptions {
        dtype: Dtype::F64,
        ..Default::default()
    };
    let checked = emit_query(source, &options).unwrap();
    assert!(checked.contains("chlo.erfc"), "{checked}");
    assert!(
        checked.contains("-> (tensor<f64>, tensor<i64>)"),
        "{checked}"
    );
    assert!(checked.contains("stablehlo.constant dense<9>"), "{checked}");
}

#[test]
fn normal_cdf_endpoint_preserves_invalid_fixed_scale_error() {
    let source = "\
        theta = external(reals)\n\
        inputs = theta\n\
        outputs = [theta, builtin_touniform(Normal, record(mu = 2.5, sigma = 0.0), inf)]\n";
    let error = emit_query(source, &EmitOptions::default()).unwrap_err();
    assert!(error.to_string().contains("sigma"), "{error}");
}
