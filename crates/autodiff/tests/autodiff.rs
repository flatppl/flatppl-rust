use flatppl_autodiff::{determinize_query, gradient};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn lower(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    assert_inferred(&mut module);
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}

fn assert_inferred(module: &mut Module) {
    let diagnostics = flatppl_infer::infer(module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
}

#[test]
fn takes_the_gradient_before_optimization() {
    let source = "
x = elementof(reals)
offset = 2.0 * 3.0
y = x + offset
inputs = x
outputs = y
";
    let module = lower(source);
    let differentiated = gradient(&module, &["x"]).unwrap();
    let text = flatppl_syntax::print(&differentiated);
    assert!(text.contains("offset = 2.0 * 3.0"), "{text}");
    assert!(text.contains("outputs = (y, d_x)"), "{text}");
    assert!(text.contains("d_x = 1.0"), "{text}");
    assert!(!flatppl_syntax::print(&module).contains("d_x"));
    let mut reparsed = flatppl_syntax::parse(&text).unwrap();
    assert_inferred(&mut reparsed);
    flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
}

#[test]
fn repeated_uses_contribute_once_per_operand() {
    let module = lower(
        "
x = elementof(reals)
square = x * x
score = square + square
inputs = x
outputs = score
",
    );
    let text = flatppl_syntax::print(&gradient(&module, &["x"]).unwrap());
    assert!(text.contains("_ad_bar = 1.0 + 1.0"), "{text}");
    assert!(text.contains("d_x = _ad_bar * x + _ad_bar * x"), "{text}");
}

#[test]
fn fixed_and_derived_inputs_cut_the_graph() {
    let module = lower(
        "
x = elementof(reals)
derived = x * x
alias = derived
score = alias + abs(-2.0)
inputs = (x, alias)
outputs = score
",
    );
    let text = flatppl_syntax::print(&gradient(&module, &["alias", "x"]).unwrap());
    assert!(text.contains("inputs = (x, alias)"), "{text}");
    assert!(text.contains("outputs = (score, d_alias, d_x)"), "{text}");
    assert!(text.contains("d_alias = 1.0"), "{text}");
    assert!(text.contains("d_x = 0.0"), "{text}");
}

#[test]
fn normal_parameters_are_matched_by_name() {
    let module = lower(
        "
mean = elementof(reals)
scale = elementof(posreals)
obs = external(reals)
score = logdensityof(Normal(sigma = scale, mu = mean), obs)
inputs = (mean, scale, obs)
outputs = score
",
    );
    let text = flatppl_syntax::print(&gradient(&module, &["obs", "scale", "mean"]).unwrap());
    assert!(
        text.contains("outputs = (score, d_obs, d_scale, d_mean)"),
        "{text}"
    );
    assert_eq!(text.matches("(obs - mean) / scale").count(), 4, "{text}");
    let mut reparsed = flatppl_syntax::parse(&text).unwrap();
    assert_inferred(&mut reparsed);
}

#[test]
fn rejects_an_active_operation_without_a_rule() {
    let module = lower(
        "
x = elementof(reals)
score = bernstein([1.0, 2.0], invlogit(x))
inputs = x
outputs = score
",
    );
    let error = gradient(&module, &["x"]).unwrap_err();
    assert_eq!(error.to_string(), "no pullback for `bernstein` operand 2");
    assert!(error.node.is_some());
}

#[test]
fn rejects_duplicate_input_aliases_and_missing_selections() {
    let aliased = lower(
        "
x = elementof(reals)
y = x
inputs = (x, y)
outputs = x
",
    );
    assert!(
        gradient(&aliased, &["x"])
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    let valid = lower("x = elementof(reals)\ninputs = x\noutputs = x\n");
    assert!(
        gradient(&valid, &["y"])
            .unwrap_err()
            .to_string()
            .contains("not a declared query input")
    );
}

#[test]
fn broadcast_gradients_match_input_shapes() {
    let module = lower(
        "
x = elementof(cartpow(reals, [2, 3]))
b = elementof(cartpow(reals, 3))
unused = elementof(cartpow(reals, [1, 3]))
z = x .+ addaxes(b, 1, 0)
inputs = (x, b, unused)
outputs = sum(z .* z)
",
    );
    let result = gradient(&module, &["b", "x", "unused"]).unwrap();
    let printed = flatppl_syntax::print(&result);
    assert!(
        printed.contains("d_unused = fill(0.0, [1, 3])"),
        "{printed}"
    );
    assert!(printed.contains("d_b = _ad_bar"), "{printed}");
    let mut reparsed = flatppl_syntax::parse(&printed).unwrap();
    assert_inferred(&mut reparsed);
    flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
}

#[test]
fn repeated_indices_use_a_reduction() {
    let module = lower(
        "
x = elementof(cartpow(reals, 4))
v = x[[2, 2, 4]]
inputs = x
outputs = sum(v .* v)
",
    );
    let result = gradient(&module, &["x"]).unwrap();
    let printed = flatppl_syntax::print(&result);
    assert!(printed.contains("d_x[.ad0] :="), "{printed}");
    assert!(printed.contains(".ad1"), "{printed}");
    let mut reparsed = flatppl_syntax::parse(&printed).unwrap();
    assert_inferred(&mut reparsed);
}

#[test]
fn directional_rules_keep_both_primal_outputs() {
    let module = lower(
        "
x = elementof(cartpow(reals, 3))
b = elementof(reals)
y = x .* b
z = sum(x .* x)
inputs = (x, b)
outputs = (y, z)
",
    );
    let forward = flatppl_autodiff::jvp(&module, &["x", "b"]).unwrap();
    let text = flatppl_syntax::print(&forward);
    assert!(
        text.contains("inputs = (x, b, tangent_x, tangent_b)"),
        "{text}"
    );
    assert!(
        text.contains("outputs = (y, z, d_output_1, d_output_2)"),
        "{text}"
    );
    assert!(
        text.contains("tangent_x = elementof(cartpow(reals, 3))"),
        "{text}"
    );

    let reverse = flatppl_autodiff::vjp(&module, &["x", "b"]).unwrap();
    let text = flatppl_syntax::print(&reverse);
    assert!(
        text.contains("inputs = (x, b, cotangent_1, cotangent_2)"),
        "{text}"
    );
    assert!(text.contains("outputs = (y, z, d_x, d_b)"), "{text}");
    assert!(
        text.contains("cotangent_1 = elementof(cartpow(reals, 3))"),
        "{text}"
    );
}

#[test]
fn conditional_masks_the_cotangent_before_it_reaches_a_singular_branch() {
    let module = lower(
        "
x = elementof(reals)
lp = ifelse(x > 0.0, log(x), log(-x))
inputs = x
outputs = lp
",
    );
    let text = flatppl_syntax::print(&gradient(&module, &["x"]).unwrap());
    assert!(text.contains("iszero(_ad_bar"), "{text}");
    let mut reparsed = flatppl_syntax::parse(&text).unwrap();
    assert_inferred(&mut reparsed);
    flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
}

#[test]
fn shape_reads_do_not_make_values_active() {
    let module = lower(
        "
x = elementof(cartpow(reals, 3))
lp = sum(x) + real(lengthof(x))
inputs = x
outputs = lp
",
    );
    let text = flatppl_syntax::print(&gradient(&module, &["x"]).unwrap());
    assert!(text.contains("d_x = fill(1.0, 3)"), "{text}");
}

#[test]
fn power_and_logsumexp_round_trip() {
    let module = lower(
        "
x = elementof(cartpow(reals, 3))
lp = logsumexp(x .^ 2)
inputs = x
outputs = lp
",
    );
    for transform in [gradient, flatppl_autodiff::jvp, flatppl_autodiff::vjp] {
        let text = flatppl_syntax::print(&transform(&module, &["x"]).unwrap());
        let mut reparsed = flatppl_syntax::parse(&text).unwrap();
        assert_inferred(&mut reparsed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn beta_shape_gradients_broadcast_special_functions() {
    let module = lower(
        r#"
alpha = elementof(posreals)
beta = elementof(cartpow(posreals, 2))
x = elementof(cartpow(reals, 2))
score = logdensityof(Beta.(alpha, beta), x)
inputs = (alpha, beta, x)
outputs = score
"#,
    );
    let result = gradient(&module, &["alpha", "beta", "x"]).unwrap();
    let text = flatppl_syntax::print(&result);
    assert!(text.contains(".digamma."), "{text}");
    let mut reparsed = flatppl_syntax::parse(&text).unwrap();
    assert_inferred(&mut reparsed);
    flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
}

#[test]
fn multivariate_density_rules_preserve_directional_shapes() {
    let module = lower(
        r#"
alpha = elementof(cartpow(posreals, 2))
simplex = elementof(stdsimplex(2))
mu = elementof(cartpow(reals, 2))
cov = elementof(cartpow(reals, [2, 2]))
x = elementof(cartpow(reals, 2))
score = logdensityof(Dirichlet(alpha), simplex) + logdensityof(MvNormal(mu, cov), x)
inputs = (alpha, simplex, mu, cov, x)
outputs = score
"#,
    );
    let wrt = ["alpha", "simplex", "mu", "cov", "x"];
    for result in [
        gradient(&module, &wrt).unwrap(),
        flatppl_autodiff::vjp(&module, &wrt).unwrap(),
        flatppl_autodiff::jvp(&module, &wrt).unwrap(),
    ] {
        let text = flatppl_syntax::print(&result);
        let mut reparsed = flatppl_syntax::parse(&text).unwrap();
        assert_inferred(&mut reparsed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn empty_selections_follow_the_primal_and_tangent_abis() {
    let input = lower("primal = 2.0\noutputs = primal\n");
    let gradient = flatppl_syntax::print(&gradient(&input, &[]).unwrap());
    assert!(gradient.contains("outputs = primal"), "{gradient}");
    assert!(!gradient.contains("inputs"), "{gradient}");

    let tangent = flatppl_syntax::print(&flatppl_autodiff::jvp(&input, &[]).unwrap());
    assert!(
        tangent.contains("outputs = (primal, d_output_1)"),
        "{tangent}"
    );
    assert!(tangent.contains("d_output_1 = 0.0"), "{tangent}");
    assert!(!tangent.contains("inputs"), "{tangent}");

    let cotangent = flatppl_syntax::print(&flatppl_autodiff::vjp(&input, &[]).unwrap());
    assert!(cotangent.contains("inputs = cotangent_1"), "{cotangent}");
    assert!(cotangent.contains("outputs = primal"), "{cotangent}");
}

#[test]
fn poisson_observation_is_discrete_even_with_a_real_abi() {
    let module = lower(
        "
rate = elementof(posreals)
observed = elementof(reals)
score = logdensityof(Poisson(rate), observed)
inputs = (rate, observed)
outputs = score
",
    );
    let error = gradient(&module, &["observed"]).unwrap_err();
    assert!(
        error.to_string().contains("discrete observation"),
        "{error}"
    );
    assert!(gradient(&module, &["rate"]).is_ok());
}
