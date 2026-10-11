use flatppl_core::Module;

fn emit(source: &str) -> String {
    emit_mode(source, flatppl_stablehlo::Mode::LogDensity)
}

fn emit_mode(source: &str, mode: flatppl_stablehlo::Mode) -> String {
    let mut model: Module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    let restrict_enzyme_compatible = matches!(mode, flatppl_stablehlo::Mode::LogDensity);
    flatppl_stablehlo::emit(
        &model,
        mode,
        &flatppl_stablehlo::EmitOptions {
            restrict_enzyme_compatible,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Completing the square in n-1 free coordinates has determinant n.
/// At n=3, sigma=2, x=[-1,0,1]: log density -2.924865283195181,
/// d/dsigma=-0.75 and d/dmu=0, even when mu is too large to square.
#[test]
fn zerosum_normal_tilt_emits_the_coordinate_density() {
    let source = "mu = external(reals)\nsigma = external(posreals)\n\
                  x = external(cartpow(reals, 3))\n\
                  q = tilt(iid(Normal(mu, sigma), 3), Lebesgue(zerosum(3)))\n\
                  inputs = (mu, sigma, x)\noutputs = logdensityof(q, x)";
    let expected = source.replace("logdensityof(q, x)",
        "ifelse(x in zerosum(3), 0.5493061443340549 - 2.0*(log(sigma) + 0.9189385332046727) - 0.5*sum((x ./ sigma) .* (x ./ sigma)), -inf)");
    assert_eq!(emit(source), emit(&expected));
}

/// An ordered chamber has 1/n! of an iid continuous law's mass.
#[test]
fn ordered_normal_tilt_keeps_the_iid_scale_and_factorial() {
    let source = "mu = external(reals)\nsigma = external(posreals)\n\
                  x = external(cartpow(reals, 3))\n\
                  q = tilt(iid(Normal(mu, sigma), 3), Lebesgue(ordered(3)))\n\
                  inputs = (mu, sigma, x)\noutputs = logdensityof(q, x)";
    let expected = source.replace("logdensityof(q, x)",
        "ifelse(x in ordered(3), loggamma(4.0) - 3.0*(log(sigma) + 0.9189385332046727) - 0.5*sum(((x .- mu) ./ sigma) .* ((x .- mu) ./ sigma)), -inf)");
    assert_eq!(emit(source), emit(&expected));
}

/// The independent fixture gives -0.8082405307719451 for Exponential(1)
/// and -5.1161667816817715 for Gamma(2.5, 1), at [0.1, 0.5, 2.0].
#[test]
fn ordered_continuous_iid_tilt_adds_the_permutation_normalizer() {
    for law in ["Exponential(p)", "Gamma(p, 1)"] {
        let source = format!(
            "p = external(posreals)\nx = external(cartpow(reals, 3))\n\
             source_law = iid({law}, 3)\nq = tilt(source_law, Lebesgue(ordered(3)))\n\
             inputs = (p, x)\noutputs = logdensityof(q, x)"
        );
        let expected = source.replace(
            "logdensityof(q, x)",
            "ifelse(x in ordered(3), logdensityof(source_law, x) + loggamma(4.0), -inf)",
        );
        assert_eq!(emit(&source), emit(&expected));
        let sample = source.replace(
            "outputs = logdensityof(q, x)",
            "outputs = rand(q, rnginit(0))",
        );
        let ir = emit_mode(&sample, flatppl_stablehlo::Mode::Sample);
        assert_eq!(ir.matches("\"stablehlo.sort\"").count(), 1);
    }
}

/// The conditional Gaussian density is -1.6912035042980555 at
/// mu=[1,-2,0.5], sigma=[1,2,0.5], x=[0.3,-0.5,0.2].
#[test]
fn zerosum_independent_normals_use_the_sum_variance() {
    let source = "mu = external(cartpow(reals, 3))\n\
                  sigma = external(cartpow(posreals, 3))\n\
                  x = external(cartpow(reals, 3))\n\
                  q = tilt(Normal.(mu, sigma), Lebesgue(zerosum(3)))\n\
                  inputs = (mu, sigma, x)\noutputs = logdensityof(q, x)";
    let expected = source.replace("outputs = logdensityof(q, x)",
        "mu_v = mu .+ fill(0.0, 3)\nsigma_v = sigma .+ fill(0.0, 3)\n\
         scale = maximum(sigma_v)\nr = sigma_v ./ scale\n\
         v = r .* r\nS = sum(v)\n\
         center = mu_v .- v .* (sum(mu_v) / S)\n\
         delta = (x .- center) ./ sigma_v\n\
         outputs = ifelse(x in zerosum(3), log(scale) + 0.5*log(S) - sum(log.(sigma_v)) - 1.8378770664093453 - 0.5*sum(delta .* delta), -inf)");
    assert_eq!(emit(source), emit(&expected));
    let sample = source.replace(
        "outputs = logdensityof(q, x)",
        "outputs = rand(q, rnginit(0))",
    );
    let ir = emit_mode(&sample, flatppl_stablehlo::Mode::Sample);
    assert_eq!(ir.matches("stablehlo.rng_bit_generator").count(), 1);
}

#[test]
fn tilt_falls_back_to_generic_normalization() {
    for base in ["Normal(1, 3)", "Dirac(0.0)"] {
        let source = format!(
            "sigma = external(posreals)\nx = external(reals)\n\
                              q = tilt(Normal(0, sigma), {base})\n\
                              inputs = (sigma, x)\noutputs = logdensityof(q, x)"
        );
        let expanded = source.replace(
            &format!("tilt(Normal(0, sigma), {base})"),
            &format!("normalize(logweighted(fn(logdensityof(Normal(0, sigma), _)), {base}))"),
        );
        assert_eq!(emit(&source), emit(&expanded));
    }
}

#[test]
fn tilt_preserves_scale_through_the_exposed_joint_prior_and_kernel() {
    let source = "scale ~ Exponential(1)\n\
                  effects ~ tilt(iid(Normal(0, scale), 3), Lebesgue(zerosum(3)))\n\
                  prior = lawof(record(scale = scale, effects = effects))\n\
                  k = kernelof(effects, scale = scale)\n\
                  s = external(posreals)\nx = external(cartpow(reals, 3))\n\
                  inputs = (s, x)\n\
                  outputs = (logdensityof(prior, record(scale = s, effects = x)), logdensityof(k(s), x), logdensityof(k(2), [-1, 0, 1]))";
    let expected = "s = external(posreals)\nx = external(cartpow(reals, 3))\n\
                    q = tilt(iid(Normal(0, s), 3), Lebesgue(zerosum(3)))\n\
                    fixed_q = tilt(iid(Normal(0, 2), 3), Lebesgue(zerosum(3)))\n\
                    inputs = (s, x)\n\
                    outputs = (logdensityof(Exponential(1), s) + logdensityof(q, x), logdensityof(q, x), logdensityof(fixed_q, [-1, 0, 1]))";
    assert_eq!(emit(source), emit(expected));
}

#[test]
fn tilt_applies_kernel_operands_by_name() {
    let source = "a = elementof(reals)\nb = elementof(posreals)\n\
                  left = functionof(Normal(a, b), a = a, b = b)\n\
                  right = functionof(Normal(a, b), b = b, a = a)\n\
                  q = tilt(left, right)\n\
                  mu = external(reals)\nsigma = external(posreals)\nx = external(reals)\n\
                  inputs = (mu, sigma, x)\noutputs = logdensityof(q(mu, sigma), x)";
    assert_eq!(
        emit(source),
        emit(&source.replace("q(mu, sigma)", "tilt(Normal(mu, sigma), Normal(mu, sigma))"))
    );
}

#[test]
fn tilt_kernelof_conditions_source_and_closed_operands_ignore_inputs() {
    for (source, direct) in [
        (
            "a ~ Normal(0, 1)\nx ~ iid(Normal(a, 2), 3)\nk = kernelof(x, a = a)\nq = tilt(k, Lebesgue(ordered(3)))",
            "tilt(iid(Normal(mu, 2), 3), Lebesgue(ordered(3)))",
        ),
        (
            "a = elementof(reals)\nk = functionof(Normal(a, 2), a = a)\nq = tilt(Normal(0, 1), k)",
            "tilt(Normal(0, 1), Normal(mu, 2))",
        ),
        (
            "a = elementof(reals)\nx ~ Normal(a, 2)\nk = kernelof(x, a = a)\nq = tilt(Normal(0, 1), k)",
            "tilt(Normal(0, 1), Normal(mu, 2))",
        ),
    ] {
        let point = if direct.contains("iid") {
            "[-1, 0, 1]"
        } else {
            "0.5"
        };
        let suffix =
            format!("\nmu = external(reals)\ninputs = mu\noutputs = logdensityof(q(mu), {point})");
        assert_eq!(
            emit(&format!("{source}{suffix}")),
            emit(&format!("{source}{}", suffix.replace("q(mu)", direct)))
        );
    }
}

#[test]
fn closed_zerosum_sampling_centers_one_normal_vector_and_threads_its_rng() {
    let source = "state = rnginit(0)\nsigma = external(posreals)\n\
                  q = tilt(iid(Normal(1e300, sigma), 3), Lebesgue(zerosum(3)))\n\
                  value, next_state = rand(q, state)\ninputs = sigma\noutputs = (value, next_state)";
    let expected = "state = rnginit(0)\nsigma = external(posreals)\n\
                    z, next_state = rand(iid(Normal(0.0, 1.0), 3), state)\n\
                    inputs = sigma\noutputs = ((z .- sum(z)/3.0) .* sigma, next_state)";
    let output = emit_mode(source, flatppl_stablehlo::Mode::Sample);
    assert_eq!(output, emit_mode(expected, flatppl_stablehlo::Mode::Sample));
    assert_eq!(output.matches("stablehlo.rng_bit_generator").count(), 1);
    assert_eq!(
        output,
        emit_mode(
            &source.replace("rand(q, state)", "rand(lawof(draw(q)), state)"),
            flatppl_stablehlo::Mode::Sample
        )
    );
}

#[test]
fn closed_ordered_sampling_uses_one_sort_and_one_rng_draw() {
    let source = "state = rnginit(0)\n\
                  q = tilt(iid(Normal(0, 2), 3), Lebesgue(ordered(3)))\n\
                  value, next_state = rand(q, state)\noutputs = (value, next_state)";
    let output = emit_mode(source, flatppl_stablehlo::Mode::Sample);
    assert_eq!(output.matches("stablehlo.rng_bit_generator").count(), 1);
    assert_eq!(output.matches("\"stablehlo.sort\"").count(), 1);
    assert!(output.contains("stablehlo.compare LT"));
    assert_eq!(
        output,
        emit_mode(
            &source.replace("rand(q, state)", "rand(lawof(draw(q)), state)"),
            flatppl_stablehlo::Mode::Sample
        )
    );

    let mut model = flatppl_syntax::parse(source).unwrap();
    flatppl_infer::infer(&mut model);
    let outputs = model.intern("outputs");
    let error = flatppl_determinizer::determinize_with_roots(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&[outputs]),
    )
    .unwrap_err();
    assert_eq!(
        error.construct, "FreeBareName",
        "the internal sort must not pass ordinary FlatPDL conformance"
    );
}

/// A captured scale is drawn once before its vector, including when the vector
/// is returned twice and consumed by a derived field. The exported successor
/// state must agree with explicit sequential draws.
#[test]
fn shared_tilt_prior_samples_in_dependency_order() {
    for (support, vector_sample) in [
        (
            "zerosum",
            "z, next_state = rand(iid(Normal(0.0, 1.0), 3), state1)\n\
             effects = (z .- sum(z)/3.0) .* scale",
        ),
        (
            "ordered",
            "effects, next_state = rand(tilt(iid(Normal(0, scale), 3), Lebesgue(ordered(3))), state1)",
        ),
    ] {
        let fields =
            "record(scale = scale, effects = effects, same = effects, twice = effects .* 2)";
        let source = format!(
            "scale ~ Exponential(1)\n\
             effects ~ tilt(iid(Normal(0, scale), 3), Lebesgue({support}(3)))\n\
             prior = lawof({fields})\n\
             value = rand(prior, rnginit(0))\noutputs = value"
        );
        let expected = format!(
            "scale, state1 = rand(Exponential(1), rnginit(0))\n\
             {vector_sample}\noutputs = ({fields}, next_state)"
        );
        let output = emit_mode(&source, flatppl_stablehlo::Mode::Sample);
        assert_eq!(output.matches("stablehlo.rng_bit_generator").count(), 2);
        assert_eq!(
            output,
            emit_mode(&expected, flatppl_stablehlo::Mode::Sample)
        );
    }
}
