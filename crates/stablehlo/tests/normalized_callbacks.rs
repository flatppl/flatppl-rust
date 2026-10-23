use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str, mode: &Mode) -> String {
    emit_with_integrals(source, mode, false)
}

fn emit_with_integrals(source: &str, mode: &Mode, numerical_integrals: bool) -> String {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        integration: numerical_integrals.then(Default::default),
        ..Default::default()
    };
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        match mode {
            Mode::LogDensity => Mode::LogDensity,
            Mode::Sample => Mode::Sample,
        },
        &options,
    )
    .unwrap()
}

#[test]
fn scalar_density_broadcast_preserves_the_scalar_call_shape() {
    for operation in ["densityof", "logdensityof"] {
        let source = format!(
            "theta = elementof(reals)\nx = external(reals)\ninputs = (theta,x)\n\
             outputs = broadcast({operation},Normal(theta,1),x)"
        );
        let expected = source.replace(
            &format!("broadcast({operation},Normal(theta,1),x)"),
            &format!("{operation}(Normal(theta,1),x)"),
        );
        assert_eq!(
            emit(&source, &Mode::LogDensity),
            emit(&expected, &Mode::LogDensity)
        );
    }
}

#[test]
fn callback_normalization_retains_scalar_weight_and_base_proofs() {
    let prefix = "mu = external(reals)\nsigma = external(reals)\nx = external(reals)\n\
                  row = (l, s) -> normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), \
                    locscale(StudentT(4), l, ifelse(s > 0 && isfinite(s), s, 1))))\n\
                  inputs = (mu, sigma, x)\n";
    let base = "normalize(weighted(ifelse(sigma > 0 && isfinite(sigma), 1, 0), \
                locscale(StudentT(4), mu, ifelse(sigma > 0 && isfinite(sigma), sigma, 1))))";
    for (mode, output) in [
        (Mode::LogDensity, "logdensityof(row(mu, sigma), x)"),
        (Mode::Sample, "rand(row(mu, sigma), rnginit(19))"),
    ] {
        let source = format!("{prefix}outputs = {output}");
        assert_eq!(
            emit(&source, &mode),
            emit(&source.replace("row(mu, sigma)", base), &mode),
        );
    }
}

#[test]
fn explicit_weights_retain_live_numerical_normalizers() {
    for weight in [
        "weighted((x,p) -> 1+p*x*x, Uniform(interval(0,1)), p)",
        "logweighted((x,p) -> log1p(p*x*x), Uniform(interval(0,1)), p)",
    ] {
        // Z(p)=1+p/3. At p=2,x=.5, log density is -.10536051565782634
        // and the derivative in p is -1/30.
        let source = format!(
            "p = external(posreals)\nx = external(reals)\n\
             M = normalize({weight})\ninputs = (p,x)\noutputs = logdensityof(M,x)"
        );
        emit_with_integrals(&source, &Mode::LogDensity, true);
    }
}

#[test]
fn explicit_pushforward_parameters_reach_both_density_maps_and_sampling() {
    let source = "p = external(cartprod(scale = posreals, shift = reals))\n\
                  y = external(reals)\n\
                  f(x, q) = q.scale * x + q.shift\n\
                  inverse(y, q) = (y - q.shift) / q.scale\n\
                  volume(x, q) = log(q.scale)\n\
                  M = pushfwd(bijection(f, inverse, volume), Normal(0, 1), p)\n\
                  inputs = (p, y)\n";
    for (mode, output) in [
        (Mode::LogDensity, "logdensityof(M, y)"),
        (Mode::Sample, "rand(M, rnginit(19))"),
    ] {
        // Passing p explicitly and resolving its fixed binding must agree.
        // Runtime probes compare the resulting Normal law and its gradient.
        let actual = emit(&format!("{source}outputs = {output}"), &mode);
        let fixed = source.replace(
            "pushfwd(bijection(f, inverse, volume), Normal(0, 1), p)",
            "pushfwd(bijection(x -> p.scale*x+p.shift, y -> (y-p.shift)/p.scale, x -> log(p.scale)), Normal(0, 1))",
        );
        let expected = emit(&format!("{fixed}outputs = {output}"), &mode);
        assert_eq!(actual, expected);
    }
}

#[test]
fn sequence_parameters_keep_record_fields_outside_the_iteration_axis() {
    // For p=(scale=2,shift=1), the scan is [2,7,18]; the final derivatives
    // are (11,7). The associative reduction a+x+shift is 8 at shift=1.
    let source = "p = external(cartprod(scale = reals, shift = reals))\n\
                  step(a, x, q) = q.scale * a + x + q.shift\n\
                  fold(a, x, q) = a + x + q.shift\ninputs = p\n\
                  outputs = (scan(step, 0.0, [1.0, 2.0, 3.0], p), \
                             reduce(fold, [1.0, 2.0, 3.0], p))";
    emit(source, &Mode::LogDensity);
    // A tuple is not an additional input of `scan` (spec §04), so only `reduce` takes it.
    let tuple = source
        .replace(
            "fold(a, x, q) = a + x + q.shift",
            "fold(a, x, q) = a + x + q[2]",
        )
        .replace(
            "reduce(fold, [1.0, 2.0, 3.0], p)",
            "reduce(fold, [1.0, 2.0, 3.0], (p.scale, p.shift))",
        );
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(&tuple, &Mode::LogDensity)
    );
    let nested = source
        .replace(
            "fold(a, x, q) = a + x + q.shift",
            "fold(a, x, q) = a + x + q[1].shift",
        )
        .replace(
            "reduce(fold, [1.0, 2.0, 3.0], p)",
            "reduce(fold, [1.0, 2.0, 3.0], (record(scale=p.scale,shift=p.shift),0.0))",
        );
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(&nested, &Mode::LogDensity)
    );
}

#[test]
fn analytic_pushforward_inversion_substitutes_the_parameter() {
    let source = "scale = external(posreals)\ny = external(reals)\ninputs = (scale, y)\n\
                  outputs = logdensityof(pushfwd((x,p) -> p*x, Normal(0,1), scale), y)";
    let closure = source.replace(
        "(x,p) -> p*x, Normal(0,1), scale",
        "x -> scale*x, Normal(0,1)",
    );
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(&closure, &Mode::LogDensity)
    );
    let builtin = source.replace("(x,p) -> p*x", "mul");
    assert_eq!(
        emit(&builtin, &Mode::LogDensity),
        emit(&closure, &Mode::LogDensity)
    );
    let source = "scale=external(posreals)\nshift=external(reals)\ny=external(reals)\n\
                  f(x,p)=p.scale*x+p.shift\ninputs=(scale,shift,y)\n\
                  outputs=logdensityof(pushfwd(f,Normal(0,1),record(scale=scale,shift=shift)),y)";
    let closure = source.replace(
        "pushfwd(f,Normal(0,1),record(scale=scale,shift=shift))",
        "pushfwd(x -> scale*x+shift,Normal(0,1))",
    );
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(&closure, &Mode::LogDensity)
    );
}

#[test]
fn sequence_object_parameters_specialize_to_live_value_leaves() {
    // At mu=.5, scan's final value is -7.1318155996140185, derivative 4.5.
    // The associative reduce is 6 + 2*logNormal(mu,1)(0), derivative -2*mu.
    for (operation, body, call) in [
        (
            "scan",
            "a + logdensityof(p,x)",
            "scan(step,0.0,[1.0,2.0,3.0],M)[3]",
        ),
        (
            "reduce",
            "a + x + logdensityof(p,0.0)",
            "reduce(step,[1.0,2.0,3.0],M)",
        ),
    ] {
        for declaration in ["elementof", "external"] {
            let source = format!(
                "mu = {declaration}(reals)\nM = Normal(mu,1.0)\nstep(a,x,p) = {body}\ninputs = mu\noutputs = {call}"
            );
            let actual = emit(&source, &Mode::LogDensity);
            assert!(
                actual.contains("stablehlo.while"),
                "{operation} retains one loop"
            );
            let expected = source
                .replace("logdensityof(p,", "logdensityof(Normal(p,1.0),")
                .replace(",M)", ",mu)");
            assert_eq!(actual, emit(&expected, &Mode::LogDensity));
        }
    }
    let source = "offset(x) = x+0.25\nstep(a,x,p) = a+x+p(0.0)\npoint = external(reals)\ninputs = point\noutputs = reduce(step,[point,2.0,3.0],offset)";
    let expected = source.replace("a+x+p(0.0)", "a+x+0.25");
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(&expected, &Mode::LogDensity)
    );

    let source = "mu = elementof(reals)\nsigma = elementof(posreals)\n\
                  offset(x) = x+0.25\nM = Normal(mu,sigma)\n\
                  step(a,x,p) = a+logdensityof(p[1],x)+p[2](0.0)\n\
                  inputs = (mu,sigma)\noutputs = reduce(step,[1.0,2.0,3.0],(M,offset))";
    let expected = source
        .replace(
            "logdensityof(p[1],x)+p[2](0.0)",
            "logdensityof(Normal(p[1],p[2]),x)+0.25",
        )
        .replace(",(M,offset))", ",(mu,sigma))");
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(&expected, &Mode::LogDensity)
    );

    let mut model = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    flatppl_determinizer::determinize(&model).unwrap();

    let mixed = source
        .replace("+p[2](0.0)", "+p[2]")
        .replace(",(M,offset))", ",(M,mu))");
    let expected = source
        .replace(
            "logdensityof(p[1],x)+p[2](0.0)",
            "logdensityof(Normal(p[1],p[2]),x)+p[1]",
        )
        .replace(",(M,offset))", ",(mu,sigma))");
    assert_eq!(
        emit(&mixed, &Mode::LogDensity),
        emit(&expected, &Mode::LogDensity)
    );
}

#[test]
fn keyword_additional_arguments_of_a_sequence_callback_refuse() {
    // Keyword binding is not lowered; the emitter refuses rather than reading the
    // keyword argument as a positional one.
    let source = "mu = external(reals)\nstep(a, x, w) = w * a + x\ninputs = mu\n\
                  outputs = scan(step, 0.0, [1.0, 2.0, mu], w = [2.0])[3]";
    let mut model = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut model).is_empty());
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let options = EmitOptions::default();
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .unwrap();
    let error = flatppl_stablehlo::emit(&model, Mode::LogDensity, &options).unwrap_err();
    assert!(
        error.to_string().contains("keyword additional arguments"),
        "{error}"
    );
}

#[test]
fn scan_holds_a_constant_input_at_every_step() {
    // `scan(step, 0, [1, 2, 3], 2)` equals the scan with `[2]` repeated per step,
    // and the record form binds its field.
    let source = "mu = external(reals)\nstep(a, x, w) = w * a + x\ninputs = mu\n\
                  outputs = scan(step, 0.0, [1.0, 2.0, mu], 2.0)[3]";
    let actual = emit(source, &Mode::LogDensity);
    assert!(actual.contains("stablehlo.while"), "{actual}");
    let record = "mu = external(reals)\nstep(a, x, p) = p.w * a + x\ninputs = mu\n\
                  outputs = scan(step, 0.0, [1.0, 2.0, mu], record(w = 2.0))[3]";
    assert!(emit(record, &Mode::LogDensity).contains("stablehlo.while"));
}

#[test]
fn a_tuple_parameter_can_carry_measure_objects() {
    // At x=0 this is three independent Normal log densities with means 0,1,2:
    // -3*log(2*pi)/2 - 5/2, with derivative 3 in x.
    let source = "x = external(reals)\ninputs = x\n\
                  M = logweighted((x,p) -> logdensityof(p[1],x) + logdensityof(p[2],x), \
                      Normal(0,1), (Normal(1,1), Normal(2,1)))\noutputs = logdensityof(M,x)";
    let expected = "x = external(reals)\ninputs = x\n\
                    outputs = logdensityof(Normal(1,1),x) + logdensityof(Normal(2,1),x) + logdensityof(Normal(0,1),x)";
    assert_eq!(
        emit(source, &Mode::LogDensity),
        emit(expected, &Mode::LogDensity)
    );
    let named = source.replace(
        "M = logweighted((x,p) -> logdensityof(p[1],x) + logdensityof(p[2],x),",
        "weight(x,p) = logdensityof(p[1],x) + logdensityof(p[2],x)\nM = logweighted(weight,",
    );
    assert_eq!(
        emit(&named, &Mode::LogDensity),
        emit(expected, &Mode::LogDensity)
    );

    let restricted = "a ~ Normal(0,1)\nb ~ Normal(a,1)\n\
                      K,marginal = disintegrate([\"b\"],lawof(record(a=a,b=b)))\n\
                      nu = logweighted((z,p) -> logdensityof(p[1](z),p[2]),marginal,(K,record(b=0.5)))\n\
                      point = external(cartprod(a=reals))\ninputs = point\noutputs = logdensityof(nu,point)";
    // This is logNormal(0,1)(point.a) + logNormal(point.a,1)(.5).
    emit(restricted, &Mode::LogDensity);
}

#[test]
fn measure_parameters_are_applied_before_nested_normalization() {
    // The inner measure has density exp(a*x)/Z(a), Z(a)=expm1(a)/a.
    // The outer normalizer is exactly 2 because the inner density integrates to 1.
    let source = "a = external(reals)\nx = external(reals)\nreference = Lebesgue(interval(0.0,1.0))\n\
                  inner = normalize(logweighted((t,p) -> p*t,reference,a))\n\
                  outer = normalize(weighted((t,p) -> 1.0+exp(logdensityof(p,t)),reference,inner))\n\
                  inputs = (a,x)\noutputs = logdensityof(outer,x)";
    emit_with_integrals(source, &Mode::LogDensity, true);
}

#[test]
fn callback_recovery_does_not_assume_constant_weight_or_probability_mass() {
    for body in [
        "normalize(weighted(s, Lebesgue(reals)))",
        "normalize(weighted(x -> x*x + 1, Normal(s, 1)))",
        "normalize(weighted(s, Poisson(2)))",
    ] {
        let source = format!(
            "s = external(reals)\nrow = s -> {body}\ninputs = s\noutputs = rand(row(s), rnginit(19))"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        flatppl_infer::infer(&mut module);
        assert!(flatppl_determinizer::determinize(&module).is_err());
    }
}

#[test]
fn applied_vector_prior_retains_its_static_shape_and_mass_guard() {
    for shape in ["3", "[2, 3]"] {
        let source = format!(
            "s = external(reals)\nshift ~ Normal(0, 1)\n\
             b ~ iid(normalize(weighted(ifelse(shift > 0 && isfinite(shift), 1, 0), \
                 locscale(StudentT(4), 0, ifelse(shift > 0 && isfinite(shift), shift, 1)))), {shape})\n\
             K = kernelof(b, shift = shift)\n\
             inputs = s\noutputs = rand(K(s), rnginit(19))"
        );
        let expected = format!(
            "s = external(reals)\ninputs = s\n\
             outputs = rand(iid(normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), \
                 locscale(StudentT(4), 0, ifelse(s > 0 && isfinite(s), s, 1)))), {shape}), rnginit(19))"
        );
        assert_eq!(emit(&source, &Mode::Sample), emit(&expected, &Mode::Sample));
    }
}
