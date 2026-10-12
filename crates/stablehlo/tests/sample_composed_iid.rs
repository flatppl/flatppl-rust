use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
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
    flatppl_stablehlo::emit(
        &model,
        Mode::Sample,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
    // Constant-folded 0/0 may have either NaN sign on different host CPUs.
    .replace("dense<0xFFC00000>", "dense<0x7FC00000>")
}

#[test]
fn affine_iid_uses_one_base_batch_and_its_successor_state() {
    for base in ["Normal(0, 1)", "StudentT(8)", "Bernoulli(0.3)"] {
        for shape in ["6", "[2, 3]"] {
            let prefix = "shift = external(reals)\nscale = external(reals)\n";
            let suffix = "next_value, next_state = rand(Normal(0, 1), state)\n\
                          inputs = (shift, scale)\noutputs = (value, next_value, next_state)";
            let expected = format!(
                "{prefix}z, state = rand(iid({base}, {shape}), rnginit(23))\n\
                 value = scale .* z .+ shift\n{suffix}"
            );
            for measure in [
                format!("locscale({base}, shift, scale)"),
                format!("locscale(normalize(weighted(2, {base})), shift, scale)"),
            ] {
                let source = format!(
                    "{prefix}value, state = rand(iid({measure}, {shape}), rnginit(23))\n{suffix}"
                );
                assert_eq!(emit(&source), emit(&expected), "{measure}, {shape}");
            }
        }
    }
}

#[test]
fn normalized_vector_coefficient_prior_preserves_shared_draws_and_mass_guard() {
    let prefix = "nu = external(posreals)\nshift = external(reals)\n\
                  scale = external(reals)\nw = external(reals)\n";
    let source = format!(
        "{prefix}b ~ iid(normalize(weighted(w, locscale(StudentT(nu), shift, scale))), 3)\n\
         prior = lawof(record(b = b, repeated = b))\n\
         sample, state = rand(prior, rnginit(23))\n\
         inputs = (nu, shift, scale, w)\noutputs = (sample, state)"
    );
    let expected = format!(
        "{prefix}z, state = rand(iid(StudentT(nu), 3), rnginit(23))\n\
         value = ifelse(isfinite(w) && w > 0.0, scale .* z .+ shift, divide(0.0, 0.0))\n\
         inputs = (nu, shift, scale, w)\noutputs = (record(b = value, repeated = value), state)"
    );
    assert_eq!(emit(&source), emit(&expected));
}

#[test]
fn nested_iid_transforms_keep_each_scalar_mass_guard() {
    let prefix = "shift = external(reals)\nscale = external(reals)\nw = external(reals)\n";
    let source = format!(
        "{prefix}value, state = rand(iid(locscale(normalize(logweighted(w, StudentT(8))), shift, scale), [2, 3]), rnginit(23))\n\
         inputs = (shift, scale, w)\noutputs = (value, state)"
    );
    let expected = format!(
        "{prefix}z, state = rand(iid(StudentT(8), [2, 3]), rnginit(23))\n\
         value = scale .* ifelse(isfinite(w), z, divide(0.0, 0.0)) .+ shift\n\
         inputs = (shift, scale, w)\noutputs = (value, state)"
    );
    assert_eq!(emit(&source), emit(&expected));
}

#[test]
fn conditional_iid_callback_keeps_its_shared_parent_and_successor_state() {
    let source = "shift ~ Normal(0, 1)\n\
                  row = l -> locscale(StudentT(8), l, -2)\n\
                  b ~ iid(row(shift), 3)\nchild ~ Normal(sum(b), 1)\n\
                  prior = lawof(record(shift = shift, b = b, repeated = b, child = child))\n\
                  sample, state = rand(prior, rnginit(23))\noutputs = (sample, state)";
    let expected = "shift, state1 = rand(Normal(0, 1), rnginit(23))\n\
                    z, state2 = rand(iid(StudentT(8), 3), state1)\nb = -2 .* z .+ shift\n\
                    child, state = rand(Normal(sum(b), 1), state2)\n\
                    outputs = (record(shift = shift, b = b, repeated = b, child = child), state)";
    assert_eq!(emit(source), emit(expected));
}

#[test]
fn iid_keeps_normalized_integer_and_rowwise_kernel_boundaries() {
    for measure in [
        "normalize(weighted(w, Bernoulli(0.3)))",
        "normalize(weighted(0, Normal(0, 1)))",
        "Normal.([0, 1], 1)",
    ] {
        let mut model = flatppl_syntax::parse(&format!(
            "w = external(reals)\ninputs = w\noutputs = rand(iid({measure}, 3), rnginit(23))"
        ))
        .unwrap();
        flatppl_infer::infer(&mut model);
        assert!(flatppl_determinizer::determinize(&model).is_err());
    }
}
