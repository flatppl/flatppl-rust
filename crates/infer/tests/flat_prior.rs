use flatppl_core::{Mass, Module, Type, ValueSet};
use flatppl_infer::{Level, ModuleBundle, Severity};

fn analyze(source: &str) -> (Module, Vec<flatppl_infer::Diagnostic>) {
    let mut module = flatppl_syntax::parse(&format!(
        "reg = standard_module(\"regression\", \"0.1\")\n{source}"
    ))
    .unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    (module, diagnostics)
}

fn infer(source: &str) -> Module {
    let (module, diagnostics) = analyze(source);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    module
}

fn mass(module: &Module, name: &str) -> Mass {
    let (_, binding) = module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)
        .unwrap();
    match module.type_of(binding.rhs).unwrap() {
        Type::Measure { mass, .. } | Type::Kernel { mass, .. } => *mass,
        other => panic!("{name}: expected a measure or kernel, got {other:?}"),
    }
}

#[test]
fn flat_compositions_admit_coordinates_without_normalizing_the_prior() {
    let module = infer(
        "b ~ iid(reg.Flat(), 2)\n\
         sigma ~ Exponential(1)\n\
         prior = lawof(record(b = b, sigma = sigma))\n\
         copy ~ prior\n\
         direct = lawof(b)\n\
         source = lawof(reg.Flat())\n\
         chain ~ jointchain(reg.Flat(), a -> Normal(a, 1))\n\
         chain_law = lawof(chain)\n\
         offset = elementof(reals)\n\
         raw ~ reg.Flat()\n\
         shifted = kernelof(raw + offset, offset = offset)\n\
         shifted_draw ~ shifted(0)\n\
         shifted_law = lawof(shifted_draw)\n\
         bounded ~ reg.Flat(support = interval(-1, 2))\n\
         bounded_law = lawof(bounded)\n\
         bounded_copy ~ bounded_law\n\
         dependent ~ reg.Flat(interval(0, sigma))\n\
         dependent_law = kernelof(dependent, sigma = sigma)\n\
         dotted ~ reg.Flat.(posreals)\n\
         dotted_law = lawof(dotted)\n\
         structured ~ reg.Flat(cartprod(a = unitinterval, b = ordered(3)))\n\
         response ~ Normal(structured.a + sum(structured.b), 1)",
    );
    assert_eq!(mass(&module, "source"), Mass::LocallyFinite);
    assert_eq!(mass(&module, "direct"), Mass::LocallyFinite);
    assert_eq!(mass(&module, "prior"), Mass::Unknown);
    assert_ne!(mass(&module, "chain_law"), Mass::Normalized);
    assert_eq!(mass(&module, "shifted_law"), Mass::Unknown);
    assert_eq!(mass(&module, "bounded_law"), Mass::Finite);
    assert_eq!(mass(&module, "dependent_law"), Mass::Unknown);
    assert_eq!(mass(&module, "dotted_law"), Mass::LocallyFinite);
    let (_, dotted) = module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == "dotted")
        .unwrap();
    assert_eq!(module.valueset_of(dotted.rhs), Some(&ValueSet::PosReals));
}

#[test]
fn reification_preserves_improper_ancestors_and_stops_at_kernel_inputs() {
    let module = infer(
        "b ~ reg.Flat()\n\
         y ~ Normal(b, 1)\n\
         conditional = kernelof(y, b = b)\n\
         conditional_draw ~ conditional(0)\n\
         predictive = lawof(y)\n\
         implicit_predictive = lawof(Normal(b, 1))\n\
         projected = lawof(record(b = b, y = y).y)\n\
         collapsed = lawof(0 * b)\n\
         shared = lawof(record(left = b, right = b))\n\
         product = joint(Normal(b, 1), Normal(0, 1))\n\
         conditional_law = functionof(lawof(y), b = b)\n\
         conditional_product = functionof(product, b = b)\n\
         from_law ~ lawof(y)\n\
         conditional_copy = kernelof(from_law, b = b)",
    );
    for name in [
        "conditional",
        "conditional_law",
        "conditional_product",
        "conditional_copy",
    ] {
        assert_eq!(mass(&module, name), Mass::Normalized, "{name}");
    }
    for name in [
        "predictive",
        "implicit_predictive",
        "projected",
        "collapsed",
        "shared",
        "product",
    ] {
        assert_eq!(mass(&module, name), Mass::Unknown, "{name}");
    }
}

#[test]
fn nested_kernel_calls_retain_enclosing_input_substitutions() {
    let module = infer(
        "b ~ reg.Flat()\n\
         c ~ reg.Flat()\n\
         y ~ Normal(b + c, 1)\n\
         inner = functionof(lawof(y), c = c)\n\
         outer = functionof(inner(0), b = b)\n\
         result = outer(0)",
    );
    assert_eq!(mass(&module, "inner"), Mass::Unknown);
    assert_eq!(mass(&module, "outer"), Mass::Normalized);
    assert_eq!(mass(&module, "result"), Mass::Normalized);
}

#[test]
fn admitted_improper_prior_still_cannot_be_sampled() {
    for support in ["", "interval(0, 1)"] {
        let (_, diagnostics) = analyze(&format!(
            "b ~ reg.Flat({support})\n\
             prior = lawof(record(b = b))\n\
             sample = rand(prior, rnginit(7))"
        ));
        assert!(diagnostics.iter().any(|d| d.severity == Severity::Error));
    }
}

#[test]
fn flat_composition_does_not_admit_an_arbitrary_unnormalized_factor() {
    let (_, diagnostics) = analyze(
        "prior = joint(reg.Flat(), weighted(2, Normal(0, 1)))\n\
         x ~ prior",
    );
    assert!(diagnostics.iter().any(|d| d.severity == Severity::Error));
}

#[test]
fn function_arguments_preserve_source_admission_without_using_the_mass_class() {
    let module = infer(
        "identity(M) = M\n\
         repeated(M) = iid(identity(M), 2)\n\
         x ~ identity(reg.Flat())\n\
         y ~ repeated(reg.Flat())\n\
         prior = lawof(record(x = x, y = y))",
    );
    assert_eq!(mass(&module, "prior"), Mass::Unknown);

    let (_, diagnostics) = analyze(
        "identity(M) = M\n\
         repeated(M) = iid(identity(M), 2)\n\
         x ~ repeated(reg.Flat())\n\
         y ~ repeated(Lebesgue(reals))",
    );
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].message.contains("source is not admitted"));
}

#[test]
fn imported_flat_sources_and_reified_priors_retain_admission() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "model.flatppl",
        std::sync::Arc::new(
            flatppl_syntax::parse(
                "reg = standard_module(\"regression\", \"0.1\")\n\
                 flat = reg.Flat()\n\
                 b ~ flat\n\
                 prior = lawof(record(b = b))\n\
                 factory(dummy) = prior\n\
                 unrestricted = Lebesgue(reals)",
            )
            .unwrap(),
        ),
    );
    bundle.insert(
        "bridge.flatppl",
        std::sync::Arc::new(
            flatppl_syntax::parse(
                "m = load_module(\"model.flatppl\")\n\
                 flat = m.flat\n\
                 prior = m.prior\n\
                 unrestricted = m.unrestricted",
            )
            .unwrap(),
        ),
    );
    let mut module = flatppl_syntax::parse(
        "m = load_module(\"bridge.flatppl\")\n\
         model = load_module(\"model.flatppl\")\n\
         x ~ m.flat\n\
         y ~ m.prior\n\
         from_factory = model.factory(0)\n\
         z ~ from_factory\n\
         prior = lawof(record(x = x, y = y))",
    )
    .unwrap();
    let diagnostics = flatppl_infer::infer_module(&mut module, &bundle, Level::Shape);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    assert_eq!(mass(&module, "prior"), Mass::Unknown);
    assert_eq!(mass(&module, "from_factory"), Mass::Unknown);

    let mut module = flatppl_syntax::parse(
        "m = load_module(\"bridge.flatppl\")\n\
         x ~ m.flat\n\
         y ~ m.unrestricted",
    )
    .unwrap();
    let diagnostics = flatppl_infer::infer_module(&mut module, &bundle, Level::Shape);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].message.contains("source is not admitted"));
}

#[test]
fn imported_measure_arguments_refuse_until_actual_substitution_is_supported() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "model.flatppl",
        std::sync::Arc::new(
            flatppl_syntax::parse(
                "reg = standard_module(\"regression\", \"0.1\")\n\
                 source = reg.Flat()\n\
                 factory = functionof(source, M = source)",
            )
            .unwrap(),
        ),
    );
    for call in [
        "m.factory(m.source)",
        "m.factory(Normal(2, 3))",
        "m.factory(Lebesgue(reals))",
        "m.factory(weighted(2, Normal(0, 1)))",
        "m.factory(M = Normal(2, 3))",
        "m.factory(record(M = Normal(2, 3)))",
        "factory(Normal(2, 3))",
    ] {
        let mut module = flatppl_syntax::parse(&format!(
            "m = load_module(\"model.flatppl\")\nfactory = m.factory\nx ~ {call}"
        ))
        .unwrap();
        let diagnostics = flatppl_infer::infer_module(&mut module, &bundle, Level::Shape);
        assert!(
            diagnostics.iter().any(|d| d.severity == Severity::Error
                && d.message.contains("actual measure substitution")),
            "{call}: {diagnostics:?}"
        );
    }
}
