//! Direct ksuperpose uses ordinary broadcast cells and contracts their axes.
use flatppl_core::{Mass, Module, Phase, Type};
use flatppl_infer::{Severity, infer};

fn infer_src(src: &str) -> Module {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    module
}

fn binding(module: &mut Module, name: &str) -> flatppl_core::NodeId {
    let name = module.intern(name);
    module.binding(module.binding_by_name(name).unwrap()).rhs
}

#[test]
fn normalization_proofs_require_unexpanded_weights() {
    for (weights, parameters, normalized) in [
        ("[0.3, 0.7]", "mu = [-1., 2.], sigma = 1.", true),
        (
            "[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]",
            "mu = 0., sigma = 1.",
            false,
        ),
        ("[1.]", "mu = [-1., 2.], sigma = 1.", false),
        ("1.", "mu = [-1., 2.], sigma = 1.", false),
        ("1.", "mu = 0., sigma = 1.", true),
        (
            "rowstack([[0.3,0.7]])",
            "mu = rowstack([[0.,1.]]), sigma = 1.",
            true,
        ),
    ] {
        let mut module = infer_src(&format!("m = ksuperpose(Normal, {weights}, {parameters})"));
        let id = binding(&mut module, "m");
        assert!(
            matches!(module.type_of(id), Some(Type::Measure { mass, .. })
            if (*mass == Mass::Normalized) == normalized),
            "{weights}: {:?}",
            module.type_of(id)
        );
    }
}

#[test]
fn simplex_proof_survives_only_without_expansion() {
    for (size, expected) in [(1, Mass::Unknown), (2, Mass::Normalized)] {
        let mut m = infer_src(&format!(
            "p = elementof(stdsimplex({size}))\nm = ksuperpose(Normal, p, mu = [-1., 2.], sigma = 1.)"
        ));
        let id = binding(&mut m, "m");
        assert!(matches!(m.type_of(id), Some(Type::Measure { mass, .. }) if *mass == expected));
    }
    infer_src(
        "n = external(posintegers)\np = elementof(stdsimplex(n))\nm = ksuperpose(Normal, p, mu = 0., sigma = 1.)\nx ~ m",
    );
}

#[test]
fn ordinary_broadcast_inputs_keep_cell_domains_and_phase() {
    let mut module = infer_src(
        "t = elementof(reals)\nw = rowstack([[0.3], [1.2]])\nmeans = rowstack([[-1., 2.]])\nm = normalize(ksuperpose(Normal, w, means, sigma = t))\nx ~ m",
    );
    let id = binding(&mut module, "m");
    assert!(
        matches!(module.type_of(id), Some(Type::Measure { domain, mass: Mass::Normalized }) if **domain == Type::Scalar(flatppl_core::ScalarType::Real))
    );
    assert_eq!(module.phase_of(id), Some(Phase::Parametric));
}

#[test]
fn tables_follow_the_ordinary_kernel_interface() {
    for kernel in ["Normal", "(r -> Normal(r.mu, r.sigma))"] {
        infer_src(&format!(
            "p = table(mu = [-1.,2.], sigma = [1.,0.5])\nm = normalize(ksuperpose({kernel}, [0.3,1.2], p))\nx ~ m"
        ));
    }
}

#[test]
fn nested_vectors_and_matrices_are_component_cells() {
    infer_src(
        "c = eye(2)\nm = normalize(ksuperpose(MvNormal, [0.2,0.8], mu = [[0.,0.], [3.,3.]], cov = [c]))\nx ~ m",
    );
    infer_src(
        "p = table(mu = [[0.,0.], [3.,3.]], cov = [eye(2), eye(2)])\nm = normalize(ksuperpose(MvNormal, [0.2,0.8], p))\nx ~ m",
    );
}

#[test]
fn scalar_and_vector_dirac_cells_keep_their_domains() {
    for labels in ["[0., 1.5]", "[[0.,1.], [2.,3.]]"] {
        infer_src(&format!(
            "m = ksuperpose(Dirac, [0.2,0.8], value = {labels})\nx ~ m"
        ));
    }
}

#[test]
fn mixed_broadcast_parameters_bind_the_same_cells_as_named_parameters() {
    for kernel in ["Normal", "((a,b) -> Normal(a,b))"] {
        let named = if kernel == "Normal" { "sigma" } else { "b" };
        infer_src(&format!(
            "m = broadcast({kernel}, [-1.,2.], {named} = [1.,0.5])\nx ~ m"
        ));
    }
}

#[test]
fn simplex_weights_make_probability_mixtures_drawable() {
    for source in [
        "n = external(posintegers)\np = elementof(stdsimplex(n))\nm = ksuperpose(Normal, p, mu = 0, sigma = 1)\nx ~ m",
        "p = elementof(stdsimplex(3))\nq = p\nm = superpose(weighted(q[3], Normal(2, 1)), weighted(p[1], Normal(0, 1)), weighted(p[2], Normal(1, 1)))\nx ~ m",
        "p = elementof(stdsimplex(1))\nm = superpose(weighted(p[1], Dirac(2)))\nx ~ m",
    ] {
        let out = flatppl_flatpir::write(&infer_src(source));
        assert!(out.contains("(%mass %normalized)"), "{out}");
    }
}

#[test]
fn simplex_recognition_does_not_invent_unit_mass() {
    for rhs in [
        "superpose(weighted(p[1], Normal(0,1)), weighted(p[2], Normal(0,1)))",
        "superpose(weighted(p[1], Normal(0,1)), weighted(p[1], Normal(0,1)), weighted(p[3], Normal(0,1)))",
        "superpose(weighted(p[1], Normal(0,1)), weighted(p[2], Normal(0,1)), weighted(q[3], Normal(0,1)))",
        "ksuperpose(x -> weighted(2, Dirac(x)), p, x = [0,1,2])",
    ] {
        let mut module = infer_src(&format!(
            "p = elementof(stdsimplex(3))\nq = elementof(stdsimplex(3))\nm = {rhs}"
        ));
        let name = module.intern("m");
        let binding = module.binding_by_name(name).unwrap();
        assert!(
            matches!(
                module.type_of(module.binding(binding).rhs),
                Some(flatppl_core::Type::Measure {
                    mass: flatppl_core::Mass::Unknown,
                    ..
                })
            ),
            "{rhs}"
        );
    }
}

#[test]
fn weight_collections_have_real_scalar_cells() {
    for weights in [
        "[[0.5,0.5],[0.5,0.5]]",
        "table(weight = [0.5,0.5])",
        "\"weight\"",
        "[\"first\",\"second\"]",
    ] {
        let source = format!("m = ksuperpose(Normal, {weights}, mu = [0.,1.], sigma = 1.)");
        let mut module = flatppl_syntax::parse(&source).unwrap();
        assert!(
            infer(&mut module)
                .iter()
                .any(|d| d.severity == Severity::Error),
            "{weights}"
        );
        let id = binding(&mut module, "m");
        assert!(
            matches!(module.type_of(id), Some(Type::Failed(_))),
            "{weights}"
        );
    }
}

#[test]
fn component_calls_yield_measures() {
    for head in ["add", "((x,y) -> x+y)"] {
        let mut module =
            flatppl_syntax::parse(&format!("m = ksuperpose({head}, [0.3,0.7], [1.,2.], 1.)"))
                .unwrap();
        assert!(
            infer(&mut module)
                .iter()
                .any(|d| d.severity == Severity::Error)
        );
        let id = binding(&mut module, "m");
        assert!(matches!(module.type_of(id), Some(Type::Failed(_))));
    }
}
