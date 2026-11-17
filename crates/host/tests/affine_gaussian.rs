use flatppl_host::{Context, EmitOptions};

#[test]
fn gaussian_chain_marginals_compose_after_prior_queries() {
    let source = "z~MvNormal([1.,-2.],rowstack([[2.,1.],[1.,3.]]))\ny~MvNormal(rowstack([[1.,2.]])*z+[.5],rowstack([[4.]]))\nx~MvNormal(rowstack([[2.],[-1.]])*y+[1.,3.],rowstack([[3.,1.],[1.,2.]]))\noutputs=(first=logdensityof(lawof(y),[1.5]),last=logdensityof(lawof(x),[4.,2.]))";
    let mut context = Context::default();
    let module = context.parse(source, None, None).unwrap();
    module.compile(&EmitOptions::default()).unwrap();
}

#[test]
fn gaussian_reverse_vector_disintegration_keeps_the_conditional() {
    for (selector, point) in [("1", "[0.]"), ("[1]", "[[0.]]")] {
        let source = format!(
            "z~MvNormal([0.],rowstack([[1.]]))\ny~MvNormal(rowstack([[1.]])*z,rowstack([[1.]]))\nposterior,predictive=disintegrate({selector},lawof([z,y]))\noutputs=logdensityof(posterior(complement=[[2.]]),{point})"
        );
        let mut context = Context::default();
        let module = context.parse(&source, None, None).unwrap();
        module.compile(&EmitOptions::default()).unwrap();
    }
}

#[test]
fn prior_query_pins_do_not_change_gaussian_conditionals() {
    let mut context = Context::default();
    let source = "z~MvNormal([1.,-2.],rowstack([[2.,1.],[1.,3.]]))\ny~MvNormal(rowstack([[1.,2.]])*z+[.5],rowstack([[4.]]))\nposterior,predictive=disintegrate(\"z\",lawof((z=z,y=y)))\noutputs=(prior=logdensityof(lawof(z),[0.,0.]),observed=logdensityof(lawof(y),[1.5]),posterior=logdensityof(posterior(y=[1.5]),[.5,-.25]))";
    let module = context.parse(source, None, None).unwrap();
    module.compile(&EmitOptions::default()).unwrap();
}

#[test]
fn gaussian_conditionals_refuse_uncarried_random_parameters_after_other_queries() {
    for (upstream, mean, matrix, earlier) in [
        (
            "w~MvNormal([0.,0.],rowstack([[1.,0.],[0.,1.]]))",
            "w",
            "rowstack([[1.,2.]])",
            "logdensityof(lawof(w),[0.,0.])",
        ),
        (
            "a~Normal(1.,1.)",
            "[1.,-2.]",
            "rowstack([[a,2.]])",
            "logdensityof(lawof(a),1.)",
        ),
    ] {
        let source = format!(
            "{upstream}\nz~MvNormal({mean},rowstack([[2.,1.],[1.,3.]]))\nA={matrix}\ny~MvNormal(A*z+[.5],rowstack([[4.]]))\nposterior,predictive=disintegrate(\"z\",lawof((z=z,y=y)))\noutputs=(earlier={earlier},posterior=logdensityof(posterior(y=[1.5]),[.5,-.25]))"
        );
        let mut context = Context::default();
        let module = context.parse(&source, None, None).unwrap();
        assert!(
            module.compile(&EmitOptions::default()).is_err(),
            "extra stochastic ancestors must not become a two-node Gaussian conditional: {source}"
        );
    }
}

#[test]
fn rectangular_affine_gaussian_marginals_compile() {
    for rows in [1, 3] {
        for mean in ["A*z+b", "b .+ A*z", "A*z"] {
            let source = format!(
                "mu=elementof(cartpow(reals,2))\nC=elementof(cartpow(reals,[2,2]))\nA=elementof(cartpow(reals,[{rows},2]))\nb=elementof(cartpow(reals,{rows}))\nR=elementof(cartpow(reals,[{rows},{rows}]))\nz~MvNormal(mu,C)\nmean={mean}\ny~MvNormal(mean,R)\npoint=external(cartpow(reals,{rows}))\ninputs=(mu,C,A,b,R,point)\noutputs=logdensityof(lawof(y),point)"
            );
            let mut context = Context::default();
            let module = context.parse(&source, None, None).unwrap();
            module
                .compile(&EmitOptions::default())
                .unwrap_or_else(|error| panic!("{source}: {error}"));
        }
    }
}

#[test]
fn latent_dependent_matrix_or_offset_stays_unintegrated() {
    for (matrix, offset) in [
        ("rowstack([[z[1],0.]])", "[.5]"),
        ("rowstack([[1.,2.]])", "[z[1]]"),
    ] {
        let source = format!(
            "z~MvNormal([1.,-2.],rowstack([[2.,1.],[1.,3.]]))\nA={matrix}\nb={offset}\nmean=A*z+b\ny~MvNormal(mean,rowstack([[4.]]))\noutputs=logdensityof(lawof(y),[1.5])"
        );
        let mut context = Context::default();
        let module = context.parse(&source, None, None).unwrap();
        assert!(
            module.compile(&EmitOptions::default()).is_err(),
            "nonlinear latent dependence must not become an affine Gaussian: {source}"
        );
    }
}

#[test]
fn gaussian_reverse_disintegration_retains_selector_shapes_and_aliases() {
    for (selector, point) in [
        ("\"latent\"", "[.5,-.25]"),
        ("[\"latent\"]", "(latent=[.5,-.25])"),
    ] {
        let source = format!(
            "mu=elementof(cartpow(reals,2))\nC=elementof(cartpow(reals,[2,2]))\nA=elementof(cartpow(reals,[1,2]))\nb=elementof(cartpow(reals,1))\nR=elementof(cartpow(reals,[1,1]))\nz~MvNormal(mu,C)\ny~MvNormal(A*z+b,R)\nz_alias=z\ny_alias=y\nposterior,predictive=disintegrate({selector},lawof((latent=z_alias,observed=y_alias)))\ngiven=external(cartpow(reals,1))\ninputs=(mu,C,A,b,R,given)\noutputs=(posterior=logdensityof(posterior(observed=given),{point}),predictive=logdensityof(predictive,(observed=given)))"
        );
        let mut context = Context::default();
        let module = context.parse(&source, None, None).unwrap();
        module
            .compile(&EmitOptions::default())
            .unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}
