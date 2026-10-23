use flatppl_determinizer::determinize_with_roots;
use flatppl_infer::{ModuleBundle, Severity};

fn lowered(source: &str) -> String {
    let mut m = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}\n{source}"
    );
    let root = m.intern("outputs");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[root])).unwrap();
    flatppl_syntax::print(&out)
}

#[test]
fn composition_and_returned_callables_keep_whole_record_inputs() {
    assert_eq!(
        lowered(
            "f(r)=r.x\nchoose(f)=f\ng=choose(f)\nfirst(x)=(x=x)\nh=fchain(first,g)\noutputs=h(3)"
        ),
        "outputs = 3"
    );
    assert_eq!(
        lowered("outputs=checked((value=3,condition=true))"),
        "outputs = 3"
    );
    assert_eq!(
        lowered(
            "choose(f)=f\nx=elementof(reals)\nf=functionof(x+1)\ng=choose(f)\noutputs=g((x=2))"
        ),
        "outputs = 3"
    );
}

#[test]
fn likelihood_factors_project_a_shared_point_but_ordinary_calls_are_exact() {
    let actual = lowered(
        "mu=elementof(reals)\nnu=elementof(reals)\nK1=functionof(Normal(mu,1))\nK2=functionof(Normal(nu,2))\nL=joint_likelihood(likelihoodof(K1,1),likelihoodof(K2,-1))\noutputs=logdensityof(L,(mu=0.5,nu=0.25))",
    );
    let expected = lowered("outputs=logdensityof(Normal(0.5,1),1)+logdensityof(Normal(0.25,2),-1)");
    assert_eq!(actual, expected);
    let mut m = flatppl_syntax::parse(
        "mu=elementof(reals)\nK=functionof(Normal(mu,1))\noutputs=K((mu=0.5,extra=1))",
    )
    .unwrap();
    assert!(
        flatppl_infer::infer(&mut m)
            .iter()
            .any(|d| d.severity == Severity::Error)
    );
}

#[test]
fn record_adapter_keeps_the_normal_normal_marginal_closed_form() {
    let actual = lowered(
        "z~Normal(0,2)\nK(z)=Normal(z,1)\nM=kchain(lawof(z), K)\noutputs=logdensityof(M,0.5)",
    );
    let expected = lowered("outputs=logdensityof(Normal(0,sqrt(5.0)),0.5)");
    assert_eq!(actual, expected);
}
