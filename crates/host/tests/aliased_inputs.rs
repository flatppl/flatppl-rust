//! An `inputs` entry behind a bare-name alias chain (`X2 = X`, `X = x`) is the
//! `elementof` it names (§04 aliasing, §13 signature). The argument keeps the
//! name the user listed.

use flatppl_host::{Context, EmitOptions};

fn compile(source: &str) -> flatppl_host::Export {
    let mut context = Context::default();
    let module = context.parse(source, None, None).unwrap();
    module
        .compile(&EmitOptions::default())
        .unwrap_or_else(|e| panic!("{e:?}"))
}

#[test]
fn an_aliased_input_compiles_like_the_direct_one() {
    let score = |mu: &str| format!("outputs = logdensityof(Normal(mu = {mu}, sigma = 1.0), 0.5)");
    let direct = compile(&format!(
        "x = elementof(reals)\ninputs = (x)\n{}",
        score("x")
    ));
    for mu in ["X2", "x"] {
        let aliased = compile(&format!(
            "x = elementof(reals)\nX = x\nX2 = X\ninputs = (X2)\n{}",
            score(mu)
        ));
        assert_eq!(aliased.stablehlo, direct.stablehlo, "mu = {mu}");
        let names: Vec<_> = aliased.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["X2"]);
    }
}

#[test]
fn an_alias_and_its_target_are_one_input() {
    let mut context = Context::default();
    let module = context
        .parse(
            "x = elementof(reals)\nX = x\ninputs = (X, x)\n\
             outputs = logdensityof(Normal(mu = x, sigma = 1.0), 0.5)",
            None,
            None,
        )
        .unwrap();
    let Err(err) = module.compile(&EmitOptions::default()) else {
        panic!("one object, two argument positions");
    };
    assert!(format!("{err:?}").contains("distinct"), "{err:?}");
}
