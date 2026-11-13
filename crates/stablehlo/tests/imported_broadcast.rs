use std::sync::Arc;

use flatppl_infer::ModuleBundle;

fn emit(source: &str, bundle: &ModuleBundle) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let (_, bundle) = flatppl_infer::infer_module_with_inputs(
        &mut model,
        bundle.clone(),
        flatppl_infer::Level::Shape,
    );
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &bundle,
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap()
}

#[test]
fn imported_function_broadcasts_at_root_and_inside_expression() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helpers.flatppl",
        Arc::new(flatppl_syntax::parse("f(x) = x + 1.0").unwrap()),
    );
    for output in ["h.f.(xs)", "sum(h.f.(xs))"] {
        let prefix = "xs = elementof(cartpow(reals, 3))\ninputs = xs\n";
        let actual = format!("h = load_module(\"helpers.flatppl\")\n{prefix}outputs = {output}");
        let expected = format!(
            "f(x) = x + 1.0\n{prefix}outputs = {}",
            output.replace("h.f", "f"),
        );
        assert_eq!(
            emit(&actual, &bundle),
            emit(&expected, &ModuleBundle::new())
        );
    }
}

#[test]
fn imported_nested_scan_preserves_record_parameters() {
    let fixture = include_str!("fixtures/callable_newton_scan.flatppl");
    let stiff = fixture.split_once("rate = elementof(reals)").unwrap().0;
    // The scalar after the multi-leaf record must retain its tuple position.
    let grid = "advance(state,t,args)=record(y=args[2](args[1],state.t,state.y,t-state.t,args[3]) .+ args[4],t=t)\n\
        solve(step,f,initial,times,p,tail)=scan(advance,record(y=initial,t=times[1]),times,(f,step,p,tail)).y";
    let facade = "stiff=load_module(\"stiff.flatppl\")\ngrid=load_module(\"grid.flatppl\")\n\
        sdirk2(f,initial,times,p,tail)=grid.solve(stiff.sdirk2_step,f,initial,times,p,tail)";
    let mut bundle = ModuleBundle::new();
    for (name, source) in [
        ("stiff.flatppl", stiff),
        ("grid.flatppl", grid),
        ("ode.flatppl", facade),
    ] {
        bundle.insert(name, Arc::new(flatppl_syntax::parse(source).unwrap()));
    }
    let ir = emit(
        "ode=load_module(\"ode.flatppl\")\nx=elementof(cartpow(reals,2))\n\
         rhs(t,y,p)=[p.input_rate+p.rate*y[1]/(p.geometry.scale*p.geometry.factor)]\n\
         p=record(rate=x[1],geometry=record(scale=1.0,factor=1.0),input_rate=0.0)\n\
         inputs=x\noutputs=rowstack(ode.sdirk2(rhs,[1.0],[0.0,0.1,0.2],p,x[2]))[:,1]\n",
        &bundle,
    );
    assert!(ir.contains("-> tensor<3xf32>"), "{ir}");
}

#[test]
fn pyhf_helper_splats_an_opaque_record_by_name() {
    let prefix = "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\n\
                  pars = elementof(cartprod(alpha = reals, hi = posreals, lo = posreals))\n\
                  inputs = pars\noutputs = ";
    let actual = format!("{prefix}pyhf.normsys_factor(pars)");
    let expected = format!("{prefix}pyhf.normsys_factor(pars.lo, pars.hi, pars.alpha)");
    assert_eq!(
        emit(&actual, &ModuleBundle::new()),
        emit(&expected, &ModuleBundle::new())
    );
}

#[test]
fn pyhf_helpers_keep_dots_only_for_bin_broadcast_products() {
    let prefix = "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\n\
                  theta = elementof(cartpow(reals, 2))\n\
                  inputs = theta\n";
    // Per-row alphas broadcast over three bins; the fixed template product folds.
    let histosys = emit(
        &format!(
            "{prefix}a = addaxes([theta[1], theta[2]], 0, 1)\n\
             outputs = sum(sum(broadcast(pyhf.histosys_shift, \
             rowstack([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]), \
             rowstack([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]), \
             rowstack([[3.0, 4.0, 5.0], [2.5, 3.5, 4.5]]), a)))"
        ),
        &ModuleBundle::new(),
    );
    assert_eq!(
        histosys.matches("stablehlo.dot_general").count(),
        2,
        "{histosys}"
    );
    // One alpha per lane leaves no broadcast product.
    let normsys = emit(
        &format!(
            "{prefix}outputs = sum(broadcast(pyhf.normsys_factor, \
             [0.9, 0.8], [1.1, 1.3], [theta[1], theta[2]]))"
        ),
        &ModuleBundle::new(),
    );
    assert!(!normsys.contains("stablehlo.dot_general"), "{normsys}");
}
