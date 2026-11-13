use flatppl_determinizer::{determinize_with_roots, is_flatpdl};
use flatppl_infer::ModuleBundle;

#[test]
fn callback_tuple_through_nested_scans() {
    let src = r#"
        rhs(t, y, p) = p * y
        inner(a, x, p) = a + p[1](x, a, p[2])
        outer(a, x, p) = scan(inner, a, [x, x], p)[2]
        solve(f, y, p) = scan(outer, y, [1.0, 2.0], (f, p))
        x = elementof(reals)
        inputs = x
        outputs = solve(rhs, 1.0, x)
    "#;
    let mut module = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut module);
    let output = module.intern("outputs");
    let input = module.intern("inputs");
    let result = determinize_with_roots(&module, &ModuleBundle::new(), Some(&[input, output]))
        .expect("specialize callback objects");
    assert!(is_flatpdl(&result).is_ok());
}

#[test]
fn value_only_tuple_specialization_terminates_on_repeat() {
    let src = "x=elementof(cartpow(reals,2))\n\
        step(a,i,p)=a+i*p[1]+p[2]\n\
        inputs=x\noutputs=scan(step,0.0,[1.0,2.0],(x[1],x[2]))";
    let mut module = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut module);
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let lowered = determinize_with_roots(&module, &ModuleBundle::new(), Some(&roots))
        .expect("first specialization");
    let repeated = determinize_with_roots(&lowered, &ModuleBundle::new(), Some(&roots))
        .expect("repeat specialization terminates");
    assert!(is_flatpdl(&repeated).is_ok());
}
