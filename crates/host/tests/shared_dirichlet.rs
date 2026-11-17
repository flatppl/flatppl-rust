use flatppl_host::{Context, EmitOptions};

#[test]
fn shared_dirichlet_families_compile_in_both_field_orders() {
    for (first, second, domain) in [
        (
            "Categorical(p)",
            "Categorical0(p)",
            "cartprod(u=integers,v=integers)",
        ),
        (
            "Multinomial(2,p)",
            "Multinomial(3,p)",
            "cartprod(u=cartpow(integers,3),v=cartpow(integers,3))",
        ),
        (
            "Categorical(p)",
            "Multinomial(3,p)",
            "cartprod(u=integers,v=cartpow(integers,3))",
        ),
    ] {
        for fields in ["(u=u,v=v)", "(v=v,u=u)"] {
            let mut context = Context::default();
            let source = format!(
                "a=elementof(cartpow(posreals,3))\np~Dirichlet(a)\nu~{first}\nv~{second}\ny=external({domain})\ninputs=(a,y)\noutputs=logdensityof(lawof({fields}),y)"
            );
            let module = context.parse(&source, None, None).unwrap();
            module
                .compile(&EmitOptions::default())
                .unwrap_or_else(|error| panic!("{source}: {error}"));
        }
    }
}

#[test]
fn singleton_partial_and_carried_simplex_laws_compile() {
    for output in [
        "logdensityof(lawof(u),1)",
        "logdensityof(lawof(v),[1,1,1])",
        "logdensityof(lawof((u=u,v=v,z=z)),(u=1,v=[1,1,1],z=1.))",
        "logdensityof(lawof((p=p,u=u,v=v)),(p=[.2,.3,.5],u=1,v=[1,1,1]))",
        "logdensityof(lawof((u=u,v=v)),(u=1,v=[1,1,1]))+logdensityof(lawof((v=v,u=u)),(v=[0,0,3],u=3))",
    ] {
        let mut context = Context::default();
        let source = format!(
            "a=elementof(cartpow(posreals,3))\np~Dirichlet(a)\nu~Categorical(p)\nv~Multinomial(3,p)\nz~Normal(0.,1.)\ninputs=a\noutputs={output}"
        );
        let module = context.parse(&source, None, None).unwrap();
        module
            .compile(&EmitOptions::default())
            .unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}
