use flatppl_host::{Context, EmitOptions};

#[test]
fn shared_beta_families_compile_in_both_field_orders() {
    for (first, second, domain) in [
        (
            "Bernoulli(p)",
            "Bernoulli(p)",
            "cartprod(u=booleans,v=booleans)",
        ),
        (
            "Binomial(2,p)",
            "Binomial(3,p)",
            "cartprod(u=integers,v=integers)",
        ),
        (
            "Bernoulli(p)",
            "Binomial(3,p)",
            "cartprod(u=booleans,v=integers)",
        ),
    ] {
        for fields in ["(u=u,v=v)", "(v=v,u=u)"] {
            let mut context = Context::default();
            let source = format!(
                "a=elementof(posreals)\nb=elementof(posreals)\np~Beta(a,b)\nu~{first}\nv~{second}\ny=external({domain})\ninputs=(a,b,y)\noutputs=logdensityof(lawof({fields}),y)"
            );
            let module = context.parse(&source, None, None).unwrap();
            module
                .compile(&EmitOptions::default())
                .unwrap_or_else(|error| panic!("{source}: {error}"));
        }
    }
}

#[test]
fn beta_bernoulli_singletons_partial_blocks_and_carried_latents_compile() {
    for output in [
        "logdensityof(lawof(u),true)",
        "logdensityof(lawof((u=u,v=v,z=z)),(u=true,v=2,z=1.0))",
        "logdensityof(lawof((p=p,u=u,v=v)),(p=.4,u=true,v=2))",
        "logdensityof(lawof((u=u,v=v)),(u=true,v=2))+logdensityof(lawof((v=v,u=u)),(v=0,u=false))",
    ] {
        let mut context = Context::default();
        let source = format!(
            "a=elementof(posreals)\nb=elementof(posreals)\np~Beta(a,b)\nu~Bernoulli(p)\nv~Binomial(3,p)\nz~Normal(0.,1.)\ninputs=(a,b)\noutputs={output}"
        );
        let module = context.parse(&source, None, None).unwrap();
        module
            .compile(&EmitOptions::default())
            .unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}
