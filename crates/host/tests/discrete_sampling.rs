use flatppl_host::{BatchSpec, Context, EmitOptions};
use flatppl_stablehlo::Dtype;
use serde_json::json;

/// §08 gives every discrete distribution the domain `integers`.
const LAWS: &[(&str, &str, usize)] = &[
    ("p = elementof(unitinterval)", "Bernoulli(p = p)", 1),
    (
        "n = 10\np = elementof(unitinterval)",
        "Binomial(n = n, p = p)",
        1,
    ),
    (
        "p = elementof(cartpow(unitinterval, 3))",
        "Categorical(p = p)",
        1,
    ),
    (
        "p = elementof(cartpow(unitinterval, 3))",
        "Categorical0(p = p)",
        1,
    ),
    ("p = elementof(unitinterval)", "Geometric(p = p)", 1),
    (
        "alpha = elementof(posreals)\nbeta = elementof(posreals)",
        "NegativeBinomial(alpha = alpha, beta = beta)",
        2,
    ),
    (
        "mu = elementof(posreals)\npsi = elementof(posreals)",
        "NegativeBinomial2(mu = mu, psi = psi)",
        2,
    ),
    ("rate = elementof(nonnegreals)", "Poisson(rate = rate)", 1),
];

#[test]
fn discrete_draws_return_integers_through_the_host_api() -> Result<(), Box<dyn std::error::Error>> {
    for &(params, law, arity) in LAWS {
        let names: Vec<&str> = params
            .lines()
            .filter(|line| line.contains("elementof"))
            .map(|line| line.split(" = ").next().unwrap())
            .collect();
        assert_eq!(names.len(), arity, "{law}");
        for (measure, cell) in [
            (law.to_string(), vec![]),
            (format!("iid({law}, 200)"), vec![200]),
        ] {
            let source = format!(
                "{params}\nx ~ {measure}\ns = rnginit([0])\n\
                 draws = rand(lawof(x), s)\ninputs = ({})\noutputs = (draws)\n",
                names.join(", ")
            );
            let query = Context::default().parse(&source, None, None)?;
            for (dtype, check_parameters) in [(Dtype::F32, false), (Dtype::F64, true)] {
                let options = EmitOptions {
                    dtype,
                    check_parameters,
                    restrict_enzyme_compatible: false,
                    ..Default::default()
                };
                let (dtype, mlir) = match dtype {
                    Dtype::F32 => ("int32", "i32"),
                    Dtype::F64 => ("int64", "i64"),
                };
                for prefix in [vec![], vec![4]] {
                    let export = if prefix.is_empty() {
                        query.compile(&options)?
                    } else {
                        query.compile_batched(
                            &options,
                            &BatchSpec {
                                shape: prefix.clone(),
                                input_axes: vec![vec![Some(0)]; arity],
                            },
                        )?
                    };
                    let shape = [prefix.clone(), cell.clone()].concat();
                    let returned = shape
                        .iter()
                        .map(|dim| format!("{dim}x"))
                        .collect::<String>();
                    assert!(
                        export
                            .stablehlo
                            .contains(&format!("-> (tensor<{returned}{mlir}>")),
                        "{measure} {prefix:?}: {}",
                        export.stablehlo
                    );
                    let schema = serde_json::to_value(&export)?;
                    let draws = &schema["output"]["items"][0];
                    assert_eq!(draws["dtype"], json!(dtype), "{measure} {prefix:?}");
                    assert_eq!(draws["shape"], json!(shape), "{measure} {prefix:?}");
                }
            }
        }
    }
    Ok(())
}
