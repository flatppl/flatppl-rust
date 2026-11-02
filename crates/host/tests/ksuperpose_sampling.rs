use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn mixture_draws_keep_their_cells_under_iid_and_host_batching()
-> Result<(), Box<dyn std::error::Error>> {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        ..Default::default()
    };
    for (law, cell) in [
        (
            "normalize(ksuperpose(Normal, w, mu = mu, sigma = 1.0))",
            vec![],
        ),
        (
            "iid(normalize(ksuperpose(Normal, w, mu = mu, sigma = 1.0)), [2, 3])",
            vec![2, 3],
        ),
        (
            "normalize(ksuperpose(MvNormal, w, mu = [[0.0, 0.0], [3.0, 3.0]], cov = [eye(2)]))",
            vec![2],
        ),
    ] {
        let source = format!(
            "w = external(cartpow(nonnegreals, 2))\nmu = external(cartpow(reals, 2))\n\
             s = rnginit(0)\ninputs = (s, w, mu)\noutputs = rand({law}, s)"
        );
        let query = Context::default().parse(&source, None, None)?;
        for prefix in [vec![], vec![4]] {
            let export = if prefix.is_empty() {
                query.compile(&options)?
            } else {
                query.compile_batched(
                    &options,
                    &BatchSpec {
                        shape: prefix.clone(),
                        input_axes: vec![vec![Some(0)], vec![Some(0)], vec![Some(0)]],
                    },
                )?
            };
            let schema = serde_json::to_value(export)?;
            assert_eq!(
                schema["output"]["items"][0]["shape"],
                json!([prefix.clone(), cell.clone()].concat()),
                "{law}"
            );
            assert_eq!(
                schema["output"]["items"][1]["shape"],
                json!([prefix, vec![2]].concat()),
                "{law}: successor RNG"
            );
        }
    }
    Ok(())
}
