use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn lkj_matrix_outputs_keep_static_cells_under_iid_and_host_batching()
-> Result<(), Box<dyn std::error::Error>> {
    let options = EmitOptions {
        restrict_enzyme_compatible: false,
        ..Default::default()
    };
    for constructor in ["LKJ", "LKJCholesky"] {
        for (law, cell) in [
            (format!("{constructor}(n, eta)"), vec![3, 3]),
            (format!("iid({constructor}(n, eta), 6)"), vec![6, 3, 3]),
            (
                format!("iid({constructor}(n, eta), [2, 3])"),
                vec![2, 3, 3, 3],
            ),
        ] {
            let source = format!(
                "n = 3\neta = external(posreals)\ns = rnginit(0)\n\
                 inputs = (s, eta)\noutputs = rand({law}, s)"
            );
            let query = Context::default().parse(&source, None, None)?;
            for prefix in [vec![], vec![2], vec![2, 3]] {
                let export = if prefix.is_empty() {
                    query.compile(&options)?
                } else {
                    let axes: Vec<_> = (0..prefix.len()).map(Some).collect();
                    query.compile_batched(
                        &options,
                        &BatchSpec {
                            shape: prefix.clone(),
                            input_axes: vec![axes.clone(), axes],
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
    }
    Ok(())
}
