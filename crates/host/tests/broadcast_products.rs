use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

const SOURCE: &str = r#"
points = external(cartpow(reals, 4))
basis(t) = [t, t*t, 1.]
left(t) = rowstack([[t,1.,2.], [3.,-1.,t*t]])
right(t) = colstack([[1.,t,2.], [t*t,0.,-1.]])
inner(t) = transpose([1.,-2.,3.]) * basis(t)
outer(t) = [t,1.] * transpose([2.,t*t,-1.])
row_matrix(t) = transpose([1.,t,2.]) * right(t)
matrix_vector(t) = left(t) * basis(t)
matrix_matrix(t) = left(t) * right(t)
transposed(t) = transpose(left(t))
inputs = points
outputs = (inner.(points), outer.(points), row_matrix.(points),
           matrix_vector.(points), matrix_matrix.(points), transposed.(points))
"#;

#[test]
fn products_keep_cell_axes_under_callable_and_host_batches()
-> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(SOURCE, None, None)?;
    let options = EmitOptions::default();
    let plain = query.compile(&options)?;
    let batched = query.compile_batched(
        &options,
        &BatchSpec {
            shape: vec![2],
            input_axes: vec![vec![Some(0)]],
        },
    )?;
    for (exported, prefix) in [(plain, vec![]), (batched, vec![2])] {
        let schema = serde_json::to_value(&exported)?;
        let cells = [
            vec![4],
            vec![4, 2, 3],
            vec![4, 2],
            vec![4, 2],
            vec![4, 2, 2],
            vec![4, 3, 2],
        ];
        for (output, cell) in schema["output"]["items"]
            .as_array()
            .unwrap()
            .iter()
            .zip(cells)
        {
            let shape = [prefix.clone(), cell].concat();
            assert_eq!(output["shape"], json!(shape));
        }
    }
    Ok(())
}
