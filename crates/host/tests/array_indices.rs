use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn index_vectors_construct_a_batched_lag_matrix() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "y = external(cartpow(reals, 5))\n\
         times = indicesof(zeros(3)) .+ 2\n\
         lagged(t) = y[t .- indicesof(zeros(2))]\n\
         X = rowstack(lagged.(times))\n\
         inputs = y\noutputs = X",
        None,
        None,
    )?;
    let options = EmitOptions::default();
    query.compile(&options)?;
    query.compile_batched(
        &options,
        &BatchSpec {
            shape: vec![2],
            input_axes: vec![vec![Some(0)]],
        },
    )?;
    Ok(())
}

#[test]
fn index_axes_preserve_array_nesting_and_table_rows() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "matrix = zeros([2, 3])\n\
         nested = [[1, 2, 3], [4, 5, 6]]\n\
         rows = table(x = [1, 2, 3, 4])\n\
         outputs = record(axes = indicesof(matrix), zero_axes = indicesof0(matrix),\n\
           outer = indicesof(nested), rows = indicesof0(rows),\n\
           cells = indicesof0.(nested))",
        None,
        None,
    )?;
    query.compile(&EmitOptions::default())?;
    Ok(())
}
