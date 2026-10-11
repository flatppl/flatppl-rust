use flatppl_host::{Constant, Context};
use serde_json::json;

#[test]
fn literal_bindings_read_back_with_markers_and_references() -> Result<(), Box<dyn std::error::Error>>
{
    let module = Context::default().parse(
        "mu = elementof(reals)\npoint = record(a = -1.5, b = fixed(2.0), c = -inf, d = mu, e = [1, 2], f = array([1.0, 2.0, 3.0, 4.0], [2, 2], [1, 2]), g = (true, 2))\nname = \"x\"",
        None,
        None,
    )?;
    assert_eq!(
        serde_json::to_value(module.constant("point")?)?,
        json!({"kind": "record", "value": [
            ["a", {"kind": "real", "value": -1.5}],
            ["b", {"kind": "fixed", "value": {"kind": "real", "value": 2.0}}],
            ["c", {"kind": "real", "value": "-inf"}],
            ["d", {"kind": "reference", "value": "mu"}],
            ["e", {"kind": "array", "value": {"shape": [2], "dtype": "integer", "data": [1, 2]}}],
            ["f", {"kind": "array", "value": {"shape": [2, 2], "dtype": "real", "data": [1.0, 2.0, 3.0, 4.0]}}],
            ["g", {"kind": "tuple", "value": [{"kind": "bool", "value": true}, {"kind": "integer", "value": 2}]}]
        ]})
    );
    assert_eq!(module.constant("name")?, Constant::String("x".into()));
    Ok(())
}

#[test]
fn expressions_and_absent_names_do_not_read_back() -> Result<(), Box<dyn std::error::Error>> {
    let module = Context::default().parse(
        "mu = elementof(reals)\ny = mu + 1.0\nz = array([1.0, 2.0], [2, 1], [2, 1])",
        None,
        None,
    )?;
    for name in ["mu", "y", "z", "absent"] {
        let error = module.constant(name).err().ok_or(name)?;
        assert_eq!(error.stage, "constant");
    }
    assert_eq!(
        module
            .constant("y")
            .err()
            .and_then(|e| e.position)
            .map(|p| p.line),
        Some(2)
    );
    Ok(())
}

#[test]
fn packed_arrays_round_trip_and_bind_externals() -> Result<(), Box<dyn std::error::Error>> {
    let packed = json!({"kind": "array", "value": {"shape": [2, 2], "dtype": "real", "data": [1.0, 2.0, "inf", 4.0]}});
    let value: Constant = serde_json::from_value(packed.clone())?;
    assert_eq!(serde_json::to_value(&value)?, packed);
    let mut context = Context::default();
    let module = context.parse(
        "x = external(cartpow(reals, [2, 2]))\nflags = external(cartpow(booleans, 3))\noutputs = sum(x)",
        None,
        None,
    )?;
    let constants: Vec<(String, Constant)> = serde_json::from_value(json!([
        ["x", {"kind": "array", "value": {"shape": [2, 2], "dtype": "integer", "data": [1, 2, 3, 4]}}],
        ["flags", {"kind": "array", "value": {"shape": [3], "dtype": "bool", "data": [true, false, true]}}]
    ]))?;
    let bound = context.set(&module, &constants)?;
    assert_eq!(
        bound.constant("x")?,
        Constant::Array {
            shape: vec![2, 2],
            data: [1.0, 2.0, 3.0, 4.0].map(Constant::Real).to_vec()
        }
    );
    // A marker or a reference is not a value to bind.
    for marker in [
        Constant::Fixed(Box::new(Constant::Real(1.0))),
        Constant::Reference("y".into()),
    ] {
        let module = context.parse("x = external(reals)", None, None)?;
        assert!(context.set(&module, &[("x".into(), marker)]).is_err());
    }
    Ok(())
}
