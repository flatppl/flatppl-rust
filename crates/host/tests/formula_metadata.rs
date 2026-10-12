use flatppl_host::{Constant, Context};
use serde_json::json;

fn field<'a>(value: &'a Constant, path: &[&str]) -> &'a Constant {
    path.iter().fold(value, |value, name| {
        let Constant::Record(fields) = value else {
            panic!("expected a metadata record at {name}");
        };
        &fields
            .iter()
            .find(|(key, _)| key == name)
            .unwrap_or_else(|| panic!("missing metadata field {name}"))
            .1
    })
}

#[test]
fn formula_constants_preserve_empty_sequences_and_full_width_labels()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.import_formula_json(
        &json!({
            "formula":"y ~ 1 + (1 | g)", "family":"categorical",
            "data":{"nrows":2,"columns":[
                {"name":"y","values":{"kind":"int","values":[i64::MIN,1]}},
                {"name":"g","values":{"kind":"int","values":[i64::MIN,i64::MIN]}}
            ]}
        })
        .to_string(),
        None,
        None,
        None,
    )?;
    let metadata = model.module.constant("formula_model")?;
    let names = model.module.constant("formula_parameter_names")?;
    assert_eq!(model.template.constant("formula_model")?, metadata);
    assert_eq!(field(&metadata, &["parameters"]), &names);
    assert_eq!(
        field(&metadata, &["design_columns"]),
        &Constant::Record(vec![("length".into(), Constant::Integer(0))])
    );
    let groups = field(&metadata, &["dpars", "mu1", "group_levels", "g", "levels"]);
    assert_eq!(field(groups, &["length"]), &Constant::Integer(1));
    for level in [
        field(groups, &["_1"]),
        field(&metadata, &["response_levels", "_1"]),
    ] {
        assert_eq!(field(level, &["kind"]), &Constant::String("integer".into()));
        assert_eq!(
            field(level, &["value"]),
            &Constant::String(i64::MIN.to_string())
        );
    }
    assert_eq!(
        field(&names, &["temp_Intercept_mu1", "scalar", "role"]),
        &Constant::String("sampled".into())
    );
    let info = serde_json::to_value(&model.info)?;
    assert_eq!(info["design_columns"], json!([]));
    assert_eq!(
        info["dpars"]["mu1"]["group_levels"]["g"]["levels"][0]["value"],
        i64::MIN
    );
    assert_eq!(info["response_levels"][0]["value"], i64::MIN);
    Ok(())
}
