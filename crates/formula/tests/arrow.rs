#![cfg(feature = "arrow")]

use arrow_array::{
    ArrayRef, BooleanArray, DictionaryArray, Int8Array, Int64Array, RecordBatch, StringArray,
    types::Int8Type,
};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use flatppl_formula::{DataMode, DataValue, Recipe, Table, build, prepare_data, read};
use std::sync::Arc;

fn stream(batches: &[RecordBatch]) -> Result<Vec<u8>, arrow_schema::ArrowError> {
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::try_new(&mut bytes, batches[0].schema().as_ref())?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(bytes)
}

#[test]
fn pyarrow_file_and_stream_preserve_precision_and_prediction_recipe()
-> Result<(), Box<dyn std::error::Error>> {
    let spec = read(r#"{"formula":"y ~ x + f"}"#)?;
    let prediction = Table::from_arrow_ipc(include_bytes!("fixtures/arrow/prediction.arrows"))?;
    for bytes in [
        include_bytes!("fixtures/arrow/training.arrow").as_slice(),
        include_bytes!("fixtures/arrow/training.arrows").as_slice(),
    ] {
        let table = Table::from_arrow_ipc(bytes)?;
        let model = build(&spec, &table)?;
        assert_eq!(model.info.training_rows, vec![0, 1, 3]);
        assert_eq!(
            model.data.constants["x_1"],
            DataValue::Real32(vec![0., 1., 3.])
        );
        assert_eq!(
            model.data.constants["x_2"],
            DataValue::Integers(vec![1, 2, 2])
        );
        let factor = model.recipe.predictors["mu"].design.predictors[1]
            .factor
            .as_ref()
            .unwrap();
        assert_eq!(factor.levels, vec!["b", "a"]);
        assert!(factor.ordered);
        assert_eq!(model.recipe.predictors["mu"].means[0], 4. / 3.);
        let recipe: Recipe = serde_json::from_str(&serde_json::to_string(&model.recipe)?)?;
        let data = prepare_data(&recipe, &prediction, DataMode::Prediction)?;
        assert_eq!(data.rows, vec![0, 2]);
        assert_eq!(data.constants["x_1"], DataValue::Real32(vec![2., 4.]));
        assert_eq!(data.constants["x_2"], DataValue::Integers(vec![2, 1]));
    }
    Ok(())
}

#[test]
fn ordered_dictionaries_merge_constraints_and_recode_prior_batches()
-> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("y", DataType::Int64, false),
        Field::new(
            "f",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        )
        .with_dict_is_ordered(true),
    ]));
    for (dictionaries, expected_levels, expected_codes) in [
        (
            vec![vec!["a", "c"], vec!["a", "b", "c"]],
            vec!["a", "b", "c"],
            vec![1, 3, 1, 2, 3],
        ),
        (
            vec![vec!["a", "c"], vec!["b", "c"], vec!["b", "a", "c"]],
            vec!["b", "a", "c"],
            vec![2, 3, 1, 3, 1, 2, 3],
        ),
    ] {
        let batches = dictionaries
            .into_iter()
            .map(|labels| {
                let n = labels.len();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from_iter_values(0..n as i64)),
                        Arc::new(DictionaryArray::<Int8Type>::try_new(
                            Int8Array::from_iter_values(0..n as i8),
                            Arc::new(StringArray::from(labels)),
                        )?),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let table = Table::from_arrow_ipc(&stream(&batches)?)?;
        let model = build(&read(r#"{"formula":"y ~ f"}"#)?, &table)?;
        assert_eq!(
            model.recipe.predictors["mu"].design.predictors[0]
                .factor
                .as_ref()
                .unwrap()
                .levels,
            expected_levels
        );
        assert_eq!(
            model.data.constants["x_1"],
            DataValue::Integers(expected_codes)
        );
    }
    Ok(())
}

#[test]
fn dictionary_replacements_and_missing_values_preserve_model_rows()
-> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("y", DataType::Int64, false),
        Field::new(
            "f",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new("s", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let first: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1, 2, 3])),
        Arc::new(DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![Some(0), Some(1), Some(2)]),
            Arc::new(StringArray::from(vec![Some("b"), None, Some("a")])),
        )?),
        Arc::new(StringArray::from(vec!["z", "unused", "a"])),
        Arc::new(BooleanArray::from(vec![
            Some(false),
            Some(false),
            Some(true),
        ])),
    ];
    let second: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![4, 5, 6])),
        Arc::new(DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![None, Some(0), Some(1)]),
            Arc::new(StringArray::from(vec!["a", "b"])),
        )?),
        Arc::new(StringArray::from(vec!["unused", "a", "z"])),
        Arc::new(BooleanArray::from(vec![None, Some(false), Some(true)])),
    ];
    let batches = vec![
        RecordBatch::try_new(schema.clone(), first)?,
        RecordBatch::try_new(schema.clone(), second)?,
    ];
    let table = Table::from_arrow_ipc(&stream(&batches)?)?;
    let model = build(&read(r#"{"formula":"y ~ f + s + flag"}"#)?, &table)?;
    assert_eq!(model.info.training_rows, vec![0, 2, 4, 5]);
    assert_eq!(
        model.data.constants["x_1"],
        DataValue::Integers(vec![1, 2, 2, 1])
    );
    assert_eq!(
        model.data.constants["x_2"],
        DataValue::Integers(vec![2, 1, 1, 2])
    );
    assert_eq!(
        model.data.constants["x_3"],
        DataValue::Integers(vec![1, 2, 1, 2])
    );
    assert_eq!(
        model.recipe.predictors["mu"].design.predictors[0]
            .factor
            .as_ref()
            .unwrap()
            .levels,
        vec!["b", "a"]
    );
    assert_eq!(
        model.recipe.predictors["mu"].design.predictors[1]
            .factor
            .as_ref()
            .unwrap()
            .levels,
        vec!["a", "z"]
    );

    // Reversing physical dictionaries is harmless for unordered categories,
    // but would give contradictory meanings to an ordered predictor.
    let mut fields: Vec<_> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields[1] = fields[1].clone().with_dict_is_ordered(true);
    let ordered = Arc::new(Schema::new(fields));
    let batches = batches
        .into_iter()
        .map(|batch| RecordBatch::try_new(ordered.clone(), batch.columns().to_vec()))
        .collect::<Result<Vec<_>, _>>()?;
    assert!(Table::from_arrow_ipc(&stream(&batches)?).is_err());
    Ok(())
}
