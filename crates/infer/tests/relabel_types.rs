//! Spec §04 "Interface adaptation": `relabel` assigns or renames fields. An array
//! becomes a record, a scalar a one-field record, a record or table is renamed, and
//! the same renaming lifts to measures. Names that are not static leave the type
//! deferred.

use flatppl_infer::{Severity, infer};

fn infer_types(source: &str, names: &[&str]) -> Vec<String> {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{source}: {errors:?}");
    names
        .iter()
        .map(|name| {
            let rhs = module
                .bindings()
                .find(|(_, b)| module.resolve(b.name) == *name)
                .unwrap()
                .1
                .rhs;
            module.display_type(module.type_of(rhs).unwrap())
        })
        .collect()
}

#[test]
fn the_spec_value_examples_give_records() {
    let types = infer_types(
        "v1 = relabel([1.0, 2.0, 3.0], [\"x\", \"y\", \"z\"])\n\
         v2 = relabel(record(a = 1.0, b = 2.0, c = 3.0), [\"x\", \"y\", \"z\"])\n\
         v3 = relabel(1.0, [\"x\"])\n",
        &["v1", "v2", "v3"],
    );
    assert_eq!(types[0], types[1], "array and record relabel agree");
    for (ty, fields) in types
        .iter()
        .zip([&["x", "y", "z"][..], &["x", "y", "z"], &["x"]])
    {
        assert!(ty.starts_with('{'), "not a record: {ty}");
        for field in fields {
            assert!(
                ty.contains(&format!("{field}: real")),
                "`{field}` not in {ty}"
            );
        }
    }
}

#[test]
fn a_relabelled_measure_has_a_record_variate() {
    let types = infer_types(
        "mean = [0.0, 0.0, 0.0]\n\
         cov = rowstack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])\n\
         m1 = relabel(MvNormal(mu = mean, cov = cov), [\"a\", \"b\", \"c\"])\n\
         m2 = pushfwd(fn(relabel(_, [\"a\", \"b\", \"c\"])), MvNormal(mu = mean, cov = cov))\n",
        &["m1", "m2"],
    );
    for ty in &types {
        assert!(
            ty.starts_with("measure<{"),
            "the variate is not a record: {ty}"
        );
        for field in ["a", "b", "c"] {
            assert!(
                ty.contains(&format!("{field}: real")),
                "`{field}` not in {ty}"
            );
        }
    }
}

#[test]
fn the_spec_patterns_likelihood_over_a_relabelled_measure_infers_clean() {
    // A record observation of a `relabel`led MvNormal is a variate of it (§04), so
    // the §06 shape check of `likelihoodof` accepts it.
    infer_types(
        "mean = [0.0, 0.0, 0.0]\n\
         cov = rowstack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])\n\
         mvmodel = pushfwd(fn(relabel(_, [\"a\", \"b\", \"c\"])), MvNormal(mu = mean, cov = cov))\n\
         L_mv = likelihoodof(functionof(mvmodel), record(a = 1.1, b = 2.1, c = 3.1))\n",
        &["L_mv"],
    );
}

#[test]
fn a_count_mismatch_is_not_guessed() {
    let types = infer_types("v1 = relabel([1.0, 2.0], [\"x\", \"y\", \"z\"])\n", &["v1"]);
    assert!(
        !types[0].starts_with('{'),
        "a count mismatch must not be guessed: {}",
        types[0]
    );
}
