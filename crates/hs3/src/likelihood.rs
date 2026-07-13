//! likelihoods -> likelihoodof / joint_likelihood (06-measure-algebra.md).
use crate::builder::Builder;
use crate::data::DataShape;
use crate::error::{Error, Result};
use crate::model::Likelihood;
use flatppl_core::id::NodeId;
use std::collections::BTreeMap;

/// Emit a likelihood binding, observing the embedded `data` tables (emitted as
/// `<name> = table(...)` by the data step).
///
/// A likelihood's `data[i]` is resolved strictly:
///   - a string naming a dataset in `data_shapes`: observe its embedded table.
///     A single-axis dataset is observed against its column vector
///     `<name>.<axis>` under an `iid(<model>, n_rows)` plate — the column holds
///     the N iid scalar observations (06-measure-algebra.md). A multi-axis
///     dataset is observed against the table itself: each row is one event over
///     the observable axes (spec §03: a multivariate event sample IS a table).
///     A scalar (`point`) datum is observed as the binding itself (`self_ref`),
///     with no `iid` plate.
///   - a number: a scalar observation literal (no plate).
///   - any other string resolves to no dataset (binned/histfactory observations
///     are consumed by the channel-assembly path, never reached by name here),
///     so it is a dangling reference — rejected with [`Error::Unsupported`]
///     rather than emitting an unbound/mis-bound name the round-trip gate cannot
///     catch.
pub fn emit_likelihood(
    b: &mut Builder,
    lk: &Likelihood,
    data_shapes: &BTreeMap<String, DataShape>,
) -> Result<()> {
    if !lk.aux_distributions.is_empty() {
        return Err(Error::Unimplemented(format!(
            "likelihood `{}` declares aux_distributions (auxiliary likelihood terms)",
            lk.name
        )));
    }
    if lk.distributions.is_empty() {
        return Ok(());
    }
    let mut terms: Vec<NodeId> = Vec::new();
    for (i, dist) in lk.distributions.iter().enumerate() {
        let model0 = b.self_ref(dist);
        // Number of iid rows to plate the model over (None ⇒ a scalar datum).
        let mut iid_n: Option<usize> = None;
        let obs = match lk.data.get(i) {
            Some(serde_json::Value::String(name)) => {
                let shape = data_shapes.get(name.as_str()).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "likelihood `{}` data reference `{name}` resolves to no datum",
                        lk.name
                    ))
                })?;
                if shape.scalar {
                    // A `point` datum: scalar binding, no iid plate.
                    b.self_ref(name)
                } else {
                    iid_n = Some(shape.n_rows);
                    let table = b.self_ref(name);
                    if shape.columns.len() == 1 {
                        // Single observable: observe the column vector
                        // `<name>.<axis>` (`get(table, "axis")` prints as
                        // `<name>.<axis>`; a table column is a vector, spec
                        // §03) — N iid scalar observations.
                        let key = b.str_lit(&shape.columns[0]);
                        b.call("get", &[table, key])
                    } else {
                        // Multivariate event sample: observe the table directly.
                        table
                    }
                }
            }
            Some(serde_json::Value::Number(n)) => b.lit_real(n.as_f64().unwrap_or(0.0)),
            other => {
                return Err(Error::Unsupported(format!(
                    "likelihood `{}` data[{i}] is not a datum-name string or number: {other:?}",
                    lk.name
                )));
            }
        };
        let model = match iid_n {
            Some(n) => {
                let n_lit = b.lit_int(n as i64);
                b.call("iid", &[model0, n_lit])
            }
            None => model0,
        };
        terms.push(b.call("likelihoodof", &[model, obs]));
    }
    let combined = if terms.len() == 1 {
        terms[0]
    } else {
        b.call("joint_likelihood", &terms)
    };
    b.bind(&lk.name, combined);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use crate::model::Likelihood;
    use flatppl_syntax::{Syntax, print_with};

    fn shape(columns: &[&str], n_rows: usize) -> DataShape {
        DataShape {
            columns: columns.iter().map(|s| s.to_string()).collect(),
            n_rows,
            scalar: false,
        }
    }

    #[test]
    fn two_term_joint_likelihood() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let one = b.lit_real(1.0);
            b.bind("obs_model", one);
            let two = b.lit_real(2.0);
            b.bind("aux_model", two);
        }
        let lk = Likelihood {
            name: "L".into(),
            distributions: vec!["obs_model".into(), "aux_model".into()],
            data: vec![serde_json::json!("obs_data"), serde_json::json!("aux_obs")],
            aux_distributions: vec![],
        };
        let mut shapes = BTreeMap::new();
        shapes.insert("obs_data".to_string(), shape(&["x"], 1));
        shapes.insert("aux_obs".to_string(), shape(&["y"], 1));
        {
            let mut b = Builder::new(&mut m);
            emit_likelihood(&mut b, &lk, &shapes).expect("data refs resolve");
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("joint_likelihood("), "got:\n{text}");
        assert!(text.contains("likelihoodof("), "got:\n{text}");
    }

    #[test]
    fn single_axis_data_observes_column_under_iid() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let one = b.lit_real(1.0);
            b.bind("model", one);
        }
        let lk = Likelihood {
            name: "L".into(),
            distributions: vec!["model".into()],
            data: vec![serde_json::json!("d")],
            aux_distributions: vec![],
        };
        let mut shapes = BTreeMap::new();
        shapes.insert("d".to_string(), shape(&["x"], 3));
        {
            let mut b = Builder::new(&mut m);
            emit_likelihood(&mut b, &lk, &shapes).expect("data ref resolves");
        }
        let text = print_with(&m, Syntax::Minimal);
        // 3 rows over one observable → iid plate, observed against the column
        // vector (`get(d, "x")` is the canonical `t.col` column access, spec §03).
        assert!(
            text.contains("likelihoodof(iid(model, 3), get(d, \"x\"))"),
            "single-axis column observation expected, got:\n{text}"
        );
    }

    #[test]
    fn multi_axis_data_observes_table() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let one = b.lit_real(1.0);
            b.bind("model", one);
        }
        let lk = Likelihood {
            name: "L".into(),
            distributions: vec!["model".into()],
            data: vec![serde_json::json!("d")],
            aux_distributions: vec![],
        };
        let mut shapes = BTreeMap::new();
        shapes.insert("d".to_string(), shape(&["x", "y"], 5));
        {
            let mut b = Builder::new(&mut m);
            emit_likelihood(&mut b, &lk, &shapes).expect("data ref resolves");
        }
        let text = print_with(&m, Syntax::Minimal);
        // Multivariate sample → observe the whole table under the iid plate.
        assert!(
            text.contains("likelihoodof(iid(model, 5), d)"),
            "table observation expected, got:\n{text}"
        );
    }

    #[test]
    fn dangling_data_ref_errors() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let one = b.lit_real(1.0);
            b.bind("model", one);
        }
        let lk = Likelihood {
            name: "L".into(),
            distributions: vec!["model".into()],
            // `nowhere` is not a known dataset — a dangling reference.
            data: vec![serde_json::json!("nowhere")],
            aux_distributions: vec![],
        };
        let empty = BTreeMap::new();
        let mut b = Builder::new(&mut m);
        let result = emit_likelihood(&mut b, &lk, &empty);
        assert!(
            matches!(result, Err(Error::Unsupported(_))),
            "got: {result:?}"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nowhere") && msg.contains("resolves to no datum"),
            "got: {msg}"
        );
    }

    // A data ref that collides with a non-datum top-level binding must NOT
    // silently bind to it; with no matching dataset it is rejected like any
    // other unknown reference.
    #[test]
    fn data_ref_colliding_with_binding_errors() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let one = b.lit_real(1.0);
            b.bind("model", one);
            let decoy = b.lit_real(7.0);
            b.bind("decoy", decoy);
        }
        let lk = Likelihood {
            name: "L".into(),
            distributions: vec!["model".into()],
            data: vec![serde_json::json!("decoy")],
            aux_distributions: vec![],
        };
        let empty = BTreeMap::new();
        let mut b = Builder::new(&mut m);
        let result = emit_likelihood(&mut b, &lk, &empty);
        assert!(
            matches!(result, Err(Error::Unsupported(_))),
            "got: {result:?}"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("resolves to no datum"),
            "collision with a binding must still be rejected"
        );
    }
}
