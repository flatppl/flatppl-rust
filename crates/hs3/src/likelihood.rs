//! likelihoods -> likelihoodof / joint_likelihood (06-measure-algebra.md).
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::model::Likelihood;
use flatppl_core::id::NodeId;
use std::collections::BTreeMap;

/// Emit a likelihood binding.
///
/// `data_map` maps datum names to their observed value vectors (built from the
/// document's unbinned `data` entries). A likelihood's `data[i]` string is
/// resolved strictly:
///   - if it names an entry in `data_map`, the values are emitted as an inline
///     `vector(...)` literal bound to that name and self-referenced;
///   - otherwise it resolves to no observation. A datum is only ever made
///     available by name through `data_map` (unbinned data); binned/histfactory
///     observations are consumed by the channel-assembly path, never reach this
///     emitter, and are not bound by name. So a `data[i]` not in `data_map` is a
///     dangling observation reference — emitting a bare `self_ref` would either
///     produce an unbound name or silently bind the observation to an unrelated
///     top-level binding (a distribution / parameter / function with a colliding
///     name). Both are wrong, and the round-trip gate cannot catch them
///     (syntactically valid), so this is rejected with [`Error::Unsupported`].
///
/// A numeric `data[i]` is emitted as a scalar literal.
pub fn emit_likelihood(
    b: &mut Builder,
    lk: &Likelihood,
    data_map: &BTreeMap<String, Vec<f64>>,
) -> Result<()> {
    if lk.distributions.is_empty() {
        return Ok(());
    }
    let mut terms: Vec<NodeId> = Vec::new();
    for (i, dist) in lk.distributions.iter().enumerate() {
        let model = b.self_ref(dist);
        let obs = match lk.data.get(i) {
            Some(serde_json::Value::String(name)) => {
                let vals = data_map.get(name.as_str()).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "likelihood `{}` data reference `{name}` resolves to no datum",
                        lk.name
                    ))
                })?;
                // Bind the datum as a named vector, then self-ref it.
                let elems: Vec<NodeId> = vals.iter().map(|&v| b.lit_real(v)).collect();
                let vec_node = b.array(&elems);
                b.bind(name, vec_node);
                b.self_ref(name)
            }
            Some(serde_json::Value::Number(n)) => b.lit_real(n.as_f64().unwrap_or(0.0)),
            other => {
                return Err(Error::Unsupported(format!(
                    "likelihood `{}` data[{i}] is not a datum-name string or number: {other:?}",
                    lk.name
                )));
            }
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
        };
        // Both data refs are real unbinned data (the only by-name datum source).
        let mut map = BTreeMap::new();
        map.insert("obs_data".to_string(), vec![3.0]);
        map.insert("aux_obs".to_string(), vec![4.0]);
        {
            let mut b = Builder::new(&mut m);
            emit_likelihood(&mut b, &lk, &map).expect("unbinned data refs resolve");
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("joint_likelihood("), "got:\n{text}");
        assert!(text.contains("likelihoodof("), "got:\n{text}");
    }

    #[test]
    fn data_ref_in_map_inlines_vector() {
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
        };
        let mut map = BTreeMap::new();
        map.insert("d".to_string(), vec![1.0, 2.0, 3.0]);
        {
            let mut b = Builder::new(&mut m);
            emit_likelihood(&mut b, &lk, &map).expect("data_map ref inlines");
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("likelihoodof("), "got:\n{text}");
        // The datum is inlined as a vector literal.
        assert!(
            text.contains("vector") || text.contains('['),
            "got:\n{text}"
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
            // `nowhere` is not in data_map — a dangling reference.
            data: vec![serde_json::json!("nowhere")],
        };
        let empty_map = BTreeMap::new();
        let mut b = Builder::new(&mut m);
        let result = emit_likelihood(&mut b, &lk, &empty_map);
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

    // A data ref that collides with an existing top-level binding (a model
    // component / parameter / function) must NOT silently bind the observation
    // to it. With path #2 removed, this is rejected exactly like any other
    // not-in-data_map reference.
    #[test]
    fn data_ref_colliding_with_binding_errors() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let one = b.lit_real(1.0);
            b.bind("model", one);
            // `decoy` exists as a binding but is NOT observed data.
            let decoy = b.lit_real(7.0);
            b.bind("decoy", decoy);
        }
        let lk = Likelihood {
            name: "L".into(),
            distributions: vec!["model".into()],
            data: vec![serde_json::json!("decoy")],
        };
        let empty_map = BTreeMap::new();
        let mut b = Builder::new(&mut m);
        let result = emit_likelihood(&mut b, &lk, &empty_map);
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
