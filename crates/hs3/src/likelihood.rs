//! likelihoods -> likelihoodof / joint_likelihood (06-measure-algebra.md).
use crate::builder::Builder;
use crate::model::Likelihood;
use flatppl_core::id::NodeId;
use std::collections::HashMap;

/// Emit a likelihood binding.
///
/// `data_map` maps datum names to their observed value vectors (from
/// `Document.data`).  When a likelihood's `data[i]` is a string that appears
/// in `data_map`, the values are emitted as an inline `vector(...)` literal
/// bound to a local name and self-referenced.  If the name is not in
/// `data_map`, it is emitted as a `self_ref` (legacy pyhf path).
pub fn emit_likelihood(b: &mut Builder, lk: &Likelihood, data_map: &HashMap<String, Vec<f64>>) {
    if lk.distributions.is_empty() {
        return;
    }
    let mut terms: Vec<NodeId> = Vec::new();
    for (i, dist) in lk.distributions.iter().enumerate() {
        let model = b.self_ref(dist);
        let obs = match lk.data.get(i) {
            Some(serde_json::Value::String(name)) => {
                if let Some(vals) = data_map.get(name.as_str()) {
                    // Bind the datum as a named vector, then self-ref it.
                    let elems: Vec<NodeId> = vals.iter().map(|&v| b.lit_real(v)).collect();
                    let vec_node = b.array(&elems);
                    b.bind(name, vec_node);
                    b.self_ref(name)
                } else {
                    // No datum found — treat as a self_ref (pyhf / pre-existing path).
                    b.self_ref(name)
                }
            }
            Some(serde_json::Value::Number(n)) => b.lit_real(n.as_f64().unwrap_or(0.0)),
            _ => b.lit_real(0.0),
        };
        terms.push(b.call("likelihoodof", &[model, obs]));
    }
    let combined = if terms.len() == 1 {
        terms[0]
    } else {
        b.call("joint_likelihood", &terms)
    };
    b.bind(&lk.name, combined);
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
        let empty_map = HashMap::new();
        {
            let mut b = Builder::new(&mut m);
            emit_likelihood(&mut b, &lk, &empty_map);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("joint_likelihood("), "got:\n{text}");
        assert!(text.contains("likelihoodof("), "got:\n{text}");
    }
}
