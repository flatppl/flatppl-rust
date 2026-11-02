//! Black-box describe replies (flatppl-bridge protocol, "Describe").

use std::collections::BTreeMap;
use std::sync::Arc;

use flatppl_core::{BoxBinding, BoxKind, BoxMember, BoxSet, Module, Node, Type};
use flatppl_infer::Severity;
use serde_json::Value;

/// A parsed describe reply. Its set types use the interner of `sets`.
#[derive(Clone, Debug)]
pub struct BoxSpec {
    pub binding: BoxBinding,
    pub sets: Arc<Module>,
}

/// Parse the describe JSON of box `name` into a [`BoxSpec`].
///
/// Each value set is a FlatPPL set expression. The parser types its elements
/// with the ordinary inference trace, so a set the language cannot express
/// fails here and not at query time.
pub fn parse_describe(name: &str, json: &str) -> Result<BoxSpec, String> {
    let valid_name = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid_name {
        return Err(format!("box name `{name}` is not an identifier"));
    }
    let root: Value =
        serde_json::from_str(json).map_err(|e| format!("box `{name}`: describe JSON: {e}"))?;
    let Value::Object(entries) = root else {
        return Err(format!("box `{name}`: describe JSON is not an object"));
    };
    let mut texts = Vec::new();
    let mut members = Vec::new();
    for (member, entry) in &entries {
        let fail = |what: &str| format!("box `{name}` member `{member}`: {what}");
        let Value::Object(fields) = entry else {
            return Err(fail("entry is not an object"));
        };
        let kind = match fields.get("kind").and_then(Value::as_str) {
            Some("function") => BoxKind::Function,
            Some("kernel") => BoxKind::Kernel,
            Some("measure") => BoxKind::Measure,
            _ => {
                return Err(fail(
                    "`kind` must be \"function\", \"kernel\" or \"measure\"",
                ));
            }
        };
        let queries = match fields.get("queries") {
            Some(Value::Array(items)) => items
                .iter()
                .map(|q| q.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| fail("`queries` must be an array of strings"))?,
            _ => return Err(fail("`queries` must be an array of strings")),
        };
        let flag = |key: &str| match fields.get(key) {
            None => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(fail(&format!("`{key}` must be a boolean"))),
        };
        let (gradient, vjp, normalized) = (flag("gradient")?, flag("vjp")?, flag("normalized")?);
        let mut set = |key: &str, required: bool| -> Result<Option<usize>, String> {
            match fields.get(key) {
                Some(Value::String(text)) => {
                    texts.push(text.clone());
                    Ok(Some(texts.len() - 1))
                }
                None if !required => Ok(None),
                _ => Err(fail(&format!("`{key}` must be a value-set string"))),
            }
        };
        let (input, output, variate) = match kind {
            BoxKind::Function => (set("input", true)?, set("output", true)?, None),
            BoxKind::Kernel => (set("input", true)?, None, set("variate", true)?),
            BoxKind::Measure => (None, None, set("variate", true)?),
        };
        members.push((
            member.clone(),
            kind,
            queries,
            [gradient, vjp, normalized],
            [input, output, variate],
        ));
    }
    let (sets, types) = type_sets(name, &texts)?;
    let set_of = |index: Option<usize>| {
        index.map(|i| BoxSet {
            text: texts[i].as_str().into(),
            ty: types[i].clone(),
        })
    };
    let members = members
        .into_iter()
        .map(
            |(member, kind, queries, [gradient, vjp, normalized], [i, o, v])| {
                (
                    member.into_boxed_str(),
                    BoxMember {
                        kind,
                        queries: queries.into_iter().map(String::into_boxed_str).collect(),
                        gradient,
                        vjp,
                        normalized,
                        input: set_of(i),
                        output: set_of(o),
                        variate: set_of(v),
                    },
                )
            },
        )
        .collect::<BTreeMap<_, _>>();
    Ok(BoxSpec {
        binding: BoxBinding {
            name: name.into(),
            members,
        },
        sets: Arc::new(sets),
    })
}

/// Type the elements of each set through `elementof`. `anything` declares no
/// shape and stays [`Type::Any`].
fn type_sets(name: &str, texts: &[String]) -> Result<(Module, Vec<Type>), String> {
    let source: String = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            if text.contains('\n') || text.contains('#') {
                return Err(format!(
                    "box `{name}`: value set `{text}` is not one expression"
                ));
            }
            Ok(format!("s{i} = elementof({text})\n"))
        })
        .collect::<Result<_, _>>()?;
    let mut module = flatppl_syntax::parse(&source)
        .map_err(|e| format!("box `{name}`: a value set does not parse: {}", e.message))?;
    let diagnostics = flatppl_infer::infer(&mut module);
    if let Some(error) = diagnostics.iter().find(|d| d.severity == Severity::Error) {
        return Err(format!(
            "box `{name}`: a value set is invalid: {}",
            error.message
        ));
    }
    let types = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            if text.trim() == "anything" {
                return Ok(Type::Any);
            }
            let symbol = module.intern(&format!("s{i}"));
            let binding = module
                .binding_by_name(symbol)
                .ok_or_else(|| format!("box `{name}`: value set `{text}` is not one binding"))?;
            let rhs = module.binding(binding).rhs;
            let Node::Call(_) = module.node(rhs) else {
                return Err(format!("box `{name}`: value set `{text}` is not a set"));
            };
            match module.type_of(rhs) {
                Some(ty) if !matches!(ty, Type::Deferred | Type::Failed(_)) => Ok(ty.clone()),
                _ => Err(format!(
                    "box `{name}`: value set `{text}` has no element type"
                )),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((module, types))
}

/// Bind top-level `load_module` aliases of `root` to boxes. Every public
/// binding of a bound interface module must be an `external(anything)` member
/// that the box describes.
pub fn bind_boxes(
    root: &mut Module,
    bundle: &flatppl_infer::ModuleBundle,
    boxes: &[(String, Arc<BoxSpec>)],
) -> Result<(), String> {
    use flatppl_core::CallHead;
    for (alias, spec) in boxes {
        let box_name = &spec.binding.name;
        let symbol = root.intern(alias);
        let binding = root
            .binding_by_name(symbol)
            .ok_or_else(|| format!("box alias `{alias}` is not a binding of the model"))?;
        let literal = match root.node(root.binding(binding).rhs) {
            Node::Call(call)
                if matches!(call.head, CallHead::Builtin(h) if root.resolve(h) == "load_module") =>
            {
                flatppl_infer::module_source(root, call)
            }
            _ => None,
        }
        .ok_or_else(|| format!("box alias `{alias}` is not a `load_module` binding"))?;
        let interface = bundle
            .get_from(bundle.root(), literal)
            .ok_or_else(|| format!("box interface `{literal}` of `{alias}` is not loaded"))?;
        for (_, b) in interface.public_bindings() {
            let name = interface.resolve(b.name);
            if !is_external_anything(interface, b.rhs) {
                return Err(format!(
                    "box interface `{literal}`: member `{name}` must be declared `external(anything)`"
                ));
            }
            if !spec.binding.members.contains_key(name) {
                return Err(format!(
                    "box `{box_name}` has no member `{name}` (interface `{literal}`)"
                ));
            }
        }
        let reintern = |root: &mut Module, set: &Option<BoxSet>| {
            set.as_ref().map(|set| BoxSet {
                text: set.text.clone(),
                ty: flatppl_infer::reintern_type(root, &spec.sets, &set.ty),
            })
        };
        let mut members = BTreeMap::new();
        for (name, member) in &spec.binding.members {
            let input = reintern(root, &member.input);
            let output = reintern(root, &member.output);
            let variate = reintern(root, &member.variate);
            members.insert(
                name.clone(),
                BoxMember {
                    input,
                    output,
                    variate,
                    ..member.clone()
                },
            );
        }
        root.bind_box(
            symbol,
            Arc::new(BoxBinding {
                name: box_name.clone(),
                members,
            }),
        );
    }
    Ok(())
}

fn is_external_anything(m: &Module, rhs: flatppl_core::NodeId) -> bool {
    use flatppl_core::CallHead;
    let Node::Call(call) = m.node(rhs) else {
        return false;
    };
    matches!(call.head, CallHead::Builtin(h) if m.resolve(h) == "external")
        && call.named.is_empty()
        && matches!(call.args[..], [arg] if matches!(m.node(arg), Node::Const(s) if m.resolve(*s) == "anything"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_protocol_example() {
        let spec = parse_describe(
            "normal",
            r#"{
              "posterior": {"kind": "measure", "queries": ["builtin_logdensityof"], "gradient": true, "vjp": false, "variate": "cartpow(reals, 2)", "normalized": false},
              "f": {"kind": "function", "queries": ["apply"], "gradient": true, "vjp": true, "input": "cartpow(reals, 3)", "output": "reals"},
              "K": {"kind": "kernel", "queries": ["builtin_logdensityof", "builtin_sample"], "gradient": false, "vjp": false, "input": "cartprod(mu = reals, sigma = posreals)", "variate": "reals", "normalized": true}
            }"#,
        )
        .unwrap();
        let k = &spec.binding.members["K"];
        assert_eq!(
            flatppl_infer::box_input_names(&spec.sets, k.input.as_ref()),
            ["mu", "sigma"]
        );
        assert!(k.normalized && k.supports("builtin_sample"));
        let posterior = &spec.binding.members["posterior"];
        assert!(matches!(
            posterior.variate.as_ref().unwrap().ty,
            Type::Array { .. }
        ));
    }

    #[test]
    fn refuses_bad_entries() {
        for (json, message) in [
            ("[]", "not an object"),
            (r#"{"m": {"kind": "thing", "queries": []}}"#, "`kind`"),
            (r#"{"m": {"kind": "measure", "queries": []}}"#, "`variate`"),
            (
                r#"{"m": {"kind": "measure", "queries": [], "variate": "cartpow(reals,"}}"#,
                "does not parse",
            ),
        ] {
            let error = parse_describe("b", json).unwrap_err();
            assert!(error.contains(message), "{error}");
        }
        assert!(
            parse_describe("1b", "{}")
                .unwrap_err()
                .contains("identifier")
        );
    }
}
