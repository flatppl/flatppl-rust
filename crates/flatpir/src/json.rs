//! FlatPIR JSON (the `.flatpir.json` encoding).
//!
//! A *syntactic* transducer between the canonical S-expression representation
//! and JSON — it carries no FlatPIR semantics of its own. All semantics
//! (the `%meta` type grammar, atom classification, `%autoinputs` side-table
//! projection, canonical float formatting) stay in [`reader`](crate::reader) /
//! [`writer`](crate::writer); the JSON layer leans on them via the text bridge:
//!
//! ```text
//! to_json   :  Module --writer::write--> text --parse_top--> Sexpr --enc--> JSON
//! from_json :  JSON --emit--> canonical text --reader::read--> Module
//! ```
//!
//! Shape: structural kinds (`%module`/`%bind`/`%ref`/`%meta`/`%doc`/
//! `%kwarg`/`%field`/`%assign`/axes, reified input lists) are tagged objects;
//! calls are arrays `[head, …elements]`; atoms are explicit literal wrappers
//! (`{int}`/`{real}`/`{str}`/`{bool}`/`{const}`/`{hole}`).

use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::reader::classify_number;
use crate::sexpr::{self, Sexpr, SexprKind};
use crate::writer::{quote_string, render_real};
use flatppl_core::{Module, Scalar};

// ===========================================================================
// Encode:  Module -> canonical text -> Sexpr -> JSON
// ===========================================================================

/// Encode a [`Module`] as FlatPIR JSON.
///
/// Precondition: the `Module` must be writable as valid FlatPIR (true for any
/// module produced by [`read`](crate::read)). Symbol/const names containing
/// whitespace or any of `()";` are out of contract; on such a module this
/// panics. Use [`try_to_json`] to handle that case as an error instead.
pub fn to_json(module: &Module) -> Value {
    try_to_json(module)
        .expect("to_json: module is not writable as valid FlatPIR (see precondition)")
}

/// Encode a [`Module`] as FlatPIR JSON, returning an error instead of panicking
/// if the module is not writable as valid FlatPIR (e.g. a symbol/const name
/// containing whitespace or `()";`, which the canonical text cannot represent).
pub fn try_to_json(module: &Module) -> Result<Value> {
    let text = crate::writer::write(module);
    // The writer emits canonical, well-formed FlatPIR for any in-contract module,
    // so this re-parse succeeds and the tree shape is what `enc_*` expects. It can
    // only fail if the module holds a name that is not a valid FlatPIR token.
    let forms = sexpr::parse_top(&text)?;
    let form = forms
        .first()
        .ok_or_else(|| Error::new("writer produced no top-level form"))?;
    Ok(enc_module(form))
}

/// The head symbol of a list form, if it has one.
fn list_head(form: &Sexpr) -> Option<&str> {
    form.as_list()
        .and_then(|items| items.first())
        .and_then(Sexpr::as_atom)
}

/// The unescaped text of a string-literal node (caller guarantees the kind).
fn str_text(form: &Sexpr) -> &str {
    match &form.kind {
        SexprKind::Str(s) => s,
        _ => unreachable!("expected a string literal"),
    }
}

fn enc_module(form: &Sexpr) -> Value {
    let items = form.as_list().expect("%module is a list");
    let mut public: Vec<Value> = Vec::new();
    let mut binds: Vec<Value> = Vec::new();
    for elem in &items[1..] {
        let inner = elem.as_list().expect("module element is a list");
        match inner.first().and_then(Sexpr::as_atom) {
            Some("%public") => {
                public = inner[1..]
                    .iter()
                    .map(|n| json!(n.as_atom().expect("public name is a symbol")))
                    .collect();
            }
            Some("%bind") => binds.push(enc_bind(inner)),
            _ => {}
        }
    }
    json!({ "%module": { "public": public, "binds": binds } })
}

fn enc_bind(items: &[Sexpr]) -> Value {
    let name = items[1].as_atom().expect("bind name is a symbol");
    let mut rest = &items[2..];
    let mut doc: Option<Value> = None;
    if let Some(last) = rest.last() {
        if list_head(last) == Some("%doc") {
            doc = Some(enc_doc(last));
            rest = &rest[..rest.len() - 1];
        }
    }
    let mut obj = Map::new();
    obj.insert("name".into(), json!(name));
    obj.insert("expr".into(), enc(&rest[0]));
    if let Some(d) = doc {
        obj.insert("doc".into(), d);
    }
    Value::Object(obj)
}

fn enc_doc(form: &Sexpr) -> Value {
    let items = form.as_list().expect("%doc is a list");
    let lines: Vec<Value> = items[2..].iter().map(|l| json!(str_text(l))).collect();
    json!({ "tag": items[1].as_atom().expect("doc tag is a symbol"), "lines": lines })
}

/// Encode one expression node.
fn enc(form: &Sexpr) -> Value {
    match &form.kind {
        SexprKind::Str(s) => json!({ "str": s }),
        SexprKind::Atom(a) => enc_atom(a),
        SexprKind::List(items) => enc_list(items),
    }
}

fn enc_atom(a: &str) -> Value {
    match a {
        "true" => json!({ "bool": true }),
        "false" => json!({ "bool": false }),
        "_" => json!({ "hole": true }),
        // A dynamic dimension. Encoded as a dedicated tagged atom (not `{const}`
        // / not a bare head) so it is unambiguous in any position, including as
        // the first element of an `%array` shape list (§10.3).
        "%dynamic" => json!({ "%dynamic": true }),
        _ => match classify_number(a) {
            Some(Scalar::Int(n)) => json!({ "int": n }),
            Some(Scalar::Real(r)) => json!({ "real": r }),
            // classify_number only yields Int/Real; a non-number is a bare symbol.
            _ => json!({ "const": a }),
        },
    }
}

/// The head of a list, if it is a *symbol* (not a numeric/boolean literal). A
/// list with a symbol head is a call or structural form; a list whose first
/// element is a literal or sub-list is a bare parenthesised list (e.g. the
/// `(<dim>…)` shape inside an `%array` type).
fn symbol_head(items: &[Sexpr]) -> Option<&str> {
    let h = items.first()?.as_atom()?;
    // `true`/`false`/numbers/`%dynamic` are atom *values*, never call heads — a
    // list led by one (e.g. an `%array` shape `(%dynamic 3)`) is headless.
    if h == "true" || h == "false" || h == "%dynamic" || classify_number(h).is_some() {
        return None;
    }
    Some(h)
}

fn enc_list(items: &[Sexpr]) -> Value {
    let Some(head) = symbol_head(items) else {
        // Headless list (no symbol head): encode every element, no leading head.
        return Value::Array(items.iter().map(enc).collect());
    };
    match head {
        "%ref" => json!({ "%ref": {
            "ns": items[1].as_atom().expect("ref ns"),
            "name": items[2].as_atom().expect("ref name"),
        }}),
        "%axis" | "%uaxis" | "%laxis" => {
            json!({ head: items[1].as_atom().expect("axis name") })
        }
        "%kwarg" | "%field" | "%assign" => json!({ head: {
            "name": items[1].as_atom().expect("named-entry name"),
            "value": enc(&items[2]),
        }}),
        "%meta" => json!({ "%meta": enc_meta(items) }),
        "%call" => {
            let mut arr = vec![json!("%call"), enc(&items[1])];
            arr.extend(items[2..].iter().map(enc));
            Value::Array(arr)
        }
        "functionof" | "kernelof" => enc_reified(head, items),
        _ => {
            let mut arr = vec![json!(head)];
            arr.extend(items[1..].iter().map(enc));
            Value::Array(arr)
        }
    }
}

fn enc_meta(items: &[Sexpr]) -> Value {
    json!({
        "type": enc(&items[1]),
        "phase": items[2].as_atom().expect("phase is a symbol"),
        "valueset": enc(&items[3]),
    })
}

fn enc_reified(head: &str, items: &[Sexpr]) -> Value {
    // (functionof <meta?> <output> <origin> <inputs|%deferred>)
    let mut arr = vec![json!(head)];
    let mut rest = &items[1..];
    if let Some(first) = rest.first() {
        if list_head(first) == Some("%meta") {
            arr.push(enc(first));
            rest = &rest[1..];
        }
    }
    arr.push(enc(&rest[0])); // output
    let origin = rest[1].as_atom().expect("reification origin tag");
    let list = match rest[2].as_atom() {
        Some(tag) => json!(tag), // "%deferred"
        None => {
            // each entry is a 2-list  (<name> (%ref …))
            let entries: Vec<Value> = rest[2]
                .as_list()
                .expect("input list")
                .iter()
                .map(|e| {
                    let pair = e.as_list().expect("input entry is a pair");
                    json!([pair[0].as_atom().expect("input name"), enc(&pair[1])])
                })
                .collect();
            Value::Array(entries)
        }
    };
    arr.push(json!({ "%inputs": { "origin": origin, "list": list } }));
    Value::Array(arr)
}

// ===========================================================================
// Decode:  JSON -> canonical text -> reader::read -> Module
// ===========================================================================

/// Decode FlatPIR JSON into a [`Module`] (via the canonical-text reader).
pub fn from_json(value: &Value) -> Result<Module> {
    let text = emit_module(value)?;
    crate::read(&text)
}

fn obj<'a>(v: &'a Value, ctx: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error::new(format!("expected a JSON object for {ctx}")))
}

fn get<'a>(o: &'a Map<String, Value>, key: &str, ctx: &str) -> Result<&'a Value> {
    o.get(key)
        .ok_or_else(|| Error::new(format!("missing `{key}` in {ctx}")))
}

fn as_str<'a>(v: &'a Value, ctx: &str) -> Result<&'a str> {
    v.as_str()
        .ok_or_else(|| Error::new(format!("expected a string for {ctx}")))
}

fn emit_module(value: &Value) -> Result<String> {
    let root = obj(value, "the top-level form")?;
    let m = obj(get(root, "%module", "the top-level form")?, "%module")?;
    let mut out = String::from("(%module");

    let publics = get(m, "public", "%module")?
        .as_array()
        .ok_or_else(|| Error::new("`public` must be an array"))?;
    // Always emit the public form, even when empty: a hand-authored empty
    // `public` array must decode to a module with no public bindings, not fall
    // through to the reader's name-convention fallback.
    out.push_str("\n  (%public");
    for p in publics {
        out.push(' ');
        out.push_str(as_str(p, "a public name")?);
    }
    out.push(')');

    for b in get(m, "binds", "%module")?
        .as_array()
        .ok_or_else(|| Error::new("`binds` must be an array"))?
    {
        out.push_str("\n\n  ");
        out.push_str(&emit_bind(b)?);
    }
    out.push(')');
    Ok(out)
}

fn emit_bind(value: &Value) -> Result<String> {
    let b = obj(value, "a binding")?;
    let name = as_str(get(b, "name", "a binding")?, "a binding name")?;
    let expr = emit(get(b, "expr", "a binding")?, 0)?;
    let mut s = format!("(%bind {name} {expr}");
    if let Some(doc) = b.get("doc") {
        s.push(' ');
        s.push_str(&emit_doc(doc)?);
    }
    s.push(')');
    Ok(s)
}

fn emit_doc(value: &Value) -> Result<String> {
    let d = obj(value, "a doc form")?;
    let tag = as_str(get(d, "tag", "a doc form")?, "a doc tag")?;
    let mut parts = vec![format!("%doc {tag}")];
    for line in get(d, "lines", "a doc form")?
        .as_array()
        .ok_or_else(|| Error::new("doc `lines` must be an array"))?
    {
        parts.push(quote_string(as_str(line, "a doc line")?));
    }
    Ok(format!("({})", parts.join(" ")))
}

/// Maximum JSON expression nesting `from_json` will follow. Past this, decoding
/// returns `Err` rather than recursing until the native stack overflows (which
/// would abort the process, uncatchably). Generous for any real FlatPIR program.
const MAX_DEPTH: usize = 128;

/// Emit one expression. The result may be more than one space-separated token
/// (the reified `%inputs` element expands to `<origin> <list>`); callers always
/// join emitted pieces with spaces, so that is fine. `depth` bounds recursion on
/// adversarial input (see [`MAX_DEPTH`]).
fn emit(value: &Value, depth: usize) -> Result<String> {
    if depth > MAX_DEPTH {
        return Err(Error::new("JSON expression nesting too deep"));
    }
    match value {
        Value::Array(arr) => emit_call(arr, depth),
        Value::Object(o) => emit_obj(o, depth),
        _ => Err(Error::new(format!("not a FlatPIR expression: {value}"))),
    }
}

fn emit_obj(o: &Map<String, Value>, depth: usize) -> Result<String> {
    // A node object carries exactly one recognized discriminator key. More than
    // one (e.g. `{"int":3,"real":4}` or `{"int":1,"%ref":…}`) is ambiguous: the
    // ladder below would pick the first and silently drop the rest.
    let node_keys = [
        "int", "real", "bool", "str", "const", "hole", "%dynamic", "%ref", "%axis", "%uaxis",
        "%laxis", "%kwarg", "%field", "%assign", "%meta", "%inputs",
    ];
    if node_keys.iter().filter(|k| o.contains_key(**k)).count() > 1 {
        return Err(Error::new(
            "ambiguous node object: multiple recognized keys",
        ));
    }
    if let Some(v) = o.get("int") {
        return Ok(v
            .as_i64()
            .ok_or_else(|| Error::new("`int` must be an integer"))?
            .to_string());
    }
    if let Some(v) = o.get("real") {
        return Ok(render_real(
            v.as_f64()
                .ok_or_else(|| Error::new("`real` must be a number"))?,
        ));
    }
    if let Some(v) = o.get("bool") {
        return Ok(if v
            .as_bool()
            .ok_or_else(|| Error::new("`bool` must be a boolean"))?
        {
            "true"
        } else {
            "false"
        }
        .to_string());
    }
    if let Some(v) = o.get("str") {
        return Ok(quote_string(as_str(v, "a string literal")?));
    }
    if let Some(v) = o.get("const") {
        let s = as_str(v, "a const symbol")?;
        // A const that would re-classify on re-read (hole/bool/number) is not
        // representable as a bare atom: reject rather than silently mutate it.
        if s == "_" || s == "true" || s == "false" || classify_number(s).is_some() {
            return Err(Error::new(format!(
                "const symbol is not representable as a bare atom: {s}"
            )));
        }
        return Ok(s.to_string());
    }
    if o.get("hole").is_some() {
        return Ok("_".to_string());
    }
    if o.get("%dynamic").is_some() {
        return Ok("%dynamic".to_string());
    }
    if let Some(v) = o.get("%ref") {
        let r = obj(v, "a %ref")?;
        return Ok(format!(
            "(%ref {} {})",
            as_str(get(r, "ns", "a %ref")?, "ref ns")?,
            as_str(get(r, "name", "a %ref")?, "ref name")?
        ));
    }
    for ax in ["%axis", "%uaxis", "%laxis"] {
        if let Some(v) = o.get(ax) {
            return Ok(format!("({ax} {})", as_str(v, "an axis name")?));
        }
    }
    for nk in ["%kwarg", "%field", "%assign"] {
        if let Some(v) = o.get(nk) {
            let e = obj(v, nk)?;
            return Ok(format!(
                "({nk} {} {})",
                as_str(get(e, "name", nk)?, "a named-entry name")?,
                emit(get(e, "value", nk)?, depth + 1)?
            ));
        }
    }
    if let Some(v) = o.get("%meta") {
        return emit_meta(obj(v, "%meta")?, depth);
    }
    if let Some(v) = o.get("%inputs") {
        return emit_inputs(obj(v, "%inputs")?, depth);
    }
    Err(Error::new("unrecognized JSON node object"))
}

fn emit_meta(m: &Map<String, Value>, depth: usize) -> Result<String> {
    Ok(format!(
        "(%meta {} {} {})",
        emit(get(m, "type", "%meta")?, depth + 1)?,
        as_str(get(m, "phase", "%meta")?, "a phase")?,
        emit(get(m, "valueset", "%meta")?, depth + 1)?
    ))
}

fn emit_inputs(i: &Map<String, Value>, depth: usize) -> Result<String> {
    let origin = as_str(get(i, "origin", "%inputs")?, "an origin tag")?;
    let list = get(i, "list", "%inputs")?;
    let list_text = match list {
        Value::String(tag) => tag.clone(), // "%deferred"
        Value::Array(entries) => {
            // SPEC §9: a reified input list MUST NOT be empty (callables cannot
            // be nullary). Enforce it here rather than deferring to the reader.
            if entries.is_empty() {
                return Err(Error::new(
                    "a reified input list cannot be empty (callables cannot be nullary)",
                ));
            }
            let mut parts = Vec::with_capacity(entries.len());
            for e in entries {
                let pair = e
                    .as_array()
                    .ok_or_else(|| Error::new("an input entry must be a [name, ref] pair"))?;
                if pair.len() != 2 {
                    return Err(Error::new("an input entry must be a [name, ref] pair"));
                }
                // SPEC §9: each entry is `[name, ref]` where `ref` is a `{"%ref"}`.
                if !matches!(pair[1].as_object(), Some(o) if o.contains_key("%ref")) {
                    return Err(Error::new("an input entry's ref must be a %ref object"));
                }
                parts.push(format!(
                    "({} {})",
                    as_str(&pair[0], "an input name")?,
                    emit(&pair[1], depth + 1)?
                ));
            }
            format!("({})", parts.join(" "))
        }
        _ => return Err(Error::new("`%inputs.list` must be a string or array")),
    };
    Ok(format!("{origin} {list_text}"))
}

fn emit_call(arr: &[Value], depth: usize) -> Result<String> {
    if arr.is_empty() {
        return Err(Error::new("a call array cannot be empty"));
    }
    let mut parts: Vec<String> = Vec::with_capacity(arr.len());
    let elements = match arr[0].as_str() {
        // A string first element is the call/structural head.
        Some(head) => {
            parts.push(head.to_string());
            &arr[1..]
        }
        // A non-string first element means a headless, bare parenthesised list
        // (e.g. an `%array` shape `(<dim>…)`): every element is a value.
        None => arr,
    };
    // The %inputs element (reified callables) expands in place to two tokens;
    // every other element emits as a single expression. emit() returns the
    // already-space-joined text either way.
    for elem in elements {
        parts.push(emit(elem, depth + 1)?);
    }
    Ok(format!("({})", parts.join(" ")))
}
