//! Reading canonical FlatPIR text (spec §11) into a [`Module`].
//!
//! Interprets the generic [`Sexpr`](crate::sexpr) tree as FlatPIR: `(%module …)`
//! with `(%public …)` and `(%bind …)` stanzas, expressions (built-in calls,
//! `(%call (%ref …) …)` user calls, refs, axes, literals), and optional
//! `(%meta <type> <phase>)` annotations that populate the side-tables.
//!
//! References carry only names (`(%ref self x)` → `Symbol`), so forward
//! references resolve for free — no binding-ID fix-up pass.

use crate::error::{Error, Result};
use crate::sexpr::{self, Sexpr};
use flatppl_core::{
    Axis, Binding, Call, CallHead, Dim, Doc, Inputs, Markup, Module, NamedArg, NamedKind, Node,
    NodeId, Phase, Ref, RefNs, Scalar, ScalarType, Symbol, Type, Variance,
};

/// Parse canonical FlatPIR text into a [`Module`].
pub fn read(input: &str) -> Result<Module> {
    let forms = sexpr::parse_top(input)?;
    if forms.len() != 1 {
        return Err(Error::new(format!(
            "expected exactly one top-level (%module …) form, found {}",
            forms.len()
        )));
    }
    read_module(&forms[0])
}

fn read_module(form: &Sexpr) -> Result<Module> {
    let items = list(form)?;
    if items.first().and_then(Sexpr::as_atom) != Some("%module") {
        return Err(Error::new("expected a (%module …) form"));
    }

    let mut module = Module::new();
    let mut pending: Vec<PendingBind> = Vec::new();
    let mut declared_public: Option<Vec<String>> = None;

    for elem in &items[1..] {
        let inner = list(elem)
            .map_err(|_| Error::new("module elements must be (%public …) or (%bind …) forms"))?;
        let head = inner
            .first()
            .and_then(Sexpr::as_atom)
            .ok_or_else(|| Error::new("module element has no head symbol"))?;
        match head {
            "%public" => {
                let mut names = Vec::new();
                for n in &inner[1..] {
                    names.push(atom(n)?.to_string());
                }
                declared_public = Some(names);
            }
            "%bind" => pending.push(parse_bind(&mut module, inner)?),
            other => return Err(Error::new(format!("unexpected module element `{other}`"))),
        }
    }

    // Decide each binding's public flag from the explicit `(%public …)` list when
    // present — it is the authored interface / rewrite root set (spec §11) and may
    // legitimately omit non-underscore bindings such as `load_module` references.
    // Absent a `(%public …)` form, fall back to the name convention (public iff
    // not underscore-prefixed, spec §04).
    let public_set: Option<std::collections::HashSet<&str>> = declared_public
        .as_ref()
        .map(|names| names.iter().map(String::as_str).collect());

    if let Some(set) = &public_set {
        for name in set {
            if !pending.iter().any(|p| p.name == *name) {
                return Err(Error::new(format!(
                    "(%public …) lists `{name}`, which is not a binding"
                )));
            }
        }
    }

    for p in pending {
        let public = match &public_set {
            Some(set) => set.contains(p.name.as_str()),
            None => !p.name.starts_with('_'),
        };
        let name = module.intern(&p.name);
        module.add_binding(Binding {
            name,
            rhs: p.rhs,
            doc: p.doc,
            public,
            synthetic: false,
        });
    }

    Ok(module)
}

/// A binding read from a `(%bind …)` stanza, held until the `(%public …)` list
/// is known (it may follow the binds) so the public flag can be set correctly.
struct PendingBind {
    name: String,
    rhs: NodeId,
    doc: Option<Doc>,
}

fn parse_bind(module: &mut Module, items: &[Sexpr]) -> Result<PendingBind> {
    if items.len() < 3 {
        return Err(Error::new("(%bind …) needs a name and an expression"));
    }
    let name = atom(&items[1])?.to_string();

    // An optional trailing `(%doc …)` follows the single RHS expression.
    let (expr_items, doc) = match items.last() {
        Some(last) if is_doc_form(last) => (&items[2..items.len() - 1], Some(read_doc(last)?)),
        _ => (&items[2..], None),
    };
    if expr_items.len() != 1 {
        return Err(Error::new(format!(
            "(%bind {name} …) expects exactly one expression, found {}",
            expr_items.len()
        )));
    }

    let rhs = read_expr(module, &expr_items[0])?;
    Ok(PendingBind { name, rhs, doc })
}

fn read_expr(module: &mut Module, form: &Sexpr) -> Result<NodeId> {
    match form {
        Sexpr::Atom(s) => read_atom_expr(module, s),
        Sexpr::Str(s) => Ok(module.alloc(Node::Lit(Scalar::Str(s.clone().into_boxed_str())))),
        Sexpr::List(items) => read_list_expr(module, items),
    }
}

fn read_atom_expr(module: &mut Module, s: &str) -> Result<NodeId> {
    let node = match s {
        "true" => Node::Lit(Scalar::Bool(true)),
        "false" => Node::Lit(Scalar::Bool(false)),
        "_" => Node::Hole,
        _ => {
            if let Some(num) = classify_number(s) {
                Node::Lit(num)
            } else if s.starts_with('%') {
                return Err(Error::new(format!(
                    "unexpected keyword `{s}` in expression position"
                )));
            } else {
                Node::Const(module.intern(s))
            }
        }
    };
    Ok(module.alloc(node))
}

fn read_list_expr(module: &mut Module, items: &[Sexpr]) -> Result<NodeId> {
    if items.is_empty() {
        return Err(Error::new("empty `()` is not an expression"));
    }
    let head = items[0].as_atom().ok_or_else(|| {
        Error::new(format!(
            "call head must be a symbol, found {}",
            describe(&items[0])
        ))
    })?;
    match head {
        "%ref" => {
            let r = parse_ref(module, items)?;
            Ok(module.alloc(Node::Ref(r)))
        }
        "%axis" => read_axis_node(module, items, None),
        "%uaxis" => read_axis_node(module, items, Some(Variance::Upper)),
        "%laxis" => read_axis_node(module, items, Some(Variance::Lower)),
        "%call" => read_user_call(module, items),
        _ if head.starts_with('%') => {
            Err(Error::new(format!("unexpected `{head}` as a call head")))
        }
        _ => read_builtin_call(module, head, &items[1..]),
    }
}

fn parse_ref(module: &mut Module, items: &[Sexpr]) -> Result<Ref> {
    if items.len() != 3 {
        return Err(Error::new(
            "(%ref <namespace> <name>) takes exactly two operands",
        ));
    }
    let ns_text = atom(&items[1])?;
    let name = atom(&items[2])?;
    let ns = match ns_text {
        "self" => RefNs::SelfMod,
        "%local" => RefNs::Local,
        alias if alias.starts_with('%') => {
            return Err(Error::new(format!("invalid reference namespace `{alias}`")));
        }
        alias => RefNs::Module(module.intern(alias)),
    };
    Ok(Ref {
        ns,
        name: module.intern(name),
    })
}

fn read_axis_node(
    module: &mut Module,
    items: &[Sexpr],
    variance: Option<Variance>,
) -> Result<NodeId> {
    if items.len() != 2 {
        return Err(Error::new("an axis form takes exactly one name"));
    }
    let name = module.intern(atom(&items[1])?);
    Ok(module.alloc(Node::Axis(Axis { name, variance })))
}

fn read_user_call(module: &mut Module, items: &[Sexpr]) -> Result<NodeId> {
    if items.len() < 2 {
        return Err(Error::new("(%call …) needs a callable"));
    }
    // The callee is an expression that must evaluate to a user-defined
    // callable — a `(%ref …)` in the common case, or an inline callable
    // expression such as a reification (spec §11). The "must evaluate to a
    // callable" part is a typing condition, not enforced here.
    let callee = read_expr(module, &items[1])?;
    let tail = parse_call_tail(module, &items[2..])?;
    let call = Call {
        head: CallHead::User(callee),
        args: tail.args.into(),
        named: tail.named.into(),
        inputs: None,
    };
    let id = module.alloc(Node::Call(call));
    apply_meta(module, id, tail.meta);
    Ok(id)
}

fn read_builtin_call(module: &mut Module, op: &str, rest: &[Sexpr]) -> Result<NodeId> {
    if op == "functionof" || op == "kernelof" {
        return read_reification(module, op, rest);
    }
    let tail = parse_call_tail(module, rest)?;
    let head = CallHead::Builtin(module.intern(op));
    let call = Call {
        head,
        args: tail.args.into(),
        named: tail.named.into(),
        inputs: None,
    };
    let id = module.alloc(Node::Call(call));
    apply_meta(module, id, tail.meta);
    Ok(id)
}

/// Read a `functionof` / `kernelof` form (spec §11 "Reified callables"):
/// `(<op> [%meta] <output> %specinputs|%autoinputs <input-list>|%deferred)`.
fn read_reification(module: &mut Module, op: &str, rest: &[Sexpr]) -> Result<NodeId> {
    // Optional `(%meta …)` immediately after the head.
    let (meta, rest) = match rest.first() {
        Some(Sexpr::List(inner)) if inner.first().and_then(Sexpr::as_atom) == Some("%meta") => {
            (parse_meta(module, &rest[0])?, &rest[1..])
        }
        _ => ((None, None), rest),
    };

    if rest.len() != 3 {
        return Err(Error::new(format!(
            "`{op}` takes <output> <origin-tag> <input-list>, found {} operands",
            rest.len()
        )));
    }
    let output = read_expr(module, &rest[0])?;
    let tag = atom(&rest[1])?;
    let mut inferred: Option<Box<[(Symbol, Ref)]>> = None;
    let inputs = match tag {
        "%specinputs" => Inputs::Spec(parse_input_entries(module, &rest[2])?),
        "%autoinputs" => {
            if rest[2].as_atom() == Some("%deferred") {
                Inputs::Auto
            } else {
                inferred = Some(parse_input_entries(module, &rest[2])?);
                Inputs::Auto
            }
        }
        other => {
            return Err(Error::new(format!(
                "`{op}` origin tag must be %specinputs or %autoinputs, found `{other}`"
            )));
        }
    };

    let head = CallHead::Builtin(module.intern(op));
    let id = module.alloc(Node::Call(Call {
        head,
        args: Box::new([output]),
        named: Box::new([]),
        inputs: Some(inputs),
    }));
    if let Some(entries) = inferred {
        module.set_auto_inputs(id, entries);
    }
    apply_meta(module, id, meta);
    Ok(id)
}

/// Parse a input list `((<name> (%ref …)) …)`.
fn parse_input_entries(module: &mut Module, form: &Sexpr) -> Result<Box<[(Symbol, Ref)]>> {
    let items = list(form)?;
    if items.is_empty() {
        return Err(Error::new(
            "a reification input list cannot be empty (callables cannot be nullary)",
        ));
    }
    items
        .iter()
        .map(|item| {
            let pair = list(item)?;
            if pair.len() != 2 {
                return Err(Error::new("a input entry takes (<name> (%ref …))"));
            }
            let name = module.intern(atom(&pair[0])?);
            let ref_items = list(&pair[1])?;
            if ref_items.first().and_then(Sexpr::as_atom) != Some("%ref") {
                return Err(Error::new("a input entry's value must be a (%ref …)"));
            }
            let r = parse_ref(module, ref_items)?;
            Ok((name, r))
        })
        .collect()
}

/// The structural pieces of a call's operand list, separated from positional args.
struct CallTail {
    args: Vec<NodeId>,
    named: Vec<NamedArg>,
    meta: (Option<Type>, Option<Phase>),
}

fn parse_call_tail(module: &mut Module, items: &[Sexpr]) -> Result<CallTail> {
    let mut args = Vec::new();
    let mut named = Vec::new();
    let mut meta = (None, None);

    for item in items {
        match tail_kind(item) {
            TailKind::Meta => meta = parse_meta(module, item)?,
            TailKind::Named(kind) => named.push(parse_named(module, item, kind)?),
            TailKind::Expr => args.push(read_expr(module, item)?),
        }
    }

    Ok(CallTail { args, named, meta })
}

enum TailKind {
    Meta,
    Named(NamedKind),
    Expr,
}

/// Classify a call operand. Only `%meta` / `%kwarg` / `%field` / `%assign`
/// forms are structural; everything else (including `(%ref …)`, `(%axis …)`,
/// `(%call …)`, and built-in calls) is a positional expression.
fn tail_kind(item: &Sexpr) -> TailKind {
    if let Sexpr::List(inner) = item {
        if let Some(head) = inner.first().and_then(Sexpr::as_atom) {
            return match head {
                "%meta" => TailKind::Meta,
                "%kwarg" => TailKind::Named(NamedKind::Kwarg),
                "%field" => TailKind::Named(NamedKind::Field),
                "%assign" => TailKind::Named(NamedKind::Assign),
                _ => TailKind::Expr,
            };
        }
    }
    TailKind::Expr
}

fn parse_named(module: &mut Module, item: &Sexpr, kind: NamedKind) -> Result<NamedArg> {
    let items = list(item)?;
    if items.len() != 3 {
        return Err(Error::new("a named entry takes (<keyword> <name> <value>)"));
    }
    let name = module.intern(atom(&items[1])?);
    let value = read_expr(module, &items[2])?;
    Ok(NamedArg { kind, name, value })
}

fn apply_meta(module: &mut Module, id: NodeId, meta: (Option<Type>, Option<Phase>)) {
    if let Some(ty) = meta.0 {
        module.set_type(id, ty);
    }
    if let Some(phase) = meta.1 {
        module.set_phase(id, phase);
    }
}

fn parse_meta(module: &mut Module, item: &Sexpr) -> Result<(Option<Type>, Option<Phase>)> {
    let items = list(item)?;
    if items.len() != 3 {
        return Err(Error::new("(%meta <type> <phase>) takes two slots"));
    }
    let ty = read_type(module, &items[1])?;
    let phase = read_phase(&items[2])?;
    Ok((ty, phase))
}

// ---- types ----

/// Read a `%meta` type slot. `%deferred` (and, by extension, an absent slot)
/// maps to `None`; every other form yields a concrete [`Type`].
fn read_type(module: &mut Module, form: &Sexpr) -> Result<Option<Type>> {
    match form {
        Sexpr::Atom(s) => match s.as_str() {
            "%deferred" => Ok(None),
            "%any" => Ok(Some(Type::Any)),
            "%module" => Ok(Some(Type::Module)),
            "%rngstate" => Ok(Some(Type::RngState)),
            other => {
                if let Some(n) = other.strip_prefix("%var") {
                    if let Ok(v) = n.parse::<u32>() {
                        return Ok(Some(Type::Var(v)));
                    }
                }
                Err(Error::new(format!("unknown type `{other}`")))
            }
        },
        Sexpr::Str(_) => Err(Error::new("a string is not a type")),
        Sexpr::List(items) => read_type_list(module, items).map(Some),
    }
}

/// A type position that must be concrete (not `%deferred`); used for nested
/// component types (array element, record fields, measure domain, …).
fn read_type_required(module: &mut Module, form: &Sexpr) -> Result<Type> {
    read_type(module, form)?
        .ok_or_else(|| Error::new("`%deferred` is not allowed in a nested type"))
}

fn read_type_list(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    let head = atom(&items[0])?;
    match head {
        "%failed" => {
            if items.len() != 2 {
                return Err(Error::new("(%failed \"reason\") takes one string"));
            }
            Ok(Type::Failed(string(&items[1])?.into()))
        }
        "%scalar" => {
            if items.len() != 2 {
                return Err(Error::new("(%scalar <kind>) takes one kind"));
            }
            Ok(Type::Scalar(parse_scalar_type(atom(&items[1])?)?))
        }
        "%array" => read_array_type(module, items),
        "%tvector" => {
            if items.len() != 3 {
                return Err(Error::new("(%tvector <len> <elem>) takes two operands"));
            }
            Ok(Type::TVector {
                len: parse_dim(&items[1])?,
                elem: Box::new(read_type_required(module, &items[2])?),
            })
        }
        "%record" => Ok(Type::Record(parse_named_types(module, &items[1..])?.into())),
        "%tuple" => {
            let mut elems = Vec::new();
            for it in &items[1..] {
                elems.push(read_type_required(module, it)?);
            }
            Ok(Type::Tuple(elems.into()))
        }
        "%table" => read_table_type(module, items),
        "%measure" => {
            if items.len() != 2 {
                return Err(Error::new("(%measure (%domain <type>)) takes one domain"));
            }
            let domain = parse_wrapped_type(module, &items[1], "%domain")?;
            Ok(Type::Measure {
                domain: Box::new(domain),
            })
        }
        "%kernel" => Ok(Type::Kernel {
            inputs: parse_inputs(module, items)?.into(),
        }),
        "%function" => Ok(Type::Function {
            inputs: parse_inputs(module, items)?.into(),
        }),
        "%likelihood" => read_likelihood_type(module, items),
        other => Err(Error::new(format!("unknown type form `{other}`"))),
    }
}

fn read_array_type(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    if items.len() != 4 {
        return Err(Error::new(
            "(%array <ndims> (<shape>) <elem>) takes three operands",
        ));
    }
    let ndims: usize = atom(&items[1])?
        .parse()
        .map_err(|_| Error::new("(%array …) ndims must be a non-negative integer"))?;
    let shape_items = list(&items[2])?;
    let shape: Vec<Dim> = shape_items.iter().map(parse_dim).collect::<Result<_>>()?;
    if shape.len() != ndims {
        return Err(Error::new(format!(
            "(%array …) ndims {ndims} disagrees with shape length {}",
            shape.len()
        )));
    }
    let elem = read_type_required(module, &items[3])?;
    Ok(Type::Array {
        shape: shape.into(),
        elem: Box::new(elem),
    })
}

fn read_table_type(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    if items.len() != 3 {
        return Err(Error::new(
            "(%table (%columns …) (%nrows N)) takes two operands",
        ));
    }
    let cols_form = list(&items[1])?;
    if cols_form.first().and_then(Sexpr::as_atom) != Some("%columns") {
        return Err(Error::new("(%table …) first operand must be (%columns …)"));
    }
    let columns = parse_named_types(module, &cols_form[1..])?;
    let nrows_form = list(&items[2])?;
    if nrows_form.first().and_then(Sexpr::as_atom) != Some("%nrows") || nrows_form.len() != 2 {
        return Err(Error::new("(%table …) second operand must be (%nrows N)"));
    }
    Ok(Type::Table {
        columns: columns.into(),
        nrows: parse_dim(&nrows_form[1])?,
    })
}

fn read_likelihood_type(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    if items.len() != 3 {
        return Err(Error::new(
            "(%likelihood (%inputs …) (%obstype <type>)) takes two operands",
        ));
    }
    // Reuse the (%kernel/%function) input reader on the leading `(%inputs …)`.
    let inputs_holder = [items[0].clone(), items[1].clone()];
    let inputs = parse_inputs(module, &inputs_holder)?;
    let obstype = parse_wrapped_type(module, &items[2], "%obstype")?;
    Ok(Type::Likelihood {
        inputs: inputs.into(),
        obstype: Box::new(obstype),
    })
}

/// Read the `(%inputs name …)` of a `%kernel` / `%function` / `%likelihood`.
/// `items` is the whole type form; the inputs list is its second element.
fn parse_inputs(module: &mut Module, items: &[Sexpr]) -> Result<Vec<flatppl_core::Symbol>> {
    if items.len() != 2 {
        return Err(Error::new("expected a single (%inputs …) list"));
    }
    let inputs = list(&items[1])?;
    if inputs.first().and_then(Sexpr::as_atom) != Some("%inputs") {
        return Err(Error::new("expected (%inputs name …)"));
    }
    inputs[1..]
        .iter()
        .map(|n| Ok(module.intern(atom(n)?)))
        .collect()
}

/// Read a single-type wrapper like `(%domain <type>)` or `(%obstype <type>)`.
fn parse_wrapped_type(module: &mut Module, form: &Sexpr, keyword: &str) -> Result<Type> {
    let items = list(form)?;
    if items.first().and_then(Sexpr::as_atom) != Some(keyword) || items.len() != 2 {
        return Err(Error::new(format!("expected ({keyword} <type>)")));
    }
    read_type_required(module, &items[1])
}

fn parse_named_types(
    module: &mut Module,
    items: &[Sexpr],
) -> Result<Vec<(flatppl_core::Symbol, Type)>> {
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let pair = list(it)?;
        if pair.len() != 2 {
            return Err(Error::new("expected (<name> <type>)"));
        }
        let name = module.intern(atom(&pair[0])?);
        out.push((name, read_type_required(module, &pair[1])?));
    }
    Ok(out)
}

fn parse_scalar_type(s: &str) -> Result<ScalarType> {
    match s {
        "real" => Ok(ScalarType::Real),
        "integer" => Ok(ScalarType::Integer),
        "boolean" => Ok(ScalarType::Boolean),
        "complex" => Ok(ScalarType::Complex),
        other => Err(Error::new(format!("unknown scalar type `{other}`"))),
    }
}

fn parse_dim(form: &Sexpr) -> Result<Dim> {
    let a = atom(form)?;
    if a == "%dynamic" {
        Ok(Dim::Dynamic)
    } else {
        a.parse::<u32>()
            .map(Dim::Static)
            .map_err(|_| Error::new(format!("invalid dimension `{a}`")))
    }
}

fn read_phase(form: &Sexpr) -> Result<Option<Phase>> {
    match form {
        Sexpr::Atom(s) => match s.as_str() {
            "%deferred" => Ok(None),
            "%fixed" => Ok(Some(Phase::Fixed)),
            "%parameterized" => Ok(Some(Phase::Parameterized)),
            "%stochastic" => Ok(Some(Phase::Stochastic)),
            other => Err(Error::new(format!("unknown phase `{other}`"))),
        },
        Sexpr::List(items) if items.first().and_then(Sexpr::as_atom) == Some("%failed") => Err(
            // core's Phase has no failure variant; phase inference rarely fails
            // (it is an ancestor walk) and any %failed module is ill-formed.
            Error::new("(%failed …) in the phase slot is not yet supported"),
        ),
        other => Err(Error::new(format!("invalid phase: {}", describe(other)))),
    }
}

// ---- docs ----

fn is_doc_form(form: &Sexpr) -> bool {
    matches!(form, Sexpr::List(items) if items.first().and_then(Sexpr::as_atom) == Some("%doc"))
}

fn read_doc(form: &Sexpr) -> Result<Doc> {
    let items = list(form)?;
    if items.len() < 2 {
        return Err(Error::new("(%doc <markup> <line>…) needs a markup tag"));
    }
    let markup = match atom(&items[1])? {
        "md" => Markup::Md,
        "typ" => Markup::Typ,
        other => return Err(Error::new(format!("unknown doc markup `{other}`"))),
    };
    let mut lines = Vec::new();
    for it in &items[2..] {
        lines.push(string(it)?.to_string().into_boxed_str());
    }
    Ok(Doc {
        markup,
        lines: lines.into(),
    })
}

// ---- numeric atom classification ----

/// Classify a bare atom as an integer or real *by lexical form* (spec §11), or
/// `None` if it is not numeric (and so a symbol / constant). Crucially, this
/// never treats `inf` / `nan` / `pi` as numbers — those are bare constants.
fn classify_number(s: &str) -> Option<Scalar> {
    let bytes = s.as_bytes();
    let first = *bytes.first()?;
    let numeric_start =
        first.is_ascii_digit() || (matches!(first, b'-' | b'+' | b'.') && bytes.len() > 1);
    if !numeric_start || !s.bytes().any(|b| b.is_ascii_digit()) {
        return None;
    }

    let (negative, rest) = match first {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };

    // Hex integer (`0xF7`).
    if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        let digits: String = hex.chars().filter(|&c| c != '_').collect();
        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let v = i64::from_str_radix(&digits, 16).ok()?;
        return Some(Scalar::Int(if negative { -v } else { v }));
    }

    let cleaned: String = s.chars().filter(|&c| c != '_').collect();
    if s.contains(['.', 'e', 'E']) {
        let v: f64 = cleaned.parse().ok()?;
        v.is_finite().then_some(Scalar::Real(v))
    } else {
        cleaned.parse::<i64>().ok().map(Scalar::Int)
    }
}

// ---- small Sexpr accessors with descriptive errors ----

fn atom(form: &Sexpr) -> Result<&str> {
    form.as_atom()
        .ok_or_else(|| Error::new(format!("expected a symbol, found {}", describe(form))))
}

fn list(form: &Sexpr) -> Result<&[Sexpr]> {
    form.as_list()
        .ok_or_else(|| Error::new(format!("expected a list, found {}", describe(form))))
}

fn string(form: &Sexpr) -> Result<&str> {
    match form {
        Sexpr::Str(s) => Ok(s),
        _ => Err(Error::new(format!(
            "expected a string, found {}",
            describe(form)
        ))),
    }
}

fn describe(form: &Sexpr) -> String {
    match form {
        Sexpr::Atom(a) => format!("`{a}`"),
        Sexpr::Str(_) => "a string".to_string(),
        Sexpr::List(items) => {
            let head = items.first().and_then(Sexpr::as_atom).unwrap_or("…");
            format!("`({head} …)`")
        }
    }
}
