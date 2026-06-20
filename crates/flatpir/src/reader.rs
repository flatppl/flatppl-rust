//! Reading canonical FlatPIR text (spec §11) into a [`Module`].
//!
//! Interprets the generic [`Sexpr`](crate::sexpr) tree as FlatPIR: `(%module …)`
//! with `(%public …)` and `(%bind …)` stanzas, expressions (built-in calls,
//! `(%call (%ref …) …)` user calls, refs, axes, literals), and optional
//! `(%meta (<type> <phase> <valueset>) <expr>)` annotation wrappers that
//! populate the side-tables.
//!
//! References carry only names (`(%ref self x)` → `Symbol`), so forward
//! references resolve for free — no binding-ID fix-up pass.
//!
//! **Diagnostics.** Every form carries a [`Span`], so structural/semantic
//! errors point back at the offending source. The [`err`] / [`err_slice`]
//! helpers turn a `&Sexpr` (or the inner items of a list) into a positioned
//! [`Error`]; the small accessors ([`atom`], [`list`], [`string`]) localize on
//! the form they inspect.

use crate::error::{Error, Result};
use crate::sexpr::{self, Sexpr, SexprKind, Span};
use flatppl_core::{
    Axis, Binding, Call, CallHead, Dim, Doc, Inputs, Markup, Mass, Module, NamedArg, NamedKind,
    Node, NodeId, Phase, Ref, RefNs, Scalar, ScalarType, Symbol, Type, ValueSet, Variance,
};

/// Parse canonical FlatPIR text into a [`Module`].
pub fn read(input: &str) -> Result<Module> {
    let forms = sexpr::parse_top(input)?;
    if forms.len() != 1 {
        let message = format!(
            "expected exactly one top-level (%module …) form, found {}",
            forms.len()
        );
        // Point at the first extra form when there is one; an empty file has no
        // form to anchor on.
        return Err(match forms.get(1) {
            Some(extra) => err(extra, message),
            None => Error::new(message),
        });
    }
    read_module(&forms[0])
}

// ---- positioned-error helpers ----

/// Build an error anchored at `span`.
fn err_at(span: Span, message: impl Into<String>) -> Error {
    Error::at_span(span.line, (span.start, span.end), message)
}

/// Build an error pointing at `form`.
fn err(form: &Sexpr, message: impl Into<String>) -> Error {
    err_at(form.span, message)
}

/// Build an error spanning a list's inner items `[first ..= last]` (the content
/// between the enclosing parens). Falls back to unpositioned for an empty slice.
fn err_slice(items: &[Sexpr], message: impl Into<String>) -> Error {
    match (items.first(), items.last()) {
        (Some(first), Some(last)) => {
            Error::at_span(first.span.line, (first.span.start, last.span.end), message)
        }
        _ => Error::new(message),
    }
}

fn read_module(form: &Sexpr) -> Result<Module> {
    let items = list(form)?;
    if items.first().and_then(Sexpr::as_atom) != Some("%module") {
        return Err(err(form, "expected a (%module …) form"));
    }

    let mut module = Module::new();
    let mut pending: Vec<PendingBind> = Vec::new();
    // Public names keep their span so an unmatched `(%public …)` entry localizes.
    let mut declared_public: Option<Vec<(String, Span)>> = None;

    for elem in &items[1..] {
        let inner = list(elem).map_err(|_| {
            err(
                elem,
                "module elements must be (%public …) or (%bind …) forms",
            )
        })?;
        let head = inner
            .first()
            .and_then(Sexpr::as_atom)
            .ok_or_else(|| err(elem, "module element has no head symbol"))?;
        match head {
            "%public" => {
                let mut names = Vec::new();
                for n in &inner[1..] {
                    names.push((atom(n)?.to_string(), n.span));
                }
                declared_public = Some(names);
            }
            "%bind" => pending.push(parse_bind(&mut module, inner)?),
            other => return Err(err(elem, format!("unexpected module element `{other}`"))),
        }
    }

    // Decide each binding's public flag from the explicit `(%public …)` list when
    // present — it is the authored interface / rewrite root set (spec §11) and may
    // legitimately omit non-underscore bindings such as `load_module` references.
    // Absent a `(%public …)` form, fall back to the name convention (public iff
    // not underscore-prefixed, spec §04).
    let public_set: Option<std::collections::HashSet<&str>> = declared_public
        .as_ref()
        .map(|names| names.iter().map(|(name, _)| name.as_str()).collect());

    if let Some(names) = &declared_public {
        for (name, span) in names {
            if !pending.iter().any(|p| p.name == *name) {
                return Err(err_at(
                    *span,
                    format!("(%public …) lists `{name}`, which is not a binding"),
                ));
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
        return Err(err_slice(items, "(%bind …) needs a name and an expression"));
    }
    let name = atom(&items[1])?.to_string();

    // An optional trailing `(%doc …)` follows the single RHS expression.
    let (expr_items, doc) = match items.last() {
        Some(last) if is_doc_form(last) => (&items[2..items.len() - 1], Some(read_doc(last)?)),
        _ => (&items[2..], None),
    };
    if expr_items.len() != 1 {
        return Err(err_slice(
            items,
            format!(
                "(%bind {name} …) expects exactly one expression, found {}",
                expr_items.len()
            ),
        ));
    }

    let rhs = read_expr(module, &expr_items[0])?;
    Ok(PendingBind { name, rhs, doc })
}

fn read_expr(module: &mut Module, form: &Sexpr) -> Result<NodeId> {
    match &form.kind {
        SexprKind::Atom(s) => read_atom_expr(module, s, form.span),
        SexprKind::Str(s) => Ok(module.alloc(Node::Lit(Scalar::Str(s.clone().into_boxed_str())))),
        SexprKind::List(items) => read_list_expr(module, items, form.span),
    }
}

fn read_atom_expr(module: &mut Module, s: &str, span: Span) -> Result<NodeId> {
    let node = match s {
        "true" => Node::Lit(Scalar::Bool(true)),
        "false" => Node::Lit(Scalar::Bool(false)),
        "_" => Node::Hole,
        _ => {
            if let Some(num) = classify_number(s) {
                Node::Lit(num)
            } else if s.starts_with('%') {
                return Err(err_at(
                    span,
                    format!("unexpected keyword `{s}` in expression position"),
                ));
            } else {
                Node::Const(module.intern(s))
            }
        }
    };
    Ok(module.alloc(node))
}

fn read_list_expr(module: &mut Module, items: &[Sexpr], span: Span) -> Result<NodeId> {
    if items.is_empty() {
        return Err(err_at(span, "empty `()` is not an expression"));
    }
    let head = items[0].as_atom().ok_or_else(|| {
        err(
            &items[0],
            format!("call head must be a symbol, found {}", describe(&items[0])),
        )
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
        "%meta" => read_meta_wrapper(module, items, span),
        _ if head.starts_with('%') => Err(err(
            &items[0],
            format!("unexpected `{head}` as a call head"),
        )),
        _ => read_builtin_call(module, head, &items[1..]),
    }
}

fn parse_ref(module: &mut Module, items: &[Sexpr]) -> Result<Ref> {
    if items.len() != 3 {
        return Err(err_slice(
            items,
            "(%ref <namespace> <name>) takes exactly two operands",
        ));
    }
    let ns_text = atom(&items[1])?;
    let name = atom(&items[2])?;
    let ns = match ns_text {
        "self" => RefNs::SelfMod,
        "%local" => RefNs::Local,
        alias if alias.starts_with('%') => {
            return Err(err(
                &items[1],
                format!("invalid reference namespace `{alias}`"),
            ));
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
        return Err(err_slice(items, "an axis form takes exactly one name"));
    }
    let name = module.intern(atom(&items[1])?);
    Ok(module.alloc(Node::Axis(Axis { name, variance })))
}

fn read_user_call(module: &mut Module, items: &[Sexpr]) -> Result<NodeId> {
    if items.len() < 2 {
        return Err(err_slice(items, "(%call …) needs a callable"));
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
    Ok(module.alloc(Node::Call(call)))
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
    Ok(module.alloc(Node::Call(call)))
}

/// Read a `functionof` / `kernelof` form (spec §11 "Reified callables"):
/// `(<op> <output> %specinputs|%autoinputs <input-list>|%deferred)`. A `%meta`
/// annotation wraps the whole form and is handled by [`read_meta_wrapper`].
fn read_reification(module: &mut Module, op: &str, rest: &[Sexpr]) -> Result<NodeId> {
    if rest.len() != 3 {
        return Err(err_slice(
            rest,
            format!(
                "`{op}` takes <output> <origin-tag> <input-list>, found {} operands",
                rest.len()
            ),
        ));
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
            return Err(err(
                &rest[1],
                format!("`{op}` origin tag must be %specinputs or %autoinputs, found `{other}`"),
            ));
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
    Ok(id)
}

/// Parse a input list `((<name> (%ref …)) …)`.
fn parse_input_entries(module: &mut Module, form: &Sexpr) -> Result<Box<[(Symbol, Ref)]>> {
    let items = list(form)?;
    if items.is_empty() {
        return Err(err(
            form,
            "a reification input list cannot be empty (callables cannot be nullary)",
        ));
    }
    items
        .iter()
        .map(|item| {
            let pair = list(item)?;
            if pair.len() != 2 {
                return Err(err(item, "a input entry takes (<name> (%ref …))"));
            }
            let name = module.intern(atom(&pair[0])?);
            let ref_items = list(&pair[1])?;
            if ref_items.first().and_then(Sexpr::as_atom) != Some("%ref") {
                return Err(err(&pair[1], "a input entry's value must be a (%ref …)"));
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
}

/// The three `%meta` slots: type, phase, value set.
type Meta = (Option<Type>, Option<Phase>, Option<ValueSet>);

fn parse_call_tail(module: &mut Module, items: &[Sexpr]) -> Result<CallTail> {
    let mut args = Vec::new();
    let mut named = Vec::new();

    for item in items {
        match tail_kind(item) {
            TailKind::Named(kind) => named.push(parse_named(module, item, kind)?),
            TailKind::Expr => args.push(read_expr(module, item)?),
        }
    }

    Ok(CallTail { args, named })
}

enum TailKind {
    Named(NamedKind),
    Expr,
}

/// Classify a call operand. Only `%kwarg` / `%field` / `%assign` forms are
/// structural; everything else (including `(%ref …)`, `(%axis …)`, `(%call …)`,
/// a `(%meta …)` wrapper, and built-in calls) is a positional expression.
fn tail_kind(item: &Sexpr) -> TailKind {
    if let SexprKind::List(inner) = &item.kind {
        if let Some(head) = inner.first().and_then(Sexpr::as_atom) {
            return match head {
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
        return Err(err(item, "a named entry takes (<keyword> <name> <value>)"));
    }
    let name = module.intern(atom(&items[1])?);
    let value = read_expr(module, &items[2])?;
    Ok(NamedArg { kind, name, value })
}

fn apply_meta(module: &mut Module, id: NodeId, meta: Meta) {
    if let Some(ty) = meta.0 {
        module.set_type(id, ty);
    }
    if let Some(phase) = meta.1 {
        module.set_phase(id, phase);
    }
    if let Some(set) = meta.2 {
        module.set_valueset(id, set);
    }
}

/// `(%meta (<type> <phase> <valueset>) <expr>)` — a transparent annotation
/// wrapper (spec §11): parse the grouped triple, read the inner expression, and
/// attach the annotation to the inner node. The wrapper adds no node and never
/// directly wraps another `%meta`.
fn read_meta_wrapper(module: &mut Module, items: &[Sexpr], span: Span) -> Result<NodeId> {
    if items.len() != 3 {
        return Err(err_at(
            span,
            "(%meta (<type> <phase> <valueset>) <expr>) takes a triple and one expression",
        ));
    }
    let meta = parse_meta_triple(module, &items[1])?;
    if items[2]
        .as_list()
        .and_then(|l| l.first())
        .and_then(Sexpr::as_atom)
        == Some("%meta")
    {
        return Err(err(
            &items[2],
            "a %meta wrapper must not directly wrap another %meta",
        ));
    }
    let inner = read_expr(module, &items[2])?;
    apply_meta(module, inner, meta);
    Ok(inner)
}

/// Parse the grouped `(<type> <phase> <valueset>)` triple of a `%meta` wrapper.
fn parse_meta_triple(module: &mut Module, form: &Sexpr) -> Result<Meta> {
    let items = list(form)?;
    if items.len() != 3 {
        return Err(err(form, "a %meta triple is (<type> <phase> <valueset>)"));
    }
    let ty = read_type(module, &items[0])?;
    let phase = read_phase(&items[1])?;
    let set = read_valueset(module, &items[2])?;
    Ok((ty, phase, set))
}

/// Read a `%meta` value-set slot: a set expression, `%unknown`, or
/// `%deferred` (→ `None`, like the other slots).
fn read_valueset(module: &mut Module, form: &Sexpr) -> Result<Option<ValueSet>> {
    match &form.kind {
        SexprKind::Atom(s) => match s.as_str() {
            "%deferred" => Ok(None),
            "%unknown" => Ok(Some(ValueSet::Unknown)),
            "reals" => Ok(Some(ValueSet::Reals)),
            "posreals" => Ok(Some(ValueSet::PosReals)),
            "nonnegreals" => Ok(Some(ValueSet::NonNegReals)),
            "unitinterval" => Ok(Some(ValueSet::UnitInterval)),
            "integers" => Ok(Some(ValueSet::Integers)),
            "posintegers" => Ok(Some(ValueSet::PosIntegers)),
            "nonnegintegers" => Ok(Some(ValueSet::NonNegIntegers)),
            "booleans" => Ok(Some(ValueSet::Booleans)),
            "complexes" => Ok(Some(ValueSet::Complexes)),
            "rngstates" => Ok(Some(ValueSet::RngStates)),
            "anything" => Ok(Some(ValueSet::Anything)),
            other => Err(err(form, format!("unknown value set `{other}`"))),
        },
        SexprKind::Str(_) => Err(err(form, "a string is not a value set")),
        SexprKind::List(items) => {
            let head = atom(&items[0])?;
            match head {
                "stdsimplex" => {
                    if items.len() != 2 {
                        return Err(err(form, "(stdsimplex <n>) takes one size"));
                    }
                    Ok(Some(ValueSet::StdSimplex(parse_dim(&items[1])?)))
                }
                "interval" => {
                    if items.len() != 3 {
                        return Err(err(form, "(interval <lo> <hi>) takes two bounds"));
                    }
                    Ok(Some(ValueSet::Interval(
                        parse_bound(&items[1])?,
                        parse_bound(&items[2])?,
                    )))
                }
                "cartpow" => {
                    if items.len() != 3 {
                        return Err(err(form, "(cartpow <set> <n>) takes a set and a size"));
                    }
                    let elem = read_valueset(module, &items[1])?
                        .ok_or_else(|| err(&items[1], "`%deferred` is not allowed inside a set"))?;
                    Ok(Some(ValueSet::CartPow(
                        Box::new(elem),
                        parse_dim(&items[2])?,
                    )))
                }
                "cartprod" => {
                    let parts: Result<Vec<ValueSet>> = items[1..]
                        .iter()
                        .map(|it| {
                            read_valueset(module, it)?
                                .ok_or_else(|| err(it, "`%deferred` is not allowed inside a set"))
                        })
                        .collect();
                    Ok(Some(ValueSet::CartProd(parts?.into())))
                }
                "record" => {
                    let mut fields = Vec::new();
                    for pair in &items[1..] {
                        let SexprKind::List(kv) = &pair.kind else {
                            return Err(err(pair, "a record value-set field is (<name> <set>)"));
                        };
                        if kv.len() != 2 {
                            return Err(err(pair, "a record value-set field is (<name> <set>)"));
                        }
                        let name = module.intern(atom(&kv[0])?);
                        let set = read_valueset(module, &kv[1])?.ok_or_else(|| {
                            err(&kv[1], "`%deferred` is not allowed inside a set")
                        })?;
                        fields.push((name, set));
                    }
                    Ok(Some(ValueSet::RecordSet(fields.into())))
                }
                other => Err(err(&items[0], format!("unknown value-set form `{other}`"))),
            }
        }
    }
}

/// An interval bound: a numeric literal, `inf`, or `-inf`.
fn parse_bound(form: &Sexpr) -> Result<f64> {
    let text = atom(form)?;
    text.parse::<f64>()
        .map_err(|_| err(form, format!("invalid interval bound `{text}`")))
}

// ---- types ----

/// Read a `%meta` type slot. `%deferred` (and, by extension, an absent slot)
/// maps to `None`; every other form yields a concrete [`Type`].
fn read_type(module: &mut Module, form: &Sexpr) -> Result<Option<Type>> {
    match &form.kind {
        SexprKind::Atom(s) => match s.as_str() {
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
                Err(err(form, format!("unknown type `{other}`")))
            }
        },
        SexprKind::Str(_) => Err(err(form, "a string is not a type")),
        SexprKind::List(items) => read_type_list(module, items).map(Some),
    }
}

/// A type position that must be concrete (not `%deferred`); used for nested
/// component types (array element, record fields, measure domain, …).
fn read_type_required(module: &mut Module, form: &Sexpr) -> Result<Type> {
    read_type(module, form)?.ok_or_else(|| err(form, "`%deferred` is not allowed in a nested type"))
}

fn read_type_list(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    let head = atom(&items[0])?;
    match head {
        "%failed" => {
            if items.len() != 2 {
                return Err(err_slice(items, "(%failed \"reason\") takes one string"));
            }
            Ok(Type::Failed(string(&items[1])?.into()))
        }
        "%scalar" => {
            if items.len() != 2 {
                return Err(err_slice(items, "(%scalar <kind>) takes one kind"));
            }
            Ok(Type::Scalar(parse_scalar_type(&items[1])?))
        }
        "%array" => read_array_type(module, items),
        "%tvector" => {
            if items.len() != 3 {
                return Err(err_slice(
                    items,
                    "(%tvector <len> <elem>) takes two operands",
                ));
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
            if items.len() != 3 {
                return Err(err_slice(
                    items,
                    "(%measure (%domain <type>) (%mass <mass>)) takes a domain and a mass",
                ));
            }
            let domain = parse_wrapped_type(module, &items[1], "%domain")?;
            let mass = parse_wrapped_mass(&items[2])?;
            Ok(Type::Measure {
                domain: Box::new(domain),
                mass,
            })
        }
        "%kernel" => {
            if items.len() != 3 {
                return Err(err_slice(
                    items,
                    "(%kernel (%inputs …) (%mass <mass>)) takes inputs and a mass",
                ));
            }
            Ok(Type::Kernel {
                inputs: parse_inputs(module, &items[..2])?.into(),
                mass: parse_wrapped_mass(&items[2])?,
            })
        }
        "%function" => Ok(Type::Function {
            inputs: parse_inputs(module, items)?.into(),
        }),
        "%likelihood" => read_likelihood_type(module, items),
        other => Err(err(&items[0], format!("unknown type form `{other}`"))),
    }
}

fn read_array_type(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    if items.len() != 4 {
        return Err(err_slice(
            items,
            "(%array <ndims> (<shape>) <elem>) takes three operands",
        ));
    }
    let ndims: usize = atom(&items[1])?
        .parse()
        .map_err(|_| err(&items[1], "(%array …) ndims must be a non-negative integer"))?;
    let shape_items = list(&items[2])?;
    let shape: Vec<Dim> = shape_items.iter().map(parse_dim).collect::<Result<_>>()?;
    if shape.len() != ndims {
        return Err(err_slice(
            items,
            format!(
                "(%array …) ndims {ndims} disagrees with shape length {}",
                shape.len()
            ),
        ));
    }
    let elem = read_type_required(module, &items[3])?;
    Ok(Type::Array {
        shape: shape.into(),
        elem: Box::new(elem),
    })
}

fn read_table_type(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    if items.len() != 3 {
        return Err(err_slice(
            items,
            "(%table (%columns …) (%nrows N)) takes two operands",
        ));
    }
    let cols_form = list(&items[1])?;
    if cols_form.first().and_then(Sexpr::as_atom) != Some("%columns") {
        return Err(err(
            &items[1],
            "(%table …) first operand must be (%columns …)",
        ));
    }
    let columns = parse_named_types(module, &cols_form[1..])?;
    let nrows_form = list(&items[2])?;
    if nrows_form.first().and_then(Sexpr::as_atom) != Some("%nrows") || nrows_form.len() != 2 {
        return Err(err(
            &items[2],
            "(%table …) second operand must be (%nrows N)",
        ));
    }
    Ok(Type::Table {
        columns: columns.into(),
        nrows: parse_dim(&nrows_form[1])?,
    })
}

fn read_likelihood_type(module: &mut Module, items: &[Sexpr]) -> Result<Type> {
    if items.len() != 3 {
        return Err(err_slice(
            items,
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
        return Err(err_slice(items, "expected a single (%inputs …) list"));
    }
    let inputs = list(&items[1])?;
    if inputs.first().and_then(Sexpr::as_atom) != Some("%inputs") {
        return Err(err(&items[1], "expected (%inputs name …)"));
    }
    inputs[1..]
        .iter()
        .map(|n| Ok(module.intern(atom(n)?)))
        .collect()
}

/// Read a `(%mass <mass>)` sub-form (spec §11 "Total-mass classes").
fn parse_wrapped_mass(form: &Sexpr) -> Result<Mass> {
    let items = list(form)?;
    if items.first().and_then(Sexpr::as_atom) != Some("%mass") || items.len() != 2 {
        return Err(err(form, "expected (%mass <mass>)"));
    }
    match atom(&items[1])? {
        "%deferred" => Ok(Mass::Deferred),
        "%null" => Ok(Mass::Null),
        "%normalized" => Ok(Mass::Normalized),
        "%finite" => Ok(Mass::Finite),
        "%locallyfinite" => Ok(Mass::LocallyFinite),
        "%unknown" => Ok(Mass::Unknown),
        other => Err(err(&items[1], format!("unknown mass class `{other}`"))),
    }
}

/// Read a single-type wrapper like `(%domain <type>)` or `(%obstype <type>)`.
fn parse_wrapped_type(module: &mut Module, form: &Sexpr, keyword: &str) -> Result<Type> {
    let items = list(form)?;
    if items.first().and_then(Sexpr::as_atom) != Some(keyword) || items.len() != 2 {
        return Err(err(form, format!("expected ({keyword} <type>)")));
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
            return Err(err(it, "expected (<name> <type>)"));
        }
        let name = module.intern(atom(&pair[0])?);
        out.push((name, read_type_required(module, &pair[1])?));
    }
    Ok(out)
}

fn parse_scalar_type(form: &Sexpr) -> Result<ScalarType> {
    match atom(form)? {
        "real" => Ok(ScalarType::Real),
        "integer" => Ok(ScalarType::Integer),
        "boolean" => Ok(ScalarType::Boolean),
        "complex" => Ok(ScalarType::Complex),
        other => Err(err(form, format!("unknown scalar type `{other}`"))),
    }
}

fn parse_dim(form: &Sexpr) -> Result<Dim> {
    let a = atom(form)?;
    if a == "%dynamic" {
        Ok(Dim::Dynamic)
    } else {
        a.parse::<u32>()
            .map(Dim::Static)
            .map_err(|_| err(form, format!("invalid dimension `{a}`")))
    }
}

fn read_phase(form: &Sexpr) -> Result<Option<Phase>> {
    match &form.kind {
        SexprKind::Atom(s) => match s.as_str() {
            "%deferred" => Ok(None),
            "%fixed" => Ok(Some(Phase::Fixed)),
            "%parameterized" => Ok(Some(Phase::Parameterized)),
            "%stochastic" => Ok(Some(Phase::Stochastic)),
            other => Err(err(form, format!("unknown phase `{other}`"))),
        },
        SexprKind::List(items) if items.first().and_then(Sexpr::as_atom) == Some("%failed") => {
            // core's Phase has no failure variant; phase inference rarely fails
            // (it is an ancestor walk) and any %failed module is ill-formed.
            Err(err(
                form,
                "(%failed …) in the phase slot is not yet supported",
            ))
        }
        _ => Err(err(form, format!("invalid phase: {}", describe(form)))),
    }
}

// ---- docs ----

fn is_doc_form(form: &Sexpr) -> bool {
    matches!(&form.kind, SexprKind::List(items) if items.first().and_then(Sexpr::as_atom) == Some("%doc"))
}

fn read_doc(form: &Sexpr) -> Result<Doc> {
    let items = list(form)?;
    if items.len() < 2 {
        return Err(err(form, "(%doc <markup> <line>…) needs a markup tag"));
    }
    let markup = match atom(&items[1])? {
        "md" => Markup::Md,
        "typ" => Markup::Typ,
        other => return Err(err(&items[1], format!("unknown doc markup `{other}`"))),
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
pub(crate) fn classify_number(s: &str) -> Option<Scalar> {
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

// ---- small Sexpr accessors with descriptive, positioned errors ----

fn atom(form: &Sexpr) -> Result<&str> {
    form.as_atom()
        .ok_or_else(|| err(form, format!("expected a symbol, found {}", describe(form))))
}

fn list(form: &Sexpr) -> Result<&[Sexpr]> {
    form.as_list()
        .ok_or_else(|| err(form, format!("expected a list, found {}", describe(form))))
}

fn string(form: &Sexpr) -> Result<&str> {
    match &form.kind {
        SexprKind::Str(s) => Ok(s),
        _ => Err(err(
            form,
            format!("expected a string, found {}", describe(form)),
        )),
    }
}

fn describe(form: &Sexpr) -> String {
    match &form.kind {
        SexprKind::Atom(a) => format!("`{a}`"),
        SexprKind::Str(_) => "a string".to_string(),
        SexprKind::List(items) => {
            let head = items.first().and_then(Sexpr::as_atom).unwrap_or("…");
            format!("`({head} …)`")
        }
    }
}
