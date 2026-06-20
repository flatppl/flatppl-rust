//! Writing a [`Module`] to canonical FlatPIR text (spec §11).
//!
//! The output is *canonical*, not byte-preserving: one `(%bind …)` per stanza,
//! blank-line separated, 2-space indented, public interface first. Each binding's
//! right-hand side renders inline on one line. This is deterministic and
//! idempotent — `write(read(write(m)))  ==  write(m)` — which is the round-trip
//! contract the tests pin (canonical fixpoint, not source fidelity).
//!
//! Annotations are emitted as a transparent `(%meta (<type> <phase> <valueset>)
//! <expr>)` wrapper, only where the side-tables hold one for a node; a bare
//! (pre-inference) module writes no `%meta`.

use flatppl_core::{
    Axis, Call, CallHead, Dim, Doc, Inputs, Markup, Mass, Module, NamedKind, Node, NodeId, Phase,
    Ref, RefNs, Scalar, ScalarType, Symbol, Type, ValueSet, Variance,
};

/// Render `module` as canonical FlatPIR text (no trailing newline).
pub fn write(module: &Module) -> String {
    let mut out = String::from("(%module");

    // Public interface: `(%public name …)`. ALWAYS emitted, even when empty, so
    // re-reading uses the explicit interface and never the name-convention
    // fallback. Omitting an empty `(%public)` is lossy: a module with no public
    // bindings would re-read with its non-underscore bindings flipped to public.
    let publics: Vec<&str> = module
        .public_bindings()
        .map(|(_, b)| module.resolve(b.name))
        .collect();
    out.push_str("\n  (%public");
    for name in &publics {
        out.push(' ');
        out.push_str(name);
    }
    out.push(')');

    // One `(%bind …)` stanza per binding, in source order, blank-line separated.
    for (id, binding) in module.bindings() {
        out.push_str("\n\n  ");
        out.push_str(&render_bind(module, id, binding));
    }

    out.push(')');
    out
}

fn render_bind(
    module: &Module,
    _id: flatppl_core::BindingId,
    binding: &flatppl_core::Binding,
) -> String {
    let name = module.resolve(binding.name);
    let rhs = render_node(module, binding.rhs);
    let mut inner = format!("%bind {name} {rhs}");
    if let Some(doc) = &binding.doc {
        inner.push(' ');
        inner.push_str(&render_doc(doc));
    }
    format!("({inner})")
}

fn render_doc(doc: &Doc) -> String {
    let tag = match doc.markup {
        Markup::Md => "md",
        Markup::Typ => "typ",
    };
    let mut parts = vec![format!("%doc {tag}")];
    for line in doc.lines.iter() {
        parts.push(quote_string(line));
    }
    format!("({})", parts.join(" "))
}

/// Render a node, wrapping it in a `(%meta (<type> <phase> <valueset>) …)`
/// annotation when warranted. `%meta` is a transparent wrapper that *can* go
/// around any expression (spec §11), but serialization is **sparse**: we emit it
/// only around composite expressions (calls, including reifications) and leave
/// atomic leaves — literals, refs, consts, axes — bare, since their annotations
/// are recoverable and the spec annotated example keeps them bare. A bare
/// (pre-inference) node also renders without a wrapper.
fn render_node(module: &Module, id: NodeId) -> String {
    let inner = render_node_inner(module, id);
    if matches!(module.node(id), Node::Call(_)) {
        if let Some(triple) = render_meta(module, id) {
            return format!("(%meta {triple} {inner})");
        }
    }
    inner
}

fn render_node_inner(module: &Module, id: NodeId) -> String {
    match module.node(id) {
        Node::Lit(lit) => render_scalar(lit),
        Node::Const(sym) => module.resolve(*sym).to_string(),
        Node::Hole => "_".to_string(),
        Node::Ref(r) => render_ref(module, r),
        Node::Axis(a) => render_axis(module, a),
        Node::Call(call) => render_call(module, id, call),
    }
}

fn render_scalar(lit: &Scalar) -> String {
    match lit {
        Scalar::Int(n) => n.to_string(),
        Scalar::Real(r) => render_real(*r),
        Scalar::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Scalar::Str(s) => quote_string(s),
    }
}

/// Format a real so it always re-reads as a real, never an integer: the default
/// shortest round-trip repr of e.g. `2.0` is `"2"`, so append `.0` when there is
/// no `.`/`e`/`E` in the output.
pub(crate) fn render_real(r: f64) -> String {
    let s = format!("{r}");
    if s.contains(['.', 'e', 'E']) || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

pub(crate) fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn render_ref(module: &Module, r: &Ref) -> String {
    let name = module.resolve(r.name);
    match r.ns {
        RefNs::SelfMod => format!("(%ref self {name})"),
        RefNs::Local => format!("(%ref %local {name})"),
        RefNs::Module(alias) => format!("(%ref {} {name})", module.resolve(alias)),
    }
}

fn render_axis(module: &Module, a: &Axis) -> String {
    let name = module.resolve(a.name);
    match a.variance {
        None => format!("(%axis {name})"),
        Some(Variance::Upper) => format!("(%uaxis {name})"),
        Some(Variance::Lower) => format!("(%laxis {name})"),
    }
}

fn render_call(module: &Module, id: NodeId, call: &Call) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Head: a bare built-in symbol, or `%call <callable>` for a user callable
    // (the callee is an expression — a `(%ref …)` in the common case, spec §11).
    match &call.head {
        CallHead::Builtin(sym) => parts.push(module.resolve(*sym).to_string()),
        CallHead::User(callee) => {
            parts.push("%call".to_string());
            parts.push(render_node(module, *callee));
        }
    }

    // Positional arguments.
    for &arg in call.args.iter() {
        parts.push(render_node(module, arg));
    }

    // Named entries, in order: `%kwarg` / `%field` / `%assign`.
    for n in call.named.iter() {
        let head = match n.kind {
            NamedKind::Kwarg => "%kwarg",
            NamedKind::Field => "%field",
            NamedKind::Assign => "%assign",
        };
        parts.push(format!(
            "({head} {} {})",
            module.resolve(n.name),
            render_node(module, n.value)
        ));
    }

    // Reified-callable input list (functionof / kernelof): the trailing
    // origin tag + list (spec §11 "Reified callables"). A filled `%autoinputs`
    // list is projected from the inference side-table.
    if let Some(inputs) = &call.inputs {
        match inputs {
            Inputs::Spec(entries) => {
                parts.push("%specinputs".to_string());
                parts.push(render_input_entries(module, entries));
            }
            Inputs::Auto => {
                parts.push("%autoinputs".to_string());
                match module.auto_inputs_of(id) {
                    Some(entries) => parts.push(render_input_entries(module, entries)),
                    None => parts.push("%deferred".to_string()),
                }
            }
        }
    }

    format!("({})", parts.join(" "))
}

fn render_input_entries(module: &Module, entries: &[(Symbol, Ref)]) -> String {
    let inner: Vec<String> = entries
        .iter()
        .map(|(name, r)| format!("({} {})", module.resolve(*name), render_ref(module, r)))
        .collect();
    format!("({})", inner.join(" "))
}

/// The grouped `(<type> <phase> <valueset>)` triple for a node's `%meta`
/// wrapper, or `None` when no slot is annotated (canonical: omit the wrapper).
/// The enclosing `(%meta … <expr>)` is added by [`render_node`].
fn render_meta(module: &Module, id: NodeId) -> Option<String> {
    let ty = module.type_of(id);
    let phase = module.phase_of(id);
    let vset = module.valueset_of(id);
    if ty.is_none() && phase.is_none() && vset.is_none() {
        return None;
    }
    let ty_s = ty.map_or_else(|| "%deferred".to_string(), |t| render_type(module, t));
    let phase_s = phase.map_or("%deferred", render_phase);
    let vset_s = vset.map_or_else(|| "%deferred".to_string(), |v| render_valueset(module, v));
    Some(format!("({ty_s} {phase_s} {vset_s})"))
}

fn render_mass(mass: Mass) -> &'static str {
    match mass {
        Mass::Deferred => "%deferred",
        Mass::Null => "%null",
        Mass::Normalized => "%normalized",
        Mass::Finite => "%finite",
        Mass::LocallyFinite => "%locallyfinite",
        Mass::Unknown => "%unknown",
    }
}

/// A value set renders as its FlatPPL set expression (bare set constants,
/// `(stdsimplex n)`, `(interval lo hi)`, `(cartpow set n)`).
fn render_valueset(module: &Module, set: &ValueSet) -> String {
    match set {
        ValueSet::Deferred => "%deferred".to_string(),
        ValueSet::Unknown => "%unknown".to_string(),
        ValueSet::Reals => "reals".to_string(),
        ValueSet::PosReals => "posreals".to_string(),
        ValueSet::NonNegReals => "nonnegreals".to_string(),
        ValueSet::UnitInterval => "unitinterval".to_string(),
        ValueSet::Integers => "integers".to_string(),
        ValueSet::PosIntegers => "posintegers".to_string(),
        ValueSet::NonNegIntegers => "nonnegintegers".to_string(),
        ValueSet::Booleans => "booleans".to_string(),
        ValueSet::Complexes => "complexes".to_string(),
        ValueSet::RngStates => "rngstates".to_string(),
        ValueSet::Anything => "anything".to_string(),
        ValueSet::StdSimplex(n) => format!("(stdsimplex {})", render_dim(*n)),
        ValueSet::Interval(lo, hi) => {
            format!("(interval {} {})", render_real(*lo), render_real(*hi))
        }
        ValueSet::CartPow(elem, n) => {
            format!(
                "(cartpow {} {})",
                render_valueset(module, elem),
                render_dim(*n)
            )
        }
        ValueSet::CartProd(parts) => {
            let inner: Vec<String> = parts.iter().map(|s| render_valueset(module, s)).collect();
            format!("(cartprod {})", inner.join(" "))
        }
        ValueSet::RecordSet(fields) => {
            let inner: Vec<String> = fields
                .iter()
                .map(|(n, s)| format!("({} {})", module.resolve(*n), render_valueset(module, s)))
                .collect();
            format!("(record {})", inner.join(" "))
        }
    }
}

fn render_phase(phase: Phase) -> &'static str {
    match phase {
        Phase::Fixed => "%fixed",
        Phase::Parameterized => "%parameterized",
        Phase::Stochastic => "%stochastic",
    }
}

fn render_type(module: &Module, ty: &Type) -> String {
    match ty {
        Type::Deferred => "%deferred".to_string(),
        Type::Failed(reason) => format!("(%failed {})", quote_string(reason)),
        Type::Any => "%any".to_string(),
        Type::Scalar(s) => format!("(%scalar {})", scalar_type_name(*s)),
        Type::Array { shape, elem } => format!(
            "(%array {} ({}) {})",
            shape.len(),
            shape
                .iter()
                .map(|d| render_dim(*d))
                .collect::<Vec<_>>()
                .join(" "),
            render_type(module, elem)
        ),
        Type::TVector { len, elem } => {
            format!(
                "(%tvector {} {})",
                render_dim(*len),
                render_type(module, elem)
            )
        }
        Type::Record(fields) => format!("(%record {})", render_named_types(module, fields)),
        Type::Tuple(elems) => format!(
            "(%tuple {})",
            elems
                .iter()
                .map(|t| render_type(module, t))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        Type::Table { columns, nrows } => format!(
            "(%table (%columns {}) (%nrows {}))",
            render_named_types(module, columns),
            render_dim(*nrows)
        ),
        Type::Measure { domain, mass } => format!(
            "(%measure (%domain {}) (%mass {}))",
            render_type(module, domain),
            render_mass(*mass)
        ),
        Type::Kernel { inputs, mass } => format!(
            "(%kernel (%inputs {}) (%mass {}))",
            render_input_names(module, inputs),
            render_mass(*mass)
        ),
        Type::Function { inputs } => {
            format!(
                "(%function (%inputs {}))",
                render_input_names(module, inputs)
            )
        }
        Type::Likelihood { inputs, obstype } => format!(
            "(%likelihood (%inputs {}) (%obstype {}))",
            render_input_names(module, inputs),
            render_type(module, obstype)
        ),
        // `%rngstate` and `%var<n>` have no spec §11 category yet: rngstate is a
        // value type the engine must carry (open spec question, see TODO), and
        // type variables are transient inference state that should not normally
        // reach serialization. Both round-trip through the reader.
        Type::RngState => "%rngstate".to_string(),
        Type::Module => "%module".to_string(),
        Type::Var(n) => format!("%var{n}"),
    }
}

fn render_named_types(module: &Module, fields: &[(flatppl_core::Symbol, Type)]) -> String {
    fields
        .iter()
        .map(|(name, ty)| format!("({} {})", module.resolve(*name), render_type(module, ty)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_input_names(module: &Module, inputs: &[flatppl_core::Symbol]) -> String {
    inputs
        .iter()
        .map(|s| module.resolve(*s))
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_dim(dim: Dim) -> String {
    match dim {
        Dim::Static(n) => n.to_string(),
        Dim::Dynamic => "%dynamic".to_string(),
    }
}

fn scalar_type_name(s: ScalarType) -> &'static str {
    match s {
        ScalarType::Real => "real",
        ScalarType::Integer => "integer",
        ScalarType::Boolean => "boolean",
        ScalarType::Complex => "complex",
    }
}
