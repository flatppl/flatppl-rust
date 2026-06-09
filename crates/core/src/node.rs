//! IR nodes — the expression model.
//!
//! `flatppl-core` holds *sugar-stripped, construct-preserving* FlatPIR: surface
//! sugar (operators, indexing, `~`, `:=`, dot-broadcast, lambda / `fn`) is gone —
//! it became `add` / `get` / `draw` / `aggregate` / `functionof` calls in
//! `flatppl-syntax` — but named constructs (`metricsum`, `aggregate`, measure
//! ops, distributions, `kernelof`, …) are kept. Lowering between levels is a
//! separate, deliberate pass.

use crate::id::{NodeId, Symbol};

/// A single IR expression node, addressed by [`NodeId`].
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// A primitive scalar literal (`3`, `1.0`, `true`, `"path.csv"`).
    Lit(Scalar),
    /// A bare built-in symbol in value position: a constant or set (`pi`, `inf`,
    /// `im`, `reals`, …) or a built-in function used as a value (the `sum` in
    /// `reduce(sum, …)`, the `add` in `aggregate(add, …)`). In FlatPIR these are
    /// all bare atoms; the constant-vs-function distinction is resolved by
    /// inference, not carried structurally. (User-binding references are
    /// [`Ref`]; this is only for `base` built-ins.)
    Const(Symbol),
    /// A reference to a binding (`(%ref <ns> <name>)`).
    Ref(Ref),
    /// A bare hole `_`. Rarely present in lowered bindings (`fn` / holes desugar
    /// to `functionof` + `%local` placeholders), but FlatPIR can carry it.
    Hole,
    /// An aggregate / metricsum axis label (`(%axis i)` / `(%uaxis)` / `(%laxis)`).
    Axis(Axis),
    /// A built-in operation or user-callable application.
    Call(Call),
}

/// A primitive scalar literal.
///
/// Complex numbers, vectors, and records are *constructor calls* (e.g.
/// `(complex re im)`, `(vector …)`, `(record …)`), not scalar literals — see
/// [`Node::Call`]. (Having `Int`/`Real` as distinct variants is what lets us drop
/// the JS engine's `numType` tag.)
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Real(f64),
    Bool(bool),
    Str(Box<str>),
}

/// A reference to a binding, in one of three namespaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ref {
    pub ns: RefNs,
    pub name: Symbol,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefNs {
    /// `(%ref self x)` — a binding in the current module.
    SelfMod,
    /// `(%ref %local x)` — a input inside `functionof` / `kernelof`.
    Local,
    /// `(%ref <alias> x)` — a binding in a loaded module.
    Module(Symbol),
}

/// An axis label inside `aggregate` / `metricsum`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axis {
    pub name: Symbol,
    /// `None` for a neutral `aggregate` axis; `Some(..)` for a metricsum
    /// variance-marked axis (`(%uaxis)` / `(%laxis)`).
    pub variance: Option<Variance>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variance {
    Upper,
    Lower,
}

/// A call: a built-in operation or a user-defined callable application.
///
/// The argument shape mirrors FlatPIR: positional `args`, `named` entries (which
/// distinguish `%kwarg` / `%field` / `%assign`), and — for the reification
/// constructs `functionof` / `kernelof` only — the [`Inputs`] list (spec §11
/// "Reified callables"; the reified output expression is the single positional
/// argument). The engine-internal ops `tuple_get` / `get_field` are ordinary
/// `Builtin` calls.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub head: CallHead,
    pub args: Box<[NodeId]>,
    pub named: Box<[NamedArg]>,
    /// `Some` iff this call is a `functionof` / `kernelof` reification.
    pub inputs: Option<Inputs>,
}

/// The input list of a reified callable (spec §11 "Reified callables"):
/// the *cut* — which nodes become the callable's inputs, under which names.
///
/// Entries are plain `(Symbol, Ref)` data, not child nodes: dependency analyses
/// read them as their own bucket (boundary sources), distinct from body deps.
/// The entry's [`Ref`] carries the binding-vs-placeholder origin structurally
/// (`%local` namespace ⇔ placeholder).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inputs {
    /// `%specinputs` — an authored (all-or-none) boundary specification.
    /// Ordered: the callable is positionally callable. A placeholder entry
    /// (`(x, %local _x_)`) is that placeholder's declaration. Never produced
    /// by inference; round-trips to surface boundary keyword arguments.
    Spec(Box<[(Symbol, Ref)]>),
    /// `%autoinputs` — a boundary-less reification (keyword-only callable).
    /// On the wire the list is `%deferred` until phase inference fills it;
    /// the filled list is inference metadata and lives in the module's
    /// auto-inputs side-table (see [`Module::auto_inputs_of`]), not here.
    ///
    /// [`Module::auto_inputs_of`]: crate::Module::auto_inputs_of
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallHead {
    /// A built-in op named bare in FlatPIR (`add`, `Normal`, `metricsum`, `draw`, …).
    Builtin(Symbol),
    /// A call to a user-defined callable — `(%call <callable> …)`. The callee
    /// is an expression node that must evaluate to a callable: a [`Node::Ref`]
    /// in the common case, or an inline callable expression such as a
    /// reification (spec §11).
    User(NodeId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedArg {
    pub kind: NamedKind,
    pub name: Symbol,
    pub value: NodeId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedKind {
    /// `%kwarg` — order-insignificant keyword argument.
    Kwarg,
    /// `%field` — order-significant named entry (`record` / `joint` / `jointchain`
    /// / `cartprod` / `table`).
    Field,
    /// `%assign` — load-time substitution (`load_module` / `standard_module`).
    Assign,
}

impl Node {
    /// Visit the child sub-nodes of this node, in a stable order: the callee
    /// (for a user call) first, then positional args, then named entries.
    ///
    /// This is the *single* declarative enumeration of child-carrying positions —
    /// every walker (inference, lowering, printing, …) traverses through here, so
    /// none can miss a position as the node set grows. (Note: [`Inputs`] entries
    /// are name/ref leaves, not child sub-nodes — dependency analyses read them
    /// as their own boundary-source bucket.)
    pub fn for_each_child(&self, mut f: impl FnMut(NodeId)) {
        if let Node::Call(c) = self {
            if let CallHead::User(callee) = c.head {
                f(callee);
            }
            for &a in c.args.iter() {
                f(a);
            }
            for n in c.named.iter() {
                f(n.value);
            }
        }
    }

    /// The child sub-nodes as a `Vec` (convenience over [`Self::for_each_child`]).
    pub fn children(&self) -> Vec<NodeId> {
        let mut kids = Vec::new();
        self.for_each_child(|c| kids.push(c));
        kids
    }
}
