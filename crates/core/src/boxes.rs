//! Black boxes bound to `load_module` aliases (flatppl-bridge).
//!
//! A box interface is an ordinary module whose members are
//! `external(anything)`. A host binds a top-level `load_module` alias to a box
//! and supplies the box's describe reply. The binding lives on the importing
//! [`Module`](crate::Module), so every pass that sees the alias sees the box.

use std::collections::BTreeMap;

use crate::ty::Type;

/// The kind of a box member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxKind {
    Function,
    Kernel,
    Measure,
}

impl BoxKind {
    pub fn name(self) -> &'static str {
        match self {
            BoxKind::Function => "function",
            BoxKind::Kernel => "kernel",
            BoxKind::Measure => "measure",
        }
    }
}

/// One declared value set: its FlatPPL text and the type of its elements, in
/// the interner of the module that holds the binding. `anything` has type
/// [`Type::Any`].
#[derive(Clone, Debug, PartialEq)]
pub struct BoxSet {
    pub text: Box<str>,
    pub ty: Type,
}

impl BoxSet {
    /// `true` for the set `anything`, which declares no shape.
    pub fn is_anything(&self) -> bool {
        self.text.trim() == "anything"
    }
}

/// One member of a box, from its describe entry.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxMember {
    pub kind: BoxKind,
    pub queries: Vec<Box<str>>,
    pub gradient: bool,
    pub vjp: bool,
    pub normalized: bool,
    pub input: Option<BoxSet>,
    pub output: Option<BoxSet>,
    pub variate: Option<BoxSet>,
}

impl BoxMember {
    pub fn supports(&self, query: &str) -> bool {
        self.queries.iter().any(|q| &**q == query)
    }
}

/// A box: its name on the server and its members.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxBinding {
    pub name: Box<str>,
    pub members: BTreeMap<Box<str>, BoxMember>,
}
