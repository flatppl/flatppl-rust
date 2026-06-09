//! Index-based handles, arenas, side-tables, and the symbol interner — the
//! backbone the rest of the IR is addressed through.
//!
//! `flatppl-core` represents the IR as an arena of nodes addressed by integer
//! handles ([`NodeId`], [`BindingId`]) rather than a pointer graph: this is
//! rewrite-friendly and avoids `Rc`/borrow-checker friction (the standard choice
//! for compiler IRs). Per-node / per-binding analysis results live in
//! *side-tables* ([`SecondaryMap`]) keyed by the same handles, so the nodes
//! themselves stay free of mutable annotation state.
//!
//! These hand-rolled types keep the crate dependency-free for now; the surface
//! is intentionally close to `la_arena` / `id-arena`, which could replace them.

use std::collections::HashMap;
use std::marker::PhantomData;

/// A handle that indexes into an [`Arena`] / [`SecondaryMap`].
pub trait Idx: Copy + Eq {
    fn from_usize(i: usize) -> Self;
    fn index(self) -> usize;
}

/// Handle for an expression node in a [`Module`](crate::Module)'s node arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(u32);

/// Handle for a top-level binding in a [`Module`](crate::Module).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct BindingId(u32);

/// An interned name (binding / field / op / axis names, and string-keyed lookups).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Symbol(u32);

impl Idx for NodeId {
    fn from_usize(i: usize) -> Self {
        NodeId(i as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}
impl Idx for BindingId {
    fn from_usize(i: usize) -> Self {
        BindingId(i as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}
impl Idx for Symbol {
    fn from_usize(i: usize) -> Self {
        Symbol(i as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Append-only arena. IR nodes are never freed individually — rewrites allocate
/// fresh nodes — so a `Vec` indexed by a typed handle is sufficient.
#[derive(Clone, Debug)]
pub struct Arena<I: Idx, T> {
    items: Vec<T>,
    _marker: PhantomData<I>,
}

impl<I: Idx, T> Default for Arena<I, T> {
    fn default() -> Self {
        Arena {
            items: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<I: Idx, T> Arena<I, T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate `value`, returning its handle.
    pub fn alloc(&mut self, value: T) -> I {
        let id = I::from_usize(self.items.len());
        self.items.push(value);
        id
    }

    pub fn get(&self, id: I) -> &T {
        &self.items[id.index()]
    }
    pub fn get_mut(&mut self, id: I) -> &mut T {
        &mut self.items[id.index()]
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate `(handle, &value)` in allocation order.
    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, v)| (I::from_usize(i), v))
    }
}

impl<I: Idx, T> std::ops::Index<I> for Arena<I, T> {
    type Output = T;
    fn index(&self, id: I) -> &T {
        self.get(id)
    }
}

/// A sparse map from a handle to an analysis result — the "side-table" for
/// annotations (types, phases, spans, …). Empty until a pass fills it.
#[derive(Clone, Debug)]
pub struct SecondaryMap<I: Idx, T> {
    slots: Vec<Option<T>>,
    _marker: PhantomData<I>,
}

impl<I: Idx, T> Default for SecondaryMap<I, T> {
    fn default() -> Self {
        SecondaryMap {
            slots: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<I: Idx, T> SecondaryMap<I, T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: I, value: T) {
        let i = id.index();
        if i >= self.slots.len() {
            self.slots.resize_with(i + 1, || None);
        }
        self.slots[i] = Some(value);
    }

    pub fn get(&self, id: I) -> Option<&T> {
        self.slots.get(id.index()).and_then(|s| s.as_ref())
    }

    pub fn contains(&self, id: I) -> bool {
        self.get(id).is_some()
    }
}

/// Interns names into compact [`Symbol`]s (cheap `Copy` / `Eq`).
#[derive(Clone, Debug, Default)]
pub struct Interner {
    lookup: HashMap<Box<str>, Symbol>,
    names: Vec<Box<str>>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, name: &str) -> Symbol {
        if let Some(&sym) = self.lookup.get(name) {
            return sym;
        }
        let sym = Symbol::from_usize(self.names.len());
        let boxed: Box<str> = name.into();
        self.names.push(boxed.clone());
        self.lookup.insert(boxed, sym);
        sym
    }

    /// Resolve a symbol back to its name. Panics on a symbol from another interner.
    pub fn resolve(&self, sym: Symbol) -> &str {
        &self.names[sym.index()]
    }
}
