//! Type, shape, and phase inference for [`flatppl_core`] modules.
//!
//! The **type-domain trace** (engine-concepts §17): a memoised walk of the IR
//! in dependency order, dispatching each call to a per-op rule and filling the
//! module's type/phase side-tables. Inference is **structure-preserving** —
//! it reads and annotates, never rewrites ("resolve, don't rewrite",
//! engine-concepts §17.1); the annotated module projects to spec-§11
//! `(%meta <type> <phase>)` on FlatPIR write.
//!
//! **Honesty over coverage.** The op catalogue ([`ops`]) grows with the
//! toolchain. An op without a rule yields `%deferred` plus a once-per-op
//! note diagnostic — never a guess; genuine type errors (reference cycles,
//! unresolved names) yield `(%failed …)` plus an error diagnostic. Bindings
//! whose types hinge on deferred features (cross-module references — see
//! `TODO-flatppl-rust.md` on `load_module`) are likewise `%deferred`.
//!
//! Phases follow the spec-§04 ancestor rule: `stochastic > parameterized >
//! fixed`, with `elementof` introducing *parameterized*, `draw` *stochastic*,
//! and `external` / loaded data / reifications *fixed*.

mod ops;
mod trace;

use flatppl_core::Module;

/// A message produced during inference. `Error` marks an ill-formed module
/// (the offending nodes carry `(%failed …)` types); `Note` records an
/// honest gap (op not yet in the catalogue, deferred feature).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Note,
}

impl Diagnostic {
    pub(crate) fn error(message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
        }
    }
    pub(crate) fn note(message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Note,
            message: message.into(),
        }
    }
}

/// How far to take the trace. The levels form a hierarchy — each includes
/// everything below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Phases only (the spec-§04 ancestor classification). Types are not
    /// annotated — `%meta` carries `%deferred` in the type slot.
    Phase,
    /// Phases + structural types. Array dimensions are static only where
    /// they are syntactically literal.
    Type,
    /// Phases + types + shape resolution: fixed-phase integer expressions at
    /// shape positions (`iid` counts, `cartpow` sizes, distribution dims)
    /// are resolved demand-driven (engine-concepts §17.1 — "resolve, don't
    /// rewrite"; the source IR is never modified).
    Shape,
}

/// Infer at [`Level::Shape`] (everything) — see [`infer_with`].
pub fn infer(module: &mut Module) -> Vec<Diagnostic> {
    infer_with(module, Level::Shape)
}

/// Infer types and phases for every binding of `module` at the given
/// [`Level`], filling its type/phase side-tables in place. Best-effort:
/// always annotates as much as it can; returned diagnostics report errors
/// (ill-formed module) and notes (honest `%deferred` gaps).
pub fn infer_with(module: &mut Module, level: Level) -> Vec<Diagnostic> {
    trace::Inferencer::new(module, level).run()
}
