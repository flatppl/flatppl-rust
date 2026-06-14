//! `flatppl-lint` — lint rules over the [`flatppl_core`] IR.
//!
//! Binary-free: all CLI surface lives in `flatppl-cli`. Rules fall into two
//! groups — an [`flatppl_infer`] *bridge* (re-surfacing inference errors/notes
//! as located lint rules) and *native* rules over the IR ([`rules`]). The
//! `not-canonical` rule needs source text and is driven from the CLI, which
//! appends it to [`lint`]'s result.

mod builtins;
mod rules;

use std::fmt;
use std::str::FromStr;

use flatppl_core::{Module, Span};

/// Effective level after [`Config`] is applied. `Allow` = suppressed (the
/// diagnostic is never emitted by [`lint`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Allow,
    Warn,
    Deny,
}

/// One variant per lint rule. `Display`/`FromStr` use the kebab name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuleId {
    UnresolvedName,
    InferenceCycle,
    InferenceGap,
    NotCanonical,
    UnusedBinding,
    ShadowsBuiltin,
    MissingDoc,
}

impl RuleId {
    /// Every rule, for iteration (config parsing, `--help` listings).
    pub const ALL: [RuleId; 7] = [
        RuleId::UnresolvedName,
        RuleId::InferenceCycle,
        RuleId::InferenceGap,
        RuleId::NotCanonical,
        RuleId::UnusedBinding,
        RuleId::ShadowsBuiltin,
        RuleId::MissingDoc,
    ];

    /// The kebab-case rule name (stable; used on the CLI and in suppression).
    pub fn name(self) -> &'static str {
        match self {
            RuleId::UnresolvedName => "unresolved-name",
            RuleId::InferenceCycle => "inference-cycle",
            RuleId::InferenceGap => "inference-gap",
            RuleId::NotCanonical => "not-canonical",
            RuleId::UnusedBinding => "unused-binding",
            RuleId::ShadowsBuiltin => "shadows-builtin",
            RuleId::MissingDoc => "missing-doc",
        }
    }

    /// The built-in (pre-config) severity for this rule (spec rule table).
    pub fn default_severity(self) -> Severity {
        match self {
            RuleId::UnresolvedName | RuleId::InferenceCycle => Severity::Deny,
            RuleId::InferenceGap
            | RuleId::NotCanonical
            | RuleId::UnusedBinding
            | RuleId::ShadowsBuiltin => Severity::Warn,
            RuleId::MissingDoc => Severity::Allow,
        }
    }

    /// Dense index into [`RuleId::ALL`] — the slot this rule occupies in a
    /// [`Config`]. `ALL` is the single source of truth for the ordering.
    pub(crate) fn index(self) -> usize {
        RuleId::ALL
            .iter()
            .position(|&r| r == self)
            .expect("every RuleId is in ALL")
    }
}

/// Number of lint rules — the width of a [`Config`]'s override array.
const NUM_RULES: usize = RuleId::ALL.len();

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for RuleId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        RuleId::ALL
            .into_iter()
            .find(|r| r.name() == s)
            .ok_or_else(|| format!("unknown lint rule `{s}`"))
    }
}

/// A single lint finding. `span` is `None` for the infer-bridge rules (inference
/// diagnostics carry no source location).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub rule: RuleId,
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
}

/// Per-rule severity overrides; [`Config::default`] = built-in levels.
///
/// Backed by a fixed-size array (one slot per [`RuleId`]) rather than a
/// `HashMap`, so `Config` lives on the stack: `clone()` is a memcpy and
/// `level()` an array index — no per-file heap allocation when linting many
/// files.
#[derive(Clone, Debug)]
pub struct Config {
    /// `None` in a slot ⇒ use that rule's `default_severity()`.
    overrides: [Option<Severity>; NUM_RULES],
}

impl Default for Config {
    fn default() -> Self {
        Config {
            overrides: [None; NUM_RULES],
        }
    }
}

impl Config {
    /// Override one rule's severity.
    pub fn set(&mut self, rule: RuleId, severity: Severity) {
        self.overrides[rule.index()] = Some(severity);
    }

    /// The effective severity for `rule` (override if set, else the default).
    pub fn level(&self, rule: RuleId) -> Severity {
        self.overrides[rule.index()].unwrap_or_else(|| rule.default_severity())
    }
}

/// The rules sourced from the [`flatppl_infer`] bridge.
const INFER_RULES: [RuleId; 3] = [
    RuleId::UnresolvedName,
    RuleId::InferenceCycle,
    RuleId::InferenceGap,
];

/// Run every rule (the infer bridge + the native rules) over `module`. Takes
/// `&mut` because the infer bridge runs [`flatppl_infer::infer_with`], which
/// fills type/phase side-tables. Diagnostics whose effective severity is
/// [`Severity::Allow`] are filtered out. The `not-canonical` rule is not run
/// here (it needs source text) — the CLI appends it.
///
/// The infer pass (the dominant cost) is skipped entirely when all three
/// infer-bridge rules are configured to [`Severity::Allow`].
pub fn lint(module: &mut Module, cfg: &Config) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if INFER_RULES.iter().any(|&r| cfg.level(r) != Severity::Allow) {
        out = infer_bridge(module, cfg);
    }
    out.extend(rules::native(module, cfg));
    out
}

/// Test-only view of the built-in name set (for the sync test).
#[doc(hidden)]
pub fn test_builtins() -> &'static [&'static str] {
    builtins::BUILTINS
}

/// Map an [`flatppl_infer`] run to located lint rules. Errors become
/// `inference-cycle` (when the message mentions a cycle) or `unresolved-name`;
/// notes become `inference-gap`. Inference diagnostics carry no span.
///
/// Runs at [`Level::Type`], not [`Level::Shape`]: the deny-level rules
/// (reference cycles, unresolved names) are detected during name resolution at
/// every level, so the linter skips the expensive demand-driven shape
/// const-eval — faster, and fewer low-value `%deferred` gap notes.
///
/// [`Level::Type`]: flatppl_infer::Level::Type
/// [`Level::Shape`]: flatppl_infer::Level::Shape
fn infer_bridge(module: &mut Module, cfg: &Config) -> Vec<Diagnostic> {
    let diags = flatppl_infer::infer_with(module, flatppl_infer::Level::Type);
    let mut out = Vec::new();
    for d in diags {
        let rule = match d.severity {
            // infer's cycle message ("…reference cycle…") is lowercase ASCII —
            // a direct `contains` avoids allocating a lowercased copy.
            flatppl_infer::Severity::Error if d.message.contains("cycle") => RuleId::InferenceCycle,
            flatppl_infer::Severity::Error => RuleId::UnresolvedName,
            flatppl_infer::Severity::Note => RuleId::InferenceGap,
        };
        push(&mut out, cfg, rule, d.message, None);
    }
    out
}

/// Push a diagnostic unless config silences the rule. Shared by every rule.
pub(crate) fn push(
    out: &mut Vec<Diagnostic>,
    cfg: &Config,
    rule: RuleId,
    message: String,
    span: Option<Span>,
) {
    let severity = cfg.level(rule);
    if severity == Severity::Allow {
        return;
    }
    out.push(Diagnostic {
        rule,
        severity,
        message,
        span,
    });
}
