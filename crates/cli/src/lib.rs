//! `flatppl-cli` shared library — verb-agnostic helpers used by every binary
//! in this crate.
//!
//! Holds format detection, parse/print dispatch, diagnostic rendering, and the
//! exit-code helper so all driver binaries share identical behaviour. The
//! converter-specific logic (clap structs, verb handlers) stays in each binary.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Config, Label, Report, ReportKind, Source};
use flatppl_core::Module;

pub mod provenance;
pub use provenance::Provenance;

// ── clap mirrors ───────────────────────────────────────────────────────────────

/// CLI mirror of [`flatppl_syntax::Syntax`].
#[cfg(any(feature = "convert", feature = "fmtlint"))]
#[derive(Clone, Copy, clap::ValueEnum)]
pub enum SyntaxLevel {
    Full,
    Minimal,
}

#[cfg(any(feature = "convert", feature = "fmtlint"))]
impl From<SyntaxLevel> for flatppl_syntax::Syntax {
    fn from(level: SyntaxLevel) -> Self {
        match level {
            SyntaxLevel::Full => flatppl_syntax::Syntax::Full,
            SyntaxLevel::Minimal => flatppl_syntax::Syntax::Minimal,
        }
    }
}

/// `fmt` arguments (shared by the `flatppl` and `flatppl-fmt` binaries).
#[cfg(feature = "fmtlint")]
#[derive(clap::Args)]
pub struct FmtArgs {
    /// Files to format (`.flatppl`). Omit, or pass `-`, for stdin.
    pub files: Vec<std::path::PathBuf>,
    /// Do not write; exit 1 if any file is not already canonical.
    #[arg(long)]
    pub check: bool,
    /// Output syntax level.
    #[arg(long, value_enum, default_value_t = SyntaxLevel::Full)]
    pub syntax: SyntaxLevel,
}

/// `lint` arguments (shared by both binaries).
#[cfg(feature = "fmtlint")]
#[derive(clap::Args)]
pub struct LintArgs {
    /// Files to lint (`.flatppl`).
    pub files: Vec<std::path::PathBuf>,
    /// Force a rule to `deny` (repeatable).
    #[arg(long = "deny", value_name = "RULE")]
    pub deny: Vec<String>,
    /// Force a rule to `warn` (repeatable).
    #[arg(long = "warn", value_name = "RULE")]
    pub warn: Vec<String>,
    /// Force a rule to `allow`, i.e. silence it (repeatable).
    #[arg(long = "allow", value_name = "RULE")]
    pub allow: Vec<String>,
    /// Promote every `warn`-level rule to `deny` (CI gate).
    #[arg(long)]
    pub deny_warnings: bool,
}

// ── Format ───────────────────────────────────────────────────────────────────

/// A serialization format, inferred from a filename extension.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Format {
    FlatPpl,
    FlatPir,
    /// The JSON encoding of FlatPIR (`.flatpir.json`).
    FlatPirJson,
}

impl Format {
    pub fn from_path(path: &Path) -> Result<Format, String> {
        // `.flatpir.json` is a double extension; match the full file name first,
        // since `Path::extension` only sees the trailing `json`.
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".flatpir.json") {
            return Ok(Format::FlatPirJson);
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some("flatppl") => Ok(Format::FlatPpl),
            Some("flatpir") => Ok(Format::FlatPir),
            Some(other) => Err(format!(
                "unsupported file extension `.{other}` for `{}` \
                 (expected `.flatppl`, `.flatpir`, or `.flatpir.json`)",
                path.display()
            )),
            None => Err(format!(
                "cannot infer a format for `{}`: no file extension \
                 (expected `.flatppl`, `.flatpir`, or `.flatpir.json`)",
                path.display()
            )),
        }
    }

    /// The line-comment marker for this format (for the provenance header), or
    /// `None` for a format that has no comment syntax (JSON — the header is
    /// omitted for it).
    pub fn line_comment(self) -> Option<&'static str> {
        match self {
            Format::FlatPpl => Some("%"),
            Format::FlatPir => Some(";"),
            Format::FlatPirJson => None,
        }
    }

    /// Human name of the format, for the header's `from:` field.
    pub fn describe(self) -> &'static str {
        match self {
            Format::FlatPpl => "FlatPPL",
            Format::FlatPir => "FlatPIR",
            Format::FlatPirJson => "FlatPIR JSON",
        }
    }
}

// ── Failure / diagnostics ────────────────────────────────────────────────────

/// Why a command failed: a plain one-line message (I/O, usage), or a parse
/// diagnostic rendered as a source-annotated report.
pub enum Failure {
    Plain(String),
    Diagnostic {
        path: PathBuf,
        source: String,
        message: String,
        /// 1-based source line (0 = unlocalized).
        line: usize,
        /// Byte span `[start, end)`, when the error carries one.
        span: Option<(usize, usize)>,
    },
}

impl From<String> for Failure {
    fn from(msg: String) -> Self {
        Failure::Plain(msg)
    }
}

/// Print a source-annotated error report to stderr. The span is the error's
/// own when it carries one; a line-only error highlights its whole line; an
/// unlocalized error degrades to a plain message.
pub(crate) fn render_diagnostic(
    path: &Path,
    source: &str,
    message: &str,
    line: usize,
    span: Option<(usize, usize)>,
) {
    let located = span.or_else(|| line_span(source, line));
    let (Some((start, end)), false) = (located, source.is_empty()) else {
        eprintln!("flatppl: {}: {message}", path.display());
        return;
    };
    // Clamp to the source: spans may legitimately point at EOF (zero-width
    // cursor past the last byte), and the renderer needs in-bounds offsets.
    let start = start.min(source.len() - 1);
    let end = end.clamp(start + 1, source.len());

    let name = path.display().to_string();
    let report = Report::build(ReportKind::Error, (name.clone(), start..end))
        .with_config(Config::default().with_color(std::io::stderr().is_terminal()))
        .with_message(message)
        .with_label(Label::new((name.clone(), start..end)).with_message("here"))
        .finish();
    let _ = report.eprint((name, Source::from(source)));
}

/// Byte range of the 1-based `line` in `source` (at least one byte, so an
/// empty line still renders a caret).
fn line_span(source: &str, line: usize) -> Option<(usize, usize)> {
    if line == 0 {
        return None;
    }
    let mut start = 0usize;
    for (i, raw) in source.split_inclusive('\n').enumerate() {
        if i + 1 == line {
            let content = raw.trim_end_matches(['\n', '\r']);
            return Some((start, start + content.len().max(1)));
        }
        start += raw.len();
    }
    None
}

/// Map a `Result<(), Failure>` to an exit code, printing errors to stderr.
/// Used as the final step of every driver binary's `main`.
pub fn report(result: Result<(), Failure>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Plain(msg)) => {
            eprintln!("flatppl: {msg}");
            ExitCode::FAILURE
        }
        Err(Failure::Diagnostic {
            path,
            source,
            message,
            line,
            span,
        }) => {
            render_diagnostic(&path, &source, &message, line, span);
            ExitCode::FAILURE
        }
    }
}

// ── Read / write dispatch ────────────────────────────────────────────────────

/// Parse/read `source`; an error comes back as `(message, line, span)` in a
/// format-agnostic shape for [`render_diagnostic`].
pub type ReadError = (String, usize, Option<(usize, usize)>);

pub fn read_module(format: Format, source: &str) -> Result<Module, ReadError> {
    fn widen(span: Option<(u32, u32)>) -> Option<(usize, usize)> {
        span.map(|(s, e)| (s as usize, e as usize))
    }
    match format {
        Format::FlatPpl => {
            flatppl_syntax::parse(source).map_err(|e| (e.message, e.line as usize, widen(e.span)))
        }
        #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
        Format::FlatPir => {
            flatppl_flatpir::read(source).map_err(|e| (e.message, e.line, widen(e.span)))
        }
        // The JSON encoding routes through synthesized canonical text, so its
        // errors are not source-positioned: report unlocalized (line 0).
        #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
        Format::FlatPirJson => {
            let value: serde_json::Value = serde_json::from_str(source)
                .map_err(|e| (format!("invalid `.flatpir.json`: {e}"), 0usize, None))?;
            flatppl_flatpir::from_json(&value).map_err(|e| (e.message, 0usize, None))
        }
        #[cfg(not(any(feature = "convert", feature = "infer", feature = "hs3")))]
        Format::FlatPir | Format::FlatPirJson => Err((
            "FlatPIR support not compiled in (missing `convert`, `infer`, or `hs3` feature)"
                .to_string(),
            0,
            None,
        )),
    }
}

pub fn write_module(
    format: Format,
    module: &Module,
    syntax: flatppl_syntax::Syntax,
) -> Result<String, Failure> {
    Ok(match format {
        Format::FlatPpl => flatppl_syntax::print_with(module, syntax),
        #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
        Format::FlatPir => flatppl_flatpir::write(module),
        #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
        Format::FlatPirJson => {
            let value = flatppl_flatpir::try_to_json(module)
                .map_err(|e| Failure::Plain(format!("encoding `.flatpir.json`: {}", e.message)))?;
            serde_json::to_string_pretty(&value)
                .map_err(|e| Failure::Plain(format!("serializing JSON: {e}")))?
        }
        #[cfg(not(any(feature = "convert", feature = "infer", feature = "hs3")))]
        Format::FlatPir | Format::FlatPirJson => unreachable!(
            "write_module called with a FlatPIR format in a lean build; all callers are guarded by a converter feature"
        ),
    })
}

// ── fmt/lint logic ────────────────────────────────────────────────────────────

/// Parse FlatPPL `source` and re-print it canonically with a trailing newline.
/// Errors come back in the `ReadError` shape `read_module` uses.
#[cfg(feature = "fmtlint")]
pub fn format_text(source: &str, syntax: flatppl_syntax::Syntax) -> Result<String, ReadError> {
    let module = read_module(Format::FlatPpl, source)?;
    let mut text = flatppl_syntax::print_with(&module, syntax);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

/// `flatppl fmt` logic — canonicalize FlatPPL in place / on stdin, or `--check`.
#[cfg(feature = "fmtlint")]
pub fn run_fmt(
    files: &[std::path::PathBuf],
    check: bool,
    syntax: flatppl_syntax::Syntax,
) -> Result<(), Failure> {
    use std::io::{Read, Write};

    let stdin_mode = files.is_empty() || (files.len() == 1 && files[0].as_os_str() == "-");
    if stdin_mode {
        let mut source = String::new();
        std::io::stdin()
            .read_to_string(&mut source)
            .map_err(|e| format!("reading stdin: {e}"))?;
        let formatted =
            format_text(&source, syntax).map_err(|(message, line, span)| Failure::Diagnostic {
                path: std::path::PathBuf::from("<stdin>"),
                source: source.clone(),
                message,
                line,
                span,
            })?;
        if check {
            if source != formatted {
                return Err(Failure::Plain("stdin is not canonically formatted".into()));
            }
            return Ok(());
        }
        std::io::stdout()
            .write_all(formatted.as_bytes())
            .map_err(|e| format!("writing stdout: {e}"))?;
        return Ok(());
    }

    let mut dirty: Vec<std::path::PathBuf> = Vec::new();
    for file in files {
        match Format::from_path(file)? {
            Format::FlatPpl => {}
            Format::FlatPir | Format::FlatPirJson => {
                return Err(Failure::Plain(format!(
                    "`fmt` only formats FlatPPL; `{}` is FlatPIR (use `convert`)",
                    file.display()
                )));
            }
        }
        let source = std::fs::read_to_string(file)
            .map_err(|e| format!("reading `{}`: {e}", file.display()))?;
        let formatted =
            format_text(&source, syntax).map_err(|(message, line, span)| Failure::Diagnostic {
                path: file.clone(),
                source: source.clone(),
                message,
                line,
                span,
            })?;
        if check {
            if source != formatted {
                dirty.push(file.clone());
            }
        } else if source != formatted {
            std::fs::write(file, &formatted)
                .map_err(|e| format!("writing `{}`: {e}", file.display()))?;
        }
    }

    if check && !dirty.is_empty() {
        for f in &dirty {
            eprintln!("flatppl: not canonically formatted: {}", f.display());
        }
        return Err(Failure::Plain(format!(
            "{} file(s) not canonically formatted",
            dirty.len()
        )));
    }
    Ok(())
}

/// `flatppl lint` logic — run the rule set, print diagnostics, fail on any deny.
///
/// The `not-canonical` rule compares against the normative `Syntax::Full` canonical form.
#[cfg(feature = "fmtlint")]
pub fn run_lint(
    files: &[std::path::PathBuf],
    deny: &[String],
    warn: &[String],
    allow: &[String],
    deny_warnings: bool,
) -> Result<(), Failure> {
    use flatppl_lint::{Config, RuleId, Severity};

    if files.is_empty() {
        return Err(Failure::Plain("lint: no input files".into()));
    }

    let mut base = Config::default();
    for (flag, level) in [
        (deny, Severity::Deny),
        (warn, Severity::Warn),
        (allow, Severity::Allow),
    ] {
        for name in flag {
            let rule: RuleId = name
                .parse()
                .map_err(|e: String| Failure::Plain(format!("lint: {e}")))?;
            base.set(rule, level);
        }
    }

    let mut any_deny = false;
    for file in files {
        let from = Format::from_path(file)?;
        let source = std::fs::read_to_string(file)
            .map_err(|e| format!("reading `{}`: {e}", file.display()))?;

        let mut cfg = base.clone();
        for rule in inline_allows(&source) {
            cfg.set(rule, Severity::Allow);
        }
        if deny_warnings {
            for rule in RuleId::ALL {
                if cfg.level(rule) == Severity::Warn {
                    cfg.set(rule, Severity::Deny);
                }
            }
        }

        let mut module = match read_module(from, &source) {
            Ok(m) => m,
            Err((message, line, span)) => {
                return Err(Failure::Diagnostic {
                    path: file.clone(),
                    source,
                    message,
                    line,
                    span,
                });
            }
        };

        let mut diags = flatppl_lint::lint(&mut module, &cfg);

        // `not-canonical` only applies to FlatPPL, and re-printing the whole
        // module is not free — skip it when the rule is silenced.
        let canonical_sev = cfg.level(RuleId::NotCanonical);
        if matches!(from, Format::FlatPpl) && canonical_sev != Severity::Allow {
            let mut canonical = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Full);
            if !canonical.ends_with('\n') {
                canonical.push('\n');
            }
            if source != canonical {
                diags.push(flatppl_lint::Diagnostic {
                    rule: RuleId::NotCanonical,
                    severity: canonical_sev,
                    message: "file is not canonically formatted (run `flatppl-fmt fmt`)".into(),
                    span: None,
                });
            }
        }

        // Lock stderr once per file and write all diagnostics under that lock —
        // one acquire instead of one per line (matters when many fire on piped,
        // unbuffered stderr).
        use std::io::Write;
        let path = file.display();
        let mut err = std::io::stderr().lock();
        for d in &diags {
            let tag = match d.severity {
                Severity::Deny => "error",
                Severity::Warn => "warning",
                Severity::Allow => continue,
            };
            let _ = writeln!(err, "{path}: {tag}[{}]: {}", d.rule, d.message);
            if matches!(d.severity, Severity::Deny) {
                any_deny = true;
            }
        }
    }

    if any_deny {
        return Err(Failure::Plain("lint found errors".into()));
    }
    Ok(())
}

/// Lint a freshly generated FlatPPL `module` and print any findings to stderr
/// as advisory diagnostics tagged with the output `path`. Conversion output is
/// canonically formatted by construction (it comes straight from the printer),
/// so `not-canonical` is silenced and this never fails the run — it only
/// surfaces modelling-level lints the importer's output happens to trip.
#[cfg(feature = "fmtlint")]
pub fn lint_generated(module: &mut Module, path: &std::path::Path) {
    use flatppl_lint::{Config, RuleId, Severity};
    use std::io::Write;

    let mut cfg = Config::default();
    cfg.set(RuleId::NotCanonical, Severity::Allow);
    let diags = flatppl_lint::lint(module, &cfg);

    let disp = path.display();
    let mut err = std::io::stderr().lock();
    for d in &diags {
        let tag = match d.severity {
            Severity::Deny => "error",
            Severity::Warn => "warning",
            Severity::Allow => continue,
        };
        let _ = writeln!(err, "{disp}: {tag}[{}]: {}", d.rule, d.message);
    }
}

/// Scan source for file-level `% flatppl-lint: allow RULE` directives.
#[cfg(feature = "fmtlint")]
fn inline_allows(source: &str) -> Vec<flatppl_lint::RuleId> {
    const PREFIX: &str = "% flatppl-lint: allow ";
    let mut out = Vec::new();
    for line in source.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(PREFIX) {
            if let Ok(rule) = rest.trim().parse() {
                out.push(rule);
            }
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "fmtlint"))]
mod tests {
    use super::*;

    #[test]
    fn format_text_is_idempotent() {
        let src = "x ~ Normal(mu=0.0,sigma=1.0)\n";
        let once = format_text(src, flatppl_syntax::Syntax::Full).unwrap();
        let twice = format_text(&once, flatppl_syntax::Syntax::Full).unwrap();
        assert_eq!(once, twice);
        assert!(once.contains("mu = 0.0"));
    }
}
