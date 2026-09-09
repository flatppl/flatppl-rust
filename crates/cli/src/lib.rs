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

/// Render a filesystem path without allowing its text to control terminal
/// layout. Filesystem operations continue to use the original [`Path`].
pub fn terminal_path(path: &Path) -> String {
    terminal_text(&path.to_string_lossy())
}

/// Require a local source path to resolve to a regular file. Metadata follows
/// symlinks, so a symlink to a regular file remains a valid input.
pub fn require_regular_file(path: &Path) -> std::io::Result<()> {
    if std::fs::metadata(path)?.is_file() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ))
    }
}

/// Read a UTF-8 local source after rejecting directories and special files.
pub fn read_regular_utf8(path: &Path) -> std::io::Result<String> {
    require_regular_file(path)?;
    std::fs::read_to_string(path)
}

/// Render untrusted text as one terminal-safe representation. Ordinary
/// printable Unicode stays readable; controls and format characters are
/// exposed with Rust debug escapes.
pub(crate) fn terminal_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        out.extend(ch.escape_debug());
    }
    out
}

pub mod provenance;
pub use provenance::banner;

/// Source resolution (local paths + `http`/`https` URLs) through the
/// `flatppl-fileaccess` layer, and `load_module` dependency-graph assembly.
#[cfg(any(feature = "infer", feature = "prepare"))]
pub mod resolve;

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
    /// An HTML page rendering the model as mathematics (`.html`). Output only.
    Html,
    /// GitHub Markdown with TeX math (`.md`, `.markdown`). Output only.
    Markdown,
    /// A standalone LaTeX document (`.tex`). Output only.
    Latex,
    /// A native Typst document (`.typ`). Output only.
    Typst,
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
            Some("html") => Ok(Format::Html),
            Some("md" | "markdown") => Ok(Format::Markdown),
            Some("tex") => Ok(Format::Latex),
            Some("typ") => Ok(Format::Typst),
            Some(other) => Err(format!(
                "unsupported file extension `.{}` for `{}` \
                 (expected `.flatppl`, `.flatpir`, `.flatpir.json`, `.html`, `.md`, `.markdown`, `.tex`, or `.typ`)",
                terminal_text(other),
                terminal_path(path)
            )),
            None => Err(format!(
                "cannot infer a format for `{}`: no file extension \
                 (expected `.flatppl`, `.flatpir`, `.flatpir.json`, `.html`, `.md`, `.markdown`, `.tex`, or `.typ`)",
                terminal_path(path)
            )),
        }
    }

    /// Infer the format of a resolved dependency [`Location`] from its final
    /// filename — so a URL dep (`https://…/helper.flatppl`) detects the same way
    /// as a local one.
    #[cfg(any(feature = "infer", feature = "prepare"))]
    pub fn from_location(loc: &flatppl_fileaccess::Location) -> Result<Format, String> {
        Format::from_path(Path::new(&loc.name()))
    }

    /// How the leading generated-file banner is commented for this format.
    ///
    /// FlatPPL and FlatPIR both use a single line comment — `#` and `;`
    /// respectively. (A `#` comment ends at the first `;` per spec §05, but the
    /// banner text carries none, so a line comment is safe.) JSON has no comment
    /// syntax, so it gets no banner.
    pub fn comment_style(self) -> CommentStyle {
        match self {
            Format::FlatPpl => CommentStyle::Line("#"),
            Format::FlatPir => CommentStyle::Line(";"),
            Format::FlatPirJson => CommentStyle::None,
            Format::Html | Format::Markdown => CommentStyle::Block("<!--", "-->"),
            Format::Latex => CommentStyle::Line("%"),
            Format::Typst => CommentStyle::Line("//"),
        }
    }

    /// Document outputs use the typed math renderer instead of a module printer.
    pub fn is_document(self) -> bool {
        matches!(
            self,
            Self::Html | Self::Markdown | Self::Latex | Self::Typst
        )
    }
}

/// How the generated-file banner is commented for a given [`Format`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CommentStyle {
    /// Prefix the banner line with this marker (e.g. FlatPPL `#`, FlatPIR `;`).
    Line(&'static str),
    /// The format has no comment syntax (JSON): no banner is written.
    None,
    /// Wrap the banner line in these open/close markers (HTML `<!-- -->`).
    Block(&'static str, &'static str),
}

// ── Failure / diagnostics ────────────────────────────────────────────────────

/// Why a command failed: a plain one-line message (I/O, usage), or a parse
/// diagnostic rendered as a source-annotated report.
#[derive(Debug)]
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
    /// A determiniser refusal — a construct that cannot be legalized to FlatPDL.
    /// Distinct exit code (3) so callers classify refuse ≠ parse/IO error.
    Refuse(String),
    /// A bad command-line argument value (e.g. an unrecognized `--mode`).
    /// Distinct exit code (2), the conventional usage-error code.
    Usage(String),
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
        eprintln!("flatppl: {}: {message}", terminal_path(path));
        return;
    };
    // Clamp to the source: spans may legitimately point at EOF (zero-width
    // cursor past the last byte), and the renderer needs in-bounds offsets.
    let start = start.min(source.len() - 1);
    let end = end.clamp(start + 1, source.len());

    let name = terminal_path(path);
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
        Err(Failure::Refuse(msg)) => {
            eprintln!("{msg}");
            ExitCode::from(3)
        }
        Err(Failure::Usage(msg)) => {
            eprintln!("flatppl: {msg}");
            ExitCode::from(2)
        }
    }
}

// ── Read / write dispatch ────────────────────────────────────────────────────

/// Parse/read `source`; an error comes back as `(message, line, span)` in a
/// format-agnostic shape for [`render_diagnostic`].
pub type ReadError = (String, usize, Option<(usize, usize)>);

#[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
const MAX_JSON_STRUCTURAL_TOKENS: usize = 262_144;

/// Bound JSON value breadth before `serde_json::Value` allocates the full tree.
#[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
fn guard_json_structure(source: &str) -> Result<(), ReadError> {
    let mut in_string = false;
    let mut escaped = false;
    let mut count = 0usize;
    for byte in source.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        if byte == b'"' {
            in_string = true;
        } else if matches!(byte, b'[' | b'{' | b',') {
            count += 1;
            if count > MAX_JSON_STRUCTURAL_TOKENS {
                return Err((
                    format!(
                        "`.flatpir.json` structure exceeds the limit of {MAX_JSON_STRUCTURAL_TOKENS} tokens; this is a resource guard, not a language rule"
                    ),
                    0,
                    None,
                ));
            }
        }
    }
    Ok(())
}

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
            guard_json_structure(source)?;
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
        Format::Html | Format::Markdown | Format::Latex | Format::Typst => Err((
            "Mathematical documents are output formats only; the input must be FlatPPL, FlatPIR, or an HS3/pyhf document"
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
        Format::FlatPir => flatppl_flatpir::try_write(module)
            .map_err(|e| Failure::Plain(format!("writing `.flatpir`: {}", e.message)))?,
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
        // Documents are rendered from the typed module by the `mathdoc` path in
        // `convert`, never from the bare module this function sees.
        Format::Html | Format::Markdown | Format::Latex | Format::Typst => {
            return Err(Failure::Plain(
                "Document output needs the `mathdoc` feature's renderer, not `write_module`"
                    .to_string(),
            ));
        }
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

/// Replace one existing file through an exclusive same-directory temporary.
/// A final symlink is resolved first so formatting preserves the link itself,
/// matching the previous in-place write behavior.
#[cfg(feature = "fmtlint")]
fn write_file_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::fs::{self, OpenOptions};
    use std::io::{ErrorKind, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    let target = match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => fs::canonicalize(path)?,
        Ok(_) => path.to_path_buf(),
        Err(error) => return Err(error),
    };
    let metadata = fs::metadata(&target)?;
    if metadata.permissions().readonly() {
        return Err(std::io::Error::new(
            ErrorKind::PermissionDenied,
            "refusing to replace a read-only file",
        ));
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let (temp, mut file) = loop {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp = parent.join(format!(".flatppl-fmt-{}-{n}.tmp", std::process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            options.mode(metadata.permissions().mode() & 0o777);
        }
        match options.open(&temp) {
            Ok(file) => break (temp, file),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };

    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    drop(file);

    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        fs::Permissions::from_mode(metadata.permissions().mode() & 0o777)
    };
    #[cfg(not(unix))]
    let permissions = metadata.permissions();
    if let Err(error) = fs::set_permissions(&temp, permissions) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temp, &target) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(())
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
                    terminal_path(file)
                )));
            }
            Format::Html | Format::Markdown | Format::Latex | Format::Typst => {
                return Err(Failure::Plain(format!(
                    "`fmt` only formats FlatPPL; `{}` is a mathematical document",
                    terminal_path(file)
                )));
            }
        }
        let source = read_regular_utf8(file)
            .map_err(|e| format!("reading `{}`: {e}", terminal_path(file)))?;
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
            write_file_atomic(file, formatted.as_bytes())
                .map_err(|e| format!("writing `{}`: {e}", terminal_path(file)))?;
        }
    }

    if check && !dirty.is_empty() {
        for f in &dirty {
            eprintln!("flatppl: not canonically formatted: {}", terminal_path(f));
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
        let source = read_regular_utf8(file)
            .map_err(|e| format!("reading `{}`: {e}", terminal_path(file)))?;

        let mut cfg = base.clone();
        match inline_allows(&source) {
            Ok(rules) => {
                for rule in rules {
                    cfg.set(rule, Severity::Allow);
                }
            }
            Err(line) => {
                return Err(Failure::Diagnostic {
                    path: file.clone(),
                    source,
                    message: LEGACY_DIRECTIVE_MESSAGE.into(),
                    line,
                    span: None,
                });
            }
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
        let path = terminal_path(file);
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

    let disp = terminal_path(path);
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

/// The file-level lint directive. `#` is FlatPPL's plain-comment syntax, which
/// the parser discards, so the directive attaches to no binding and is valid at
/// any position in the file.
#[cfg(feature = "fmtlint")]
const ALLOW_DIRECTIVE: &str = "# flatppl-lint: allow ";

/// The superseded `%` spelling of the directive. `%` is doc-comment syntax, and
/// a doc-comment that attaches to no binding is invalid code, so this form is
/// rejected instead of honoured.
#[cfg(feature = "fmtlint")]
const LEGACY_DIRECTIVE: &str = "% flatppl-lint:";

#[cfg(feature = "fmtlint")]
const LEGACY_DIRECTIVE_MESSAGE: &str = concat!(
    "the lint directive must be a plain comment: write ",
    "`# flatppl-lint: allow RULE`. The `%` spelling is a doc-comment, and a ",
    "doc-comment must attach to a binding",
);

/// Scan source for file-level `# flatppl-lint: allow RULE` directives.
///
/// This is a raw-text scan, so it runs before the parse and sees a directive
/// anywhere in the file. `Err` carries the 1-based line of a `%`-spelled
/// directive.
#[cfg(feature = "fmtlint")]
fn inline_allows(source: &str) -> Result<Vec<flatppl_lint::RuleId>, usize> {
    let mut out = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let line = line.trim();
        if line.starts_with(LEGACY_DIRECTIVE) {
            return Err(idx + 1);
        }
        if let Some(rest) = line.strip_prefix(ALLOW_DIRECTIVE)
            && let Ok(rule) = rest.trim().parse()
        {
            out.push(rule);
        }
    }
    Ok(out)
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

    #[cfg(any(feature = "convert", feature = "infer", feature = "hs3"))]
    #[test]
    fn flatpir_json_breadth_is_guarded_before_deserialization() {
        let source = format!("[{}]", "[],".repeat(300_000));
        let err = read_module(Format::FlatPirJson, &source)
            .expect_err("shallow JSON breadth must be bounded");
        assert!(err.0.contains("structure exceeds"), "{err:?}");
        assert!(
            guard_json_structure(r#"{"text":"[,{\\\""}"#).is_ok(),
            "structural punctuation inside strings does not spend the budget"
        );
    }
}
