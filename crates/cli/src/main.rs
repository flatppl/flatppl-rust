//! `flatppl` — the FlatPPL command-line driver.
//!
//! Thin wiring over the library crates: argument parsing, format dispatch by
//! file extension, and I/O. All conversion logic lives in `flatppl-syntax` /
//! `flatppl-flatpir` — the libraries are the test target, the CLI is the
//! surface. Later toolchain capabilities (infer, lower, check) arrive as
//! further subcommands.

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ariadne::{Config, Label, Report, ReportKind, Source};
use clap::{Parser, Subcommand, ValueEnum};
use flatppl_core::Module;

#[derive(Parser)]
#[command(name = "flatppl", version, about = "FlatPPL toolchain driver")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert between FlatPPL and FlatPIR.
    ///
    /// Formats are inferred from the file extensions (`.flatppl` /
    /// `.flatpir`). Converting to the same format canonicalizes the file.
    Convert {
        /// Input file (`.flatppl` or `.flatpir`)
        input: PathBuf,
        /// Output file (`.flatppl` or `.flatpir`)
        output: PathBuf,
        /// FlatPPL output syntax level (ignored for FlatPIR output):
        /// `full` re-applies all syntactic sugar (operators, indexing,
        /// lambdas, `:=`); `minimal` emits the lowered function-call form.
        #[arg(long, value_enum, default_value_t = SyntaxLevel::Full)]
        syntax: SyntaxLevel,
    },
}

/// CLI mirror of [`flatppl_syntax::Syntax`] (the library stays clap-free).
#[derive(Clone, Copy, ValueEnum)]
enum SyntaxLevel {
    Full,
    Minimal,
}

impl From<SyntaxLevel> for flatppl_syntax::Syntax {
    fn from(level: SyntaxLevel) -> Self {
        match level {
            SyntaxLevel::Full => flatppl_syntax::Syntax::Full,
            SyntaxLevel::Minimal => flatppl_syntax::Syntax::Minimal,
        }
    }
}

/// A serialization format, inferred from a filename extension.
#[derive(Clone, Copy)]
enum Format {
    FlatPpl,
    FlatPir,
}

impl Format {
    fn from_path(path: &Path) -> Result<Format, String> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("flatppl") => Ok(Format::FlatPpl),
            Some("flatpir") => Ok(Format::FlatPir),
            Some(other) => Err(format!(
                "unsupported file extension `.{other}` for `{}` \
                 (expected `.flatppl` or `.flatpir`)",
                path.display()
            )),
            None => Err(format!(
                "cannot infer a format for `{}`: no file extension \
                 (expected `.flatppl` or `.flatpir`)",
                path.display()
            )),
        }
    }
}

/// Why a command failed: a plain one-line message (I/O, usage), or a parse
/// diagnostic rendered as a source-annotated report.
enum Failure {
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Convert {
            input,
            output,
            syntax,
        } => convert(&input, &output, syntax.into()),
    };
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

/// Print a source-annotated error report to stderr. The span is the error's
/// own when it carries one; a line-only error highlights its whole line; an
/// unlocalized error degrades to a plain message.
fn render_diagnostic(
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

fn convert(input: &Path, output: &Path, syntax: flatppl_syntax::Syntax) -> Result<(), Failure> {
    let from = Format::from_path(input)?;
    let to = Format::from_path(output)?;
    let source =
        fs::read_to_string(input).map_err(|e| format!("reading `{}`: {e}", input.display()))?;

    let module = match read_module(from, &source) {
        Ok(module) => module,
        Err((message, line, span)) => {
            return Err(Failure::Diagnostic {
                path: input.to_path_buf(),
                source,
                message,
                line,
                span,
            });
        }
    };
    let mut text = write_module(to, &module, syntax);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    fs::write(output, text)
        .map_err(|e| Failure::Plain(format!("writing `{}`: {e}", output.display())))
}

/// Parse/read `source`; an error comes back as `(message, line, span)` in a
/// format-agnostic shape for [`render_diagnostic`].
type ReadError = (String, usize, Option<(usize, usize)>);

fn read_module(format: Format, source: &str) -> Result<Module, ReadError> {
    fn widen(span: Option<(u32, u32)>) -> Option<(usize, usize)> {
        span.map(|(s, e)| (s as usize, e as usize))
    }
    match format {
        Format::FlatPpl => {
            flatppl_syntax::parse(source).map_err(|e| (e.message, e.line as usize, widen(e.span)))
        }
        Format::FlatPir => {
            flatppl_flatpir::read(source).map_err(|e| (e.message, e.line, widen(e.span)))
        }
    }
}

fn write_module(format: Format, module: &Module, syntax: flatppl_syntax::Syntax) -> String {
    match format {
        Format::FlatPpl => flatppl_syntax::print_with(module, syntax),
        Format::FlatPir => flatppl_flatpir::write(module),
    }
}
