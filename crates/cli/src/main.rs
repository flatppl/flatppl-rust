//! `flatppl` — the FlatPPL command-line driver.
//!
//! Thin wiring over the library crates: argument parsing, format dispatch by
//! file extension, and I/O. All conversion logic lives in the library crates —
//! `flatppl-syntax` / `flatppl-flatpir` for FlatPPL text and IR, and
//! `flatppl-hs3` for HS3 / pyhf import — the libraries are the test target, the
//! CLI is the surface. Later toolchain capabilities (infer, lower, check)
//! arrive as further subcommands.

use std::process::ExitCode;

#[cfg(any(feature = "convert", feature = "infer"))]
use std::fs;
#[cfg(any(feature = "convert", feature = "infer"))]
use std::path::Path;
#[cfg(any(feature = "convert", feature = "infer"))]
use std::path::PathBuf;

#[cfg(any(feature = "convert", feature = "infer"))]
use clap::ValueEnum;
use clap::{CommandFactory, Parser, Subcommand};
#[cfg(feature = "convert")]
use flatppl_cli::SyntaxLevel;
use flatppl_cli::report;
#[cfg(feature = "convert")]
use flatppl_cli::write_module;
#[cfg(any(feature = "convert", feature = "infer"))]
use flatppl_cli::{Failure, Format, Provenance};
#[cfg(feature = "fmtlint")]
use flatppl_cli::{run_fmt, run_lint};

#[derive(Parser)]
#[command(name = "flatppl", version, about = "FlatPPL toolchain driver")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert between FlatPPL, FlatPIR, and the FlatPIR JSON encoding.
    ///
    /// Formats are inferred from the file extensions (`.flatppl` / `.flatpir` /
    /// `.flatpir.json`). Converting to the same format canonicalizes the file.
    /// `.flatpir.json` is an alternate representation of FlatPIR (same content,
    /// including `%meta` annotations). Use `--from hs3` or `--from pyhf` to
    /// import a JSON file instead.
    #[cfg(feature = "convert")]
    Convert {
        /// Input file (`.flatppl`, `.flatpir`, `.flatpir.json`, native HS3 JSON
        /// with `--from hs3`, or pyhf workspace JSON with `--from pyhf`)
        input: PathBuf,
        /// Output file (`.flatppl`, `.flatpir`, or `.flatpir.json`)
        output: PathBuf,
        /// FlatPPL output syntax level (ignored for FlatPIR output):
        /// `full` re-applies all syntactic sugar (operators, indexing,
        /// lambdas, `:=`); `minimal` emits the lowered function-call form.
        #[arg(long, value_enum, default_value_t = SyntaxLevel::Full)]
        syntax: SyntaxLevel,
        /// Input format: `auto` infers from the file extension (`.flatppl` /
        /// `.flatpir`); `hs3` reads a native HS3 JSON document
        /// (`distributions`, `likelihoods`, …); `pyhf` reads a pyhf workspace
        /// JSON document (top-level `channels` array).
        #[arg(long, value_enum, default_value_t = FromFormat::Auto)]
        from: FromFormat,
        /// Omit the leading provenance header comment. The header (records
        /// when/from-what/by-whom the file was generated) is included by
        /// default; pass this — or set `SOURCE_DATE_EPOCH` — for reproducible
        /// byte-output.
        #[arg(long)]
        no_header: bool,
    },
    /// Infer types and phases; emit annotated FlatPIR.
    ///
    /// Runs the type/shape/phase trace over the module and writes FlatPIR
    /// with `(%meta <type> <phase>)` annotations (spec §11). Inference notes
    /// (honest `%deferred` gaps) go to stderr; inference errors (cycles,
    /// unresolved names) fail the command.
    #[cfg(feature = "infer")]
    Infer {
        /// Input file (`.flatppl` or `.flatpir`)
        input: PathBuf,
        /// Output file (`.flatpir` — FlatPPL cannot carry annotations)
        output: PathBuf,
        /// Inference level — a hierarchy, each including the previous:
        /// `phase` classifies bindings only (types stay `%deferred`);
        /// `type` adds structural types (literal dims only); `valueset`
        /// adds the strongest known value set per node (the third `%meta`
        /// slot); `normalization` adds total-mass classes on measure and
        /// kernel types; `shape` also resolves fixed-phase dims (`iid`
        /// counts, distribution lengths).
        #[arg(long, value_enum, default_value_t = InferLevel::Shape)]
        level: InferLevel,
        /// Omit the leading provenance header comment (see `convert --no-header`).
        #[arg(long)]
        no_header: bool,
    },
    /// Print a shell completion script to stdout.
    ///
    /// Covers every subcommand and flag. Install, e.g.:
    /// `flatppl completions bash > /etc/bash_completion.d/flatppl`,
    /// `flatppl completions zsh > ~/.zfunc/_flatppl`, or
    /// `flatppl completions fish > ~/.config/fish/completions/flatppl.fish`.
    Completions {
        /// Target shell.
        shell: clap_complete::Shell,
    },
    /// Format FlatPPL files in place to canonical form.
    ///
    /// With no path (or `-`), formats stdin to stdout. `--check` writes nothing
    /// and exits non-zero if any file is not already canonical.
    #[cfg(feature = "fmtlint")]
    Fmt(flatppl_cli::FmtArgs),
    /// Lint FlatPPL files: report style/hygiene/correctness issues.
    #[cfg(feature = "fmtlint")]
    Lint(flatppl_cli::LintArgs),
}

/// CLI mirror of [`flatppl_infer::Level`] (the library stays clap-free).
#[cfg(feature = "infer")]
#[derive(Clone, Copy, ValueEnum)]
enum InferLevel {
    Phase,
    Type,
    Valueset,
    Normalization,
    Shape,
}

#[cfg(feature = "infer")]
impl From<InferLevel> for flatppl_infer::Level {
    fn from(level: InferLevel) -> Self {
        match level {
            InferLevel::Phase => flatppl_infer::Level::Phase,
            InferLevel::Type => flatppl_infer::Level::Type,
            InferLevel::Valueset => flatppl_infer::Level::Valueset,
            InferLevel::Normalization => flatppl_infer::Level::Normalization,
            InferLevel::Shape => flatppl_infer::Level::Shape,
        }
    }
}

/// Input format selector for `--from`.
#[cfg(feature = "convert")]
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FromFormat {
    /// Infer the input format from the file extension (`.flatppl` / `.flatpir`).
    Auto,
    /// Read a native HS3 JSON document (`distributions`, `likelihoods`, …).
    Hs3,
    /// Read a pyhf workspace JSON document (top-level `channels` array).
    Pyhf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        #[cfg(feature = "convert")]
        Command::Convert {
            input,
            output,
            syntax,
            from,
            no_header,
        } => convert(&input, &output, syntax.into(), from, no_header),
        #[cfg(feature = "infer")]
        Command::Infer {
            input,
            output,
            level,
            no_header,
        } => infer_cmd(&input, &output, level.into(), no_header),
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "flatppl", &mut std::io::stdout());
            Ok(())
        }
        #[cfg(feature = "fmtlint")]
        Command::Fmt(a) => run_fmt(&a.files, a.check, a.syntax.into()),
        #[cfg(feature = "fmtlint")]
        Command::Lint(a) => run_lint(&a.files, &a.deny, &a.warn, &a.allow, a.deny_warnings),
    };
    report(result)
}

/// Print a note when an HS3/pyhf document carried an `analyses` block: it is
/// not imported by `convert` (inference configuration is out of scope), and the
/// user should know part of the document was skipped.
///
// NOTE (L3): this re-parses `source` to inspect a single top-level key, after
// `read_hs3`/`read_pyhf` already parsed it. The clean fix is for the importer
// to surface an analyses-present flag from its existing parse (in `convert.rs`);
// that's a cross-crate change outside this CLI's scope, so the extra parse is
// left in place (Low severity, accepted by review). The cheaper `&Value`
// alternative would require adding a `serde_json` dependency to this crate's
// `Cargo.toml`, which is likewise out of scope here.
#[cfg(feature = "hs3")]
fn note_dropped_analyses(source: &str) {
    if flatppl_hs3::document_has_analyses(source) {
        eprintln!(
            "flatppl: note: the input's `analyses` block was not imported \
             (inference configuration is out of scope for `convert`)"
        );
    }
}

#[cfg(feature = "convert")]
fn convert(
    input: &Path,
    output: &Path,
    syntax: flatppl_syntax::Syntax,
    from_format: FromFormat,
    no_header: bool,
) -> Result<(), Failure> {
    let to = Format::from_path(output)?;

    // Read the module: HS3/pyhf paths (feature-gated) or the standard
    // extension-based FlatPPL/FlatPIR path.
    #[cfg_attr(not(feature = "fmtlint"), allow(unused_mut))]
    let mut module = match from_format {
        #[cfg(feature = "hs3")]
        FromFormat::Hs3 => {
            let source = fs::read_to_string(input)
                .map_err(|e| format!("reading `{}`: {e}", input.display()))?;
            let module =
                flatppl_hs3::read_hs3(&source).map_err(|e| Failure::Plain(format!("hs3: {e}")))?;
            note_dropped_analyses(&source);
            module
        }
        #[cfg(feature = "hs3")]
        FromFormat::Pyhf => {
            let source = fs::read_to_string(input)
                .map_err(|e| format!("reading `{}`: {e}", input.display()))?;
            let module = flatppl_hs3::read_pyhf(&source)
                .map_err(|e| Failure::Plain(format!("pyhf: {e}")))?;
            note_dropped_analyses(&source);
            module
        }
        // `Auto` (and, when the feature is absent, the `Hs3`/`Pyhf` arms that
        // can never be reached) falls through to extension inference.
        // Check the input extension BEFORE reading the file so that an
        // unknown extension is reported even when the file does not exist.
        _ => {
            let from = Format::from_path(input)?;
            let source = fs::read_to_string(input)
                .map_err(|e| format!("reading `{}`: {e}", input.display()))?;
            match flatppl_cli::read_module(from, &source) {
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
            }
        }
    };

    let mut text = write_module(to, &module, syntax)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    // The provenance header is a leading comment; skip it for a format that has
    // no comment syntax (`.flatpir.json` — JSON can't carry it).
    if let (false, Some(comment)) = (no_header, to.line_comment()) {
        let from_label = match from_format {
            FromFormat::Hs3 => "HS3 JSON",
            FromFormat::Pyhf => "pyhf workspace JSON",
            FromFormat::Auto => Format::from_path(input)
                .map(Format::describe)
                .unwrap_or("FlatPPL/FlatPIR"),
        };
        let header = Provenance {
            converted_from: from_label,
            source: input,
            generator: "convert",
        }
        .header(comment);
        text.insert_str(0, &header);
    }
    // Surface lint findings on generated FlatPPL (advisory — the file is still
    // written). The output is already canonically formatted by the printer.
    #[cfg(feature = "fmtlint")]
    if matches!(to, Format::FlatPpl) {
        flatppl_cli::lint_generated(&mut module, output);
    }
    fs::write(output, text)
        .map_err(|e| Failure::Plain(format!("writing `{}`: {e}", output.display())))
}

/// `flatppl infer <in> <out.flatpir>` — run the type/phase trace, report
/// diagnostics, write annotated FlatPIR.
#[cfg(feature = "infer")]
fn infer_cmd(
    input: &Path,
    output: &Path,
    level: flatppl_infer::Level,
    no_header: bool,
) -> Result<(), Failure> {
    let from = Format::from_path(input)?;
    if !matches!(Format::from_path(output)?, Format::FlatPir) {
        return Err(Failure::Plain(format!(
            "`infer` writes annotated FlatPIR; `{}` must have a `.flatpir` extension \
             (FlatPPL cannot carry %meta annotations)",
            output.display()
        )));
    }
    let source =
        fs::read_to_string(input).map_err(|e| format!("reading `{}`: {e}", input.display()))?;
    let mut module = match flatppl_cli::read_module(from, &source) {
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

    let diags = flatppl_infer::infer_with(&mut module, level);
    let mut errors = 0u32;
    for d in &diags {
        match d.severity {
            flatppl_infer::Severity::Error => {
                errors += 1;
                eprintln!("error: {}", d.message);
            }
            flatppl_infer::Severity::Note => eprintln!("note: {}", d.message),
        }
    }
    if errors > 0 {
        return Err(Failure::Plain(format!(
            "inference found {errors} error(s) in `{}`",
            input.display()
        )));
    }

    let mut text = flatppl_flatpir::write(&module);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    if !no_header {
        // `infer` always writes FlatPIR (`;` comments); records the input format.
        let header = Provenance {
            converted_from: from.describe(),
            source: input,
            generator: "infer",
        }
        .header(
            Format::FlatPir
                .line_comment()
                .expect("FlatPIR has a line comment"),
        );
        text.insert_str(0, &header);
    }
    fs::write(output, text)
        .map_err(|e| Failure::Plain(format!("writing `{}`: {e}", output.display())))
}
