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
#[cfg(any(feature = "convert", feature = "infer", feature = "fetch"))]
use std::path::PathBuf;

#[cfg(any(feature = "convert", feature = "infer"))]
use clap::ValueEnum;
use clap::{CommandFactory, Parser, Subcommand};
#[cfg(any(feature = "convert", feature = "infer", feature = "fetch"))]
use flatppl_cli::Failure;
#[cfg(feature = "convert")]
use flatppl_cli::SyntaxLevel;
use flatppl_cli::report;
#[cfg(any(feature = "infer", feature = "fetch"))]
use flatppl_cli::resolve::CliResolver;
#[cfg(feature = "convert")]
use flatppl_cli::write_module;
#[cfg(any(feature = "convert", feature = "infer"))]
use flatppl_cli::{Format, banner};
#[cfg(feature = "fmtlint")]
use flatppl_cli::{run_fmt, run_lint};
#[cfg(any(feature = "infer", feature = "fetch"))]
use flatppl_fileaccess::Location;

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
    /// `.flatpir.json`, and `.hs3.json` / `.pyhf.json` for HS3 / pyhf import).
    /// Converting to the same format canonicalizes the file. `.flatpir.json` is
    /// an alternate representation of FlatPIR (same content, including `%meta`
    /// annotations). `--from hs3` / `--from pyhf` force HS3 / pyhf import of any
    /// JSON file (and override the extension); both need the optional `hs3` feature.
    #[cfg(feature = "convert")]
    Convert {
        /// Input file (`.flatppl`, `.flatpir`, `.flatpir.json`, native HS3 JSON
        /// with `--from hs3`, or pyhf workspace JSON with `--from pyhf`). A local
        /// path — a model with remote `load_module` deps must be pre-fetched
        /// with `flatppl fetch`.
        input: PathBuf,
        /// Output file (`.flatppl`, `.flatpir`, or `.flatpir.json`)
        output: PathBuf,
        /// FlatPPL output syntax level (ignored for FlatPIR output):
        /// `full` re-applies all syntactic sugar (operators, indexing,
        /// lambdas, `:=`); `minimal` emits the lowered function-call form.
        #[arg(long, value_enum, default_value_t = SyntaxLevel::Full)]
        syntax: SyntaxLevel,
        /// Input format: `auto` infers from the file extension (`.flatppl` /
        /// `.flatpir`, and `.hs3.json` / `.pyhf.json`); `hs3` reads a native HS3
        /// JSON document (`distributions`, `likelihoods`, …); `pyhf` reads a pyhf
        /// workspace JSON document (top-level `channels` array). `hs3` / `pyhf`
        /// override the extension.
        #[arg(long, value_enum, default_value_t = FromFormat::Auto)]
        from: FromFormat,
        /// Omit the leading "do not edit" banner comment (a single line, no
        /// timestamp/user/host/platform/command — included by default). The
        /// targeted FlatPPL version is recorded in-model as `flatppl_compat`,
        /// not in the banner.
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
        /// Input file (`.flatppl` or `.flatpir`). A local path; `infer` resolves
        /// the model's `load_module` dependencies from the local cache only —
        /// run `flatppl fetch` first if it has remote deps.
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
    /// Fetch a model's remote dependencies into the local cache.
    ///
    /// Walks each input model's transitive `load_module` (and `load_data`)
    /// dependencies and downloads the `http`/`https` ones into the shared cache
    /// (`$FLATPPL_CACHEDIR`), so `convert`/`infer` — which never touch the
    /// network — can then resolve them locally. Arguments are local files;
    /// purely-local models with no remote deps need no fetch.
    #[cfg(feature = "fetch")]
    Fetch {
        /// Model files to fetch dependencies for (`.flatppl` / `.flatpir`).
        files: Vec<PathBuf>,
        /// Re-fetch dependencies even if already cached (refresh).
        #[arg(long)]
        update: bool,
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
    /// Infer the input format from the file extension: `.flatppl`, `.flatpir`,
    /// `.flatpir.json`, `.hs3.json` (HS3), `.pyhf.json` (pyhf).
    Auto,
    /// Read a native HS3 JSON document (`distributions`, `likelihoods`, …).
    /// Requires the optional `hs3` build feature.
    Hs3,
    /// Read a pyhf workspace JSON document (top-level `channels` array).
    /// Requires the optional `hs3` build feature.
    Pyhf,
}

/// Resolve `--from auto` against the input filename: `*.hs3.json` / `*.pyhf.json`
/// select the HS3 / pyhf importers (mirroring the `*.flatpir.json` convention);
/// any other name stays `Auto` (extension-inferred). An explicit `--from`
/// overrides and is returned unchanged.
#[cfg(feature = "convert")]
fn resolve_from_format(from: FromFormat, input: &Path) -> FromFormat {
    if from != FromFormat::Auto {
        return from;
    }
    let name = input.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.ends_with(".hs3.json") {
        FromFormat::Hs3
    } else if name.ends_with(".pyhf.json") {
        FromFormat::Pyhf
    } else {
        FromFormat::Auto
    }
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
        #[cfg(feature = "fetch")]
        Command::Fetch { files, update } => fetch_cmd(&files, update),
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
    // `--from auto` keys off the input name: `*.hs3.json` / `*.pyhf.json` select
    // the importers; an explicit `--from` overrides.
    let from_format = resolve_from_format(from_format, input);

    // Read the module: HS3/pyhf paths (feature-gated) or the standard
    // extension-based FlatPPL/FlatPIR path. The input is a local file — a model
    // with remote `load_module` deps is pre-fetched with `flatppl fetch`.
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
        // HS3/pyhf were selected (via `--from` or a `*.hs3.json` / `*.pyhf.json`
        // name) but this binary was built without the `hs3` feature.
        #[cfg(not(feature = "hs3"))]
        FromFormat::Hs3 | FromFormat::Pyhf => {
            return Err(Failure::Plain(
                "HS3/pyhf import is not compiled in — rebuild with `--features hs3`".into(),
            ));
        }
        // Extension-based FlatPPL / FlatPIR / FlatPIR-JSON. Check the extension
        // BEFORE reading the file so an unknown one is reported even if the file
        // is missing; for a bare `.json`, hint at the importers.
        FromFormat::Auto => {
            let from = Format::from_path(input).map_err(|mut e| {
                if input.extension().and_then(|x| x.to_str()) == Some("json") {
                    e.push_str(
                        "; for an HS3 or pyhf JSON document pass `--from hs3` / `--from pyhf` \
                         (or name it `*.hs3.json` / `*.pyhf.json`)",
                    );
                }
                e
            })?;
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
    // A leading "do not edit" banner; a format with no comment syntax
    // (`.flatpir.json`) renders nothing (`CommentStyle::None`). The targeted
    // FlatPPL version is recorded in-model as `flatppl_compat`, not here.
    if !no_header {
        text.insert_str(0, &banner(to.comment_style()));
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
    // The input is a local file; cross-module deps resolve from the local cache
    // only (run `flatppl fetch` first for remote deps).
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

    // Assemble the cross-module bundle: resolve the model's transitive
    // `load_module` dependencies from the local cache (+ local files) so the
    // engine — which stays I/O-free — can type cross-module references. A
    // cache-only resolver never touches the network; a remote dep that isn't
    // cached errors with a "run `flatppl fetch`" hint. `load_data` sources are
    // discovered but NOT resolved — inference never reads data.
    let resolver = CliResolver::cache_only();
    let in_loc = Location::Local(input.to_path_buf());
    let (bundle, _data_sources) = flatppl_cli::resolve::build_bundle(&module, &in_loc, &resolver)?;
    let diags = flatppl_infer::infer_module(&mut module, &bundle, level);
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
        // `infer` always writes FlatPIR (`;` comments).
        text.insert_str(0, &banner(Format::FlatPir.comment_style()));
    }
    fs::write(output, text)
        .map_err(|e| Failure::Plain(format!("writing `{}`: {e}", output.display())))
}

/// `flatppl fetch <file>… [--update]` — fetch each model's remote dependencies
/// into the local cache so `convert`/`infer` can resolve them offline.
#[cfg(feature = "fetch")]
fn fetch_cmd(files: &[PathBuf], update: bool) -> Result<(), Failure> {
    if files.is_empty() {
        return Err(Failure::Plain(
            "no input files — usage: `flatppl fetch <model.flatppl>…`".to_string(),
        ));
    }
    let resolver = CliResolver::fetching(update);
    let locations: Vec<Location> = files.iter().map(|f| Location::Local(f.clone())).collect();
    flatppl_cli::resolve::fetch_graph(&locations, &resolver)
}
