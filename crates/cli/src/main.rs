//! `flatppl` — the FlatPPL command-line driver.
//!
//! Thin wiring over the library crates: argument parsing, format dispatch by
//! file extension, and I/O. All conversion logic lives in the library crates —
//! `flatppl-syntax` / `flatppl-flatpir` for FlatPPL text and IR, and
//! `flatppl-hs3` for HS3 / pyhf import — the libraries are the test target, the
//! CLI is the surface. Later toolchain capabilities (infer, lower, check)
//! arrive as further subcommands.

use std::process::ExitCode;

#[cfg(any(
    feature = "convert",
    feature = "infer",
    feature = "determinize",
    feature = "stablehlo"
))]
use std::fs;
#[cfg(any(
    feature = "convert",
    feature = "infer",
    feature = "determinize",
    feature = "stablehlo"
))]
use std::path::Path;
#[cfg(any(
    feature = "convert",
    feature = "infer",
    feature = "prepare",
    feature = "determinize",
    feature = "stablehlo"
))]
use std::path::PathBuf;

#[cfg(any(feature = "convert", feature = "infer"))]
use clap::ValueEnum;
use clap::{CommandFactory, Parser, Subcommand};
#[cfg(any(
    feature = "convert",
    feature = "infer",
    feature = "prepare",
    feature = "determinize",
    feature = "stablehlo"
))]
use flatppl_cli::Failure;
#[cfg(any(
    feature = "convert",
    feature = "infer",
    feature = "determinize",
    feature = "stablehlo"
))]
use flatppl_cli::Format;
#[cfg(feature = "convert")]
use flatppl_cli::SyntaxLevel;
#[cfg(any(feature = "convert", feature = "infer"))]
use flatppl_cli::banner;
use flatppl_cli::report;
#[cfg(any(feature = "infer", feature = "prepare"))]
use flatppl_cli::resolve::CliResolver;
#[cfg(feature = "convert")]
use flatppl_cli::write_module;
#[cfg(feature = "fmtlint")]
use flatppl_cli::{run_fmt, run_lint};
#[cfg(any(feature = "infer", feature = "prepare"))]
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
        /// with `flatppl prepare`.
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
        /// run `flatppl prepare` first if it has remote deps.
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
    /// Prepare a model for offline use: fetch its remote dependencies into the cache.
    ///
    /// Walks each input model's transitive `load_module` (and `load_data`)
    /// dependencies and downloads the `http`/`https` ones into the shared cache
    /// (`$FLATPPL_CACHEDIR`), so `convert`/`infer` — which never touch the
    /// network — can then resolve them locally. Arguments are local files;
    /// purely-local models with no remote deps need no fetch.
    #[cfg(feature = "prepare")]
    Prepare {
        /// Model files to fetch dependencies for (`.flatppl` / `.flatpir`).
        files: Vec<PathBuf>,
        /// Re-fetch dependencies even if already cached (refresh).
        #[arg(long)]
        update: bool,
    },
    /// Determinize a FlatPPL model to the deterministic FlatPDL profile
    /// (eliminate the measure layer). Refuses (exit 3) any construct it cannot
    /// legalize, per refuse-don't-mislower.
    #[cfg(feature = "determinize")]
    Determinize {
        /// Input FlatPPL file.
        input: PathBuf,
        /// Output file (`.flatppl`); stdout if omitted.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Requested-output binding name to keep (repeatable). Bindings not
        /// reachable from any `--keep` root are removed. With no `--keep`, all
        /// bindings are kept (backward-compatible).
        #[arg(long = "keep")]
        keep: Vec<String>,
    },
    /// Emit textual StableHLO for a FlatPPL model.
    ///
    /// Always determinizes first (eliminate the measure layer), then prints
    /// StableHLO for the requested `--mode`. Refuses (exit 3) any construct
    /// the determiniser or emitter cannot legalize, per refuse-don't-mislower;
    /// an unrecognized `--mode` exits 2.
    #[cfg(feature = "stablehlo")]
    Stablehlo {
        /// Input FlatPPL file.
        input: PathBuf,
        /// Computation to emit: `logdensity` or `sample`.
        #[arg(long, default_value = "logdensity")]
        mode: String,
        /// Output file (`.mlir`); stdout if omitted.
        #[arg(short, long)]
        output: Option<PathBuf>,
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
        #[cfg(feature = "prepare")]
        Command::Prepare { files, update } => prepare_cmd(&files, update),
        #[cfg(feature = "determinize")]
        Command::Determinize {
            input,
            output,
            keep,
        } => determinize_cmd(&input, output.as_deref(), &keep),
        #[cfg(feature = "stablehlo")]
        Command::Stablehlo {
            input,
            mode,
            output,
        } => stablehlo_cmd(&input, &mode, output.as_deref()),
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
    // with remote `load_module` deps is pre-fetched with `flatppl prepare`.
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
    // only (run `flatppl prepare` first for remote deps).
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
    // cached errors with a "run `flatppl prepare`" hint. `load_data` sources are
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

/// `flatppl prepare <file>… [--update]` — fetch each model's remote dependencies
/// into the local cache so `convert`/`infer` can resolve them offline.
#[cfg(feature = "prepare")]
fn prepare_cmd(files: &[PathBuf], update: bool) -> Result<(), Failure> {
    if files.is_empty() {
        return Err(Failure::Plain(
            "no input files — usage: `flatppl prepare <model.flatppl>…`".to_string(),
        ));
    }
    let resolver = CliResolver::fetching(update);
    let locations: Vec<Location> = files.iter().map(|f| Location::Local(f.clone())).collect();
    flatppl_cli::resolve::fetch_graph(&locations, &resolver)
}

/// Parse `input` and assemble its cross-module bundle, then run the type/shape
/// trace over it — the common front end for `determinize`/`stablehlo` before
/// legalizing to FlatPDL. Mirrors `infer_cmd`'s bundle-building (same
/// cache-only resolver, same `Level::Shape`) so `load_module` refs resolve
/// identically across verbs; surfaces inference errors as a `Failure` so a
/// mistyped model refuses loudly instead of being lowered blind.
#[cfg(any(feature = "determinize", feature = "stablehlo"))]
fn load_and_infer(
    input: &Path,
) -> Result<(flatppl_core::Module, flatppl_infer::ModuleBundle), Failure> {
    let source =
        fs::read_to_string(input).map_err(|e| format!("reading `{}`: {e}", input.display()))?;
    let mut module = match flatppl_cli::read_module(Format::FlatPpl, &source) {
        Ok(m) => m,
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
    // determiniser can graft in cross-module measure refs, same as `infer`.
    let resolver = CliResolver::cache_only();
    let in_loc = Location::Local(input.to_path_buf());
    let (bundle, _data_sources) = flatppl_cli::resolve::build_bundle(&module, &in_loc, &resolver)?;
    let diags = flatppl_infer::infer_module(&mut module, &bundle, flatppl_infer::Level::Shape);
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

    Ok((module, bundle))
}

/// `flatppl determinize <in.flatppl> [-o out] [--keep name]…` — legalize a
/// FlatPPL model to the deterministic FlatPDL profile (eliminate the measure
/// layer), printing canonical FlatPPL syntax to `output`/stdout. With one or
/// more `--keep <name>`, only bindings reachable from those requested-output
/// roots survive (root-based DCE, Buffy #263 Pass 4-A); with none, every
/// binding is kept (unchanged behavior). Refuses (exit 3, via
/// `Failure::Refuse`) any construct the determiniser cannot legalize.
#[cfg(feature = "determinize")]
fn determinize_cmd(input: &Path, output: Option<&Path>, keep: &[String]) -> Result<(), Failure> {
    let (mut module, bundle) = load_and_infer(input)?;
    let syms: Vec<flatppl_core::Symbol> = keep.iter().map(|name| module.intern(name)).collect();
    let roots = if syms.is_empty() {
        None
    } else {
        Some(syms.as_slice())
    };
    let lowered =
        flatppl_determinizer::determinize_with_roots(&module, &bundle, roots).map_err(|e| {
            Failure::Refuse(format!(
                "determinize: refuse {} (node {:?}): {}",
                e.construct, e.node, e.reason
            ))
        })?;
    let rendered = flatppl_syntax::print(&lowered);
    match output {
        Some(path) => fs::write(path, rendered)
            .map_err(|e| Failure::Plain(format!("writing `{}`: {e}", path.display())))?,
        None => print!("{rendered}"),
    }
    Ok(())
}

/// `flatppl stablehlo <in.flatppl> [--mode logdensity|sample] [-o out]` —
/// determinize a FlatPPL model to FlatPDL, then emit textual StableHLO for
/// `mode`, printing to `output`/stdout. An unrecognized `--mode` exits 2
/// (`Failure::Usage`); refuses (exit 3, via `Failure::Refuse`) any construct
/// the determiniser or emitter cannot legalize — the same exit-code
/// convention as `determinize`.
#[cfg(feature = "stablehlo")]
fn stablehlo_cmd(input: &Path, mode: &str, output: Option<&Path>) -> Result<(), Failure> {
    let (module, bundle) = load_and_infer(input)?;

    let mode = match mode {
        "logdensity" => flatppl_stablehlo::Mode::LogDensity,
        "sample" => flatppl_stablehlo::Mode::Sample,
        other => {
            return Err(Failure::Usage(format!(
                "stablehlo: unrecognized `--mode {other}` (expected `logdensity` or `sample`)"
            )));
        }
    };

    // The `inputs`/`outputs` compilation ABI (design doc
    // `docs/superpowers/specs/2026-07-17-inputs-outputs-abi-design.md`) is
    // REQUIRED, both modes: the last-public-binding query heuristic has been
    // removed. DCE roots on both present reserved names (so the outputs'
    // backward cone AND the declared inputs survive — an unused declared input
    // is still a stable ABI arg, not pruned); the emitter reads the ABI itself
    // off the determinized module (`modes::read_abi`). A surface model that
    // declares neither reserved binding is refused here — the compiled
    // function's arguments and results must be designated explicitly.
    let abi_syms: Vec<flatppl_core::Symbol> = ["inputs", "outputs"]
        .iter()
        .filter_map(|name| {
            module
                .public_bindings()
                .find(|(_, b)| module.resolve(b.name) == *name)
                .map(|(_, b)| b.name)
        })
        .collect();

    if abi_syms.is_empty() {
        return Err(Failure::Refuse(
            "stablehlo: no inputs/outputs ABI declared; the last-public-binding \
             query heuristic has been removed — declare `inputs = (…)` and \
             `outputs = (…)` (typically in a query module) to designate the \
             compiled function's arguments and results"
                .to_string(),
        ));
    }
    let roots = Some(abi_syms);

    // Compile-time shape pins for `load_data` ABI inputs (design doc
    // "load_data — shape, not values"): read each `load_data` binding named in
    // `inputs` for its LENGTH only, so the emitter types it `tensor<N×f32>`
    // rather than an unusable `tensor<?×f32>`. The values are NOT read here —
    // they are the runtime argument, never baked.
    let input_shapes = {
        let input_names = abi_input_names(&module);
        let in_loc = Location::Local(input.to_path_buf());
        let resolver = CliResolver::cache_only();
        let mut shapes: std::collections::HashMap<String, Vec<u64>> =
            std::collections::HashMap::new();
        for (name, loc) in flatppl_cli::resolve::load_data_bindings_of(&module, &in_loc) {
            if input_names.iter().any(|n| n == &name) {
                let path = resolver.resolve_path(&loc)?;
                let n = load_data_vector_len(&path)?;
                shapes.insert(name, vec![n as u64]);
            }
        }
        shapes
    };

    let lowered = flatppl_determinizer::determinize_with_roots(&module, &bundle, roots.as_deref())
        .map_err(|e| {
            Failure::Refuse(format!(
                "determinize: refuse {} (node {:?}): {}",
                e.construct, e.node, e.reason
            ))
        })?;
    let opts = flatppl_stablehlo::EmitOptions {
        input_shapes,
        ..Default::default()
    };
    let rendered = flatppl_stablehlo::emit(&lowered, mode, &opts)
        .map_err(|e| Failure::Refuse(e.to_string()))?;
    match output {
        Some(path) => fs::write(path, rendered)
            .map_err(|e| Failure::Plain(format!("writing `{}`: {e}", path.display())))?,
        None => print!("{rendered}"),
    }
    Ok(())
}

/// The binding names listed in a surface model's reserved `inputs` binding, in
/// declared order — the elements of an `inputs = (a, b, …)` tuple (or a single
/// `inputs = a`) resolved through `(%ref self x)` to their names. Used by
/// [`stablehlo_cmd`] to decide which `load_data` bindings need a compile-time
/// shape pin (a `load_data` NOT in `inputs` is not an ABI argument). A
/// non-`ref` element is skipped (the StableHLO emitter's own `read_abi` applies
/// the authoritative checks).
#[cfg(feature = "stablehlo")]
fn abi_input_names(module: &flatppl_core::Module) -> Vec<String> {
    use flatppl_core::{CallHead, Node, Ref, RefNs};
    let Some((_, b)) = module
        .public_bindings()
        .find(|(_, b)| module.resolve(b.name) == "inputs")
    else {
        return Vec::new();
    };
    let elems: Vec<flatppl_core::NodeId> = match module.node(b.rhs) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if module.resolve(s) == "tuple") => {
            c.args.to_vec()
        }
        _ => vec![b.rhs],
    };
    elems
        .iter()
        .filter_map(|&e| match module.node(e) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => Some(module.resolve(*name).to_string()),
            _ => None,
        })
        .collect()
}

/// The element count of a 1-D `load_data` vector (spec §07), read from the
/// resolved delimited file (CSV/WSV) — used ONLY for the compile-time shape pin
/// of a `load_data` ABI input (`tensor<N×f32>`); the values themselves are
/// never read here (they are the runtime argument, never baked). Mirrors the
/// reference engine's delimited single-column rule (`flatppl-js`
/// `dataload.ts`): skip blank and `#`-comment lines, treat the first remaining
/// row as the column header, and count the remaining data rows.
#[cfg(feature = "stablehlo")]
fn load_data_vector_len(path: &Path) -> Result<usize, Failure> {
    // Dispatch by extension, mirroring the reference engine (`flatppl-js`
    // `dataload.ts`): only the delimited TEXT formats (`.csv` comma-separated,
    // `.wsv` whitespace-separated) are supported for the compile-time shape pin.
    // Any other format (`.json`, Arrow IPC, or no/unknown extension) is REFUSED
    // rather than blindly line-counted — a blind count would silently mis-shape
    // the emitted `tensor<N×f32>` argument (refuse, don't mis-lower). Pinning a
    // `.json`/Arrow length is a follow-up.
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if !matches!(ext.as_deref(), Some("csv") | Some("wsv")) {
        return Err(Failure::Refuse(format!(
            "load_data shape pin: unsupported format {} for `{}` — a `load_data` ABI input's \
             shape can be pinned only from `.csv` / `.wsv` (matching the reference engine; \
             `.json` / Arrow IPC not yet supported)",
            ext.map(|e| format!("`.{e}`"))
                .unwrap_or_else(|| "(no extension)".into()),
            path.display()
        )));
    }
    let text = fs::read_to_string(path).map_err(|e| {
        Failure::Plain(format!(
            "reading load_data source `{}` for its shape: {e}",
            path.display()
        ))
    })?;
    let data_rows = text
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .count();
    // Drop the header row (present when there is at least one line).
    Ok(data_rows.saturating_sub(1))
}
