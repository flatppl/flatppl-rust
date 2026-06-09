//! `flatppl` — the FlatPPL command-line driver.
//!
//! Thin wiring over the library crates: argument parsing, format dispatch by
//! file extension, and I/O. All conversion logic lives in `flatppl-syntax` /
//! `flatppl-flatpir` — the libraries are the test target, the CLI is the
//! surface. Later toolchain capabilities (infer, lower, check) arrive as
//! further subcommands.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

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
        Err(msg) => {
            eprintln!("flatppl: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn convert(input: &Path, output: &Path, syntax: flatppl_syntax::Syntax) -> Result<(), String> {
    let from = Format::from_path(input)?;
    let to = Format::from_path(output)?;
    let source =
        fs::read_to_string(input).map_err(|e| format!("reading `{}`: {e}", input.display()))?;

    let module = read_module(from, &source, input)?;
    let mut text = write_module(to, &module, syntax);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    fs::write(output, text).map_err(|e| format!("writing `{}`: {e}", output.display()))
}

fn read_module(format: Format, source: &str, path: &Path) -> Result<Module, String> {
    match format {
        Format::FlatPpl => {
            flatppl_syntax::parse(source).map_err(|e| format!("{}: {e}", path.display()))
        }
        Format::FlatPir => {
            flatppl_flatpir::read(source).map_err(|e| format!("{}: {e}", path.display()))
        }
    }
}

fn write_module(format: Format, module: &Module, syntax: flatppl_syntax::Syntax) -> String {
    match format {
        Format::FlatPpl => flatppl_syntax::print_with(module, syntax),
        Format::FlatPir => flatppl_flatpir::write(module),
    }
}
