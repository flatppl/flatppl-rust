//! `flatppl-fmt` — standalone lean FlatPPL formatter + linter.
//!
//! Combined tool: `fmt` canonicalizes (over the `flatppl-syntax` printer),
//! `lint` runs the `flatppl-lint` rule set. Links only core+syntax+infer+lint
//! (no converter), so CI can format/lint without the full toolchain. All logic
//! lives in the `flatppl_cli` library; this is thin clap wiring.

use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use flatppl_cli::{report, run_fmt, run_lint};

#[derive(Parser)]
#[command(name = "flatppl-fmt", version, about = "FlatPPL formatter + linter")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Format FlatPPL files in place to canonical form.
    ///
    /// With no path (or `-`), formats stdin to stdout. `--check` writes nothing
    /// and exits non-zero if any file is not already canonical.
    Fmt(flatppl_cli::FmtArgs),
    /// Lint FlatPPL files: report style/hygiene/correctness issues.
    Lint(flatppl_cli::LintArgs),
    /// Print a shell completion script to stdout.
    Completions {
        /// Target shell.
        shell: clap_complete::Shell,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Fmt(a) => run_fmt(&a.files, a.check, a.syntax.into()),
        Command::Lint(a) => run_lint(&a.files, &a.deny, &a.warn, &a.allow, a.deny_warnings),
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "flatppl-fmt", &mut std::io::stdout());
            Ok(())
        }
    };
    report(result)
}
