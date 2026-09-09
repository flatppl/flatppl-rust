//! Standalone documents from the same statement trees as the MathML viewer.
//!
//! GitHub Markdown keeps Markdown doc-comments. TeX and Typst quote doc-comment
//! source as text: markup and embedded math remain readable, without executing
//! author commands or translating one math language into another.

use std::fmt::Write;

use flatppl_core::{Markup, Module};

use crate::ast::{Math, Rel, Statement};
use crate::document::{doc_text, split_title};
use crate::lower::Lowerer;
use crate::render::Rendering;
use crate::{tex, typst};

/// GitHub Markdown with TeX display math and the full data and notation appendices.
pub fn github_markdown(module: &Module, rendering: &Rendering, title: &str) -> String {
    document(module, rendering, title, Format::Markdown)
}

/// A standalone LaTeX document. Compile with LuaLaTeX or XeLaTeX: identifiers
/// and literal source text may contain Unicode, supported by `unicode-math`.
pub fn latex(module: &Module, rendering: &Rendering, title: &str) -> String {
    document(module, rendering, title, Format::Latex)
}

/// A standalone Typst document, using native math syntax throughout.
pub fn typst(module: &Module, rendering: &Rendering, title: &str) -> String {
    document(module, rendering, title, Format::Typst)
}

#[derive(Clone, Copy)]
enum Format {
    Markdown,
    Latex,
    Typst,
}

impl Format {
    fn heading(self, out: &mut String, text: &str, title: bool) {
        match self {
            Self::Markdown => {
                let _ = writeln!(
                    out,
                    "{} {}\n",
                    if title { "#" } else { "##" },
                    markdown_text(text)
                );
            }
            Self::Latex => {
                let _ = writeln!(
                    out,
                    "\\{}*{{{}}}\n",
                    if title { "section" } else { "subsection" },
                    tex::escape(text)
                );
            }
            Self::Typst => {
                let _ = writeln!(
                    out,
                    "#heading(level: {})[#{}]\n",
                    if title { 1 } else { 2 },
                    typst::quote(text)
                );
            }
        }
    }

    fn prose(self, out: &mut String, text: &str, markup: Option<Markup>) {
        if text.trim().is_empty() {
            return;
        }
        match self {
            Self::Markdown if markup == Some(Markup::Md) => {
                let _ = writeln!(out, "{text}\n");
            }
            Self::Markdown if markup == Some(Markup::Typ) => {
                // A longer fence cannot be closed by the author's source.
                let fence = "`".repeat(
                    text.split(|c| c != '`')
                        .map(str::len)
                        .max()
                        .unwrap_or(0)
                        .max(2)
                        + 1,
                );
                let _ = writeln!(out, "{fence}typst\n{text}\n{fence}\n");
            }
            Self::Markdown => {
                let _ = writeln!(out, "{}\n", markdown_text(text));
            }
            Self::Latex => {
                // Paragraphs survive, while every source command remains text.
                for paragraph in text.split("\n\n") {
                    let _ = writeln!(out, "{}\n", tex::escape(paragraph));
                }
            }
            Self::Typst => {
                let _ = writeln!(out, "#text({})\n", typst::quote(text));
            }
        }
    }

    fn equations(self, out: &mut String, rows: &[(&Statement, Option<String>)]) {
        if rows.is_empty() {
            return;
        }
        let (start, separator, end) = match self {
            Self::Markdown => (
                "```math\n\\begin{aligned}\n",
                " \\\\\n",
                "\n\\end{aligned}\n```\n\n",
            ),
            Self::Latex => (
                "\\[\n\\begin{aligned}\n",
                " \\\\\n",
                "\n\\end{aligned}\n\\]\n\n",
            ),
            Self::Typst => ("$\n", " \\\n", "\n$\n\n"),
        };
        out.push_str(start);
        for (i, (stmt, annotation)) in rows.iter().enumerate() {
            if i > 0 {
                out.push_str(separator);
            }
            let (lhs, rel, rhs) = match self {
                Self::Markdown | Self::Latex => (
                    tex::expr(&stmt.lhs),
                    tex::relation(stmt.rel),
                    tex::expr(&stmt.rhs),
                ),
                Self::Typst => (
                    typst::expr(&stmt.lhs),
                    typst::relation(stmt.rel),
                    typst::expr(&stmt.rhs),
                ),
            };
            let _ = write!(out, "{lhs} &{rel} {rhs}");
            if let Some(annotation) = annotation {
                match self {
                    Self::Markdown | Self::Latex => {
                        let _ = write!(out, " \\qquad \\text{{{}}}", tex::escape(annotation));
                    }
                    Self::Typst => {
                        let _ = write!(out, " quad #text({})", typst::quote(annotation));
                    }
                }
            }
        }
        out.push_str(end);
    }

    fn legend(self, out: &mut String, entry: &crate::notation::NotationEntry) {
        let form = &entry.form;
        // The note's mathematics goes through this format's own math printer;
        // its text through this format's escaping. No code spelling: the
        // legend explains the notation, not the source.
        match self {
            Self::Markdown => {
                let _ = writeln!(out, "```math\n{}\n```\n", tex::expr(form));
                let note = entry.note_with(markdown_text, |m| format!("${}$", tex::expr(m)));
                let _ = writeln!(out, "{note}\n");
            }
            Self::Latex => {
                let _ = writeln!(out, "\\[{}\\]\n", tex::expr(form));
                let note = entry.note_with(tex::escape, |m| format!("${}$", tex::expr(m)));
                let _ = writeln!(out, "{note}\n");
            }
            Self::Typst => {
                let _ = writeln!(out, "$ {} $\n", typst::expr(form));
                let note = entry.note_with(
                    |t| format!("#text({})", typst::quote(t)),
                    |m| format!("${}$", typst::expr(m)),
                );
                let _ = writeln!(out, "{note}\n");
            }
        }
    }

    fn data_grid(self, out: &mut String, name: &str, grid: &crate::data::Grid) {
        self.heading(out, name, false);
        let columns = grid.headers.len();
        match self {
            Self::Markdown => {}
            Self::Latex => {
                let _ = writeln!(out, "\\begin{{tabular}}{{{}}}", "r".repeat(columns));
            }
            Self::Typst => {
                let _ = writeln!(out, "#table(columns: {columns}, align: right,");
            }
        }
        for (i, row) in std::iter::once(&grid.headers).chain(&grid.rows).enumerate() {
            match self {
                Self::Markdown => {
                    let cells = row
                        .iter()
                        .map(|m| format!("${}$", tex::expr(m)))
                        .collect::<Vec<_>>()
                        .join(" | ");
                    let _ = writeln!(out, "| {cells} |");
                    if i == 0 {
                        let _ = writeln!(out, "|{}", " ---: |".repeat(columns));
                    }
                }
                Self::Latex => {
                    let cells = row
                        .iter()
                        .map(|m| format!("${}$", tex::expr(m)))
                        .collect::<Vec<_>>()
                        .join(" & ");
                    let _ = writeln!(out, "{cells} \\\\");
                    if i == 0 {
                        out.push_str("\\hline\n");
                    }
                }
                Self::Typst => {
                    for value in row {
                        let _ = writeln!(out, "[$ {} $],", typst::expr(value));
                    }
                }
            }
        }
        match self {
            Self::Markdown => out.push('\n'),
            Self::Latex => out.push_str("\\end{tabular}\n\n"),
            Self::Typst => out.push_str(")\n\n"),
        }
    }
}

fn document(
    module: &Module,
    rendering: &Rendering,
    fallback_title: &str,
    format: Format,
) -> String {
    let module_doc = rendering.module_doc.as_ref().map(|doc| {
        let source = doc_text(doc);
        let (title, source) = if doc.markup == Markup::Md {
            split_title(&source)
        } else {
            (None, source)
        };
        (title, source, doc.markup)
    });
    let title = module_doc
        .as_ref()
        .and_then(|d| d.0.as_deref())
        .unwrap_or(fallback_title);
    let mut out = match format {
        Format::Markdown => String::new(),
        Format::Latex => "% Compile with LuaLaTeX or XeLaTeX (Unicode identifiers and source text).\n\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{unicode-math}\n\\begin{document}\n\n".to_string(),
        Format::Typst => "#set page(margin: 2cm)\n#set text(size: 11pt)\n\n".to_string(),
    };
    format.heading(&mut out, title, true);
    if let Some((_, source, markup)) = module_doc {
        format.prose(&mut out, &source, Some(markup));
    }
    let mut equations = Vec::new();
    for binding in &rendering.bindings {
        let mut annotation = binding.annotation.clone();
        if let Some(doc) = &binding.doc {
            let source = doc_text(doc);
            if doc.lines.len() > 1 {
                format.equations(&mut out, &equations);
                equations.clear();
                format.prose(&mut out, &source, Some(doc.markup));
            } else {
                annotation = Some(match annotation {
                    Some(existing) => format!("{source}; {existing}"),
                    None => source,
                });
            }
        }
        equations.push((&binding.statement, annotation));
    }
    format.equations(&mut out, &equations);
    if !rendering.diagnostics.is_empty() {
        format.heading(&mut out, "Diagnostics", false);
        for diagnostic in &rendering.diagnostics {
            let message = if diagnostic.binding.is_empty() {
                diagnostic.message.clone()
            } else {
                format!("{}: {}", diagnostic.binding, diagnostic.message)
            };
            format.prose(&mut out, &message, None);
        }
    }
    if rendering.bindings.iter().any(|b| b.elided) {
        format.heading(&mut out, "Data", false);
        let mut lowerer = Lowerer::new(module);
        for binding in rendering.bindings.iter().filter(|b| b.elided) {
            if let Some((id, _)) = module
                .bindings()
                .find(|(_, b)| module.resolve(b.name) == binding.name)
            {
                let value = lowerer.full_value(id);
                if let Some(grid) = crate::data::grid(&value) {
                    format.data_grid(&mut out, &binding.name, &grid);
                    continue;
                }
                let stmt = Statement {
                    lhs: Math::binding(&binding.name),
                    rel: Rel::Eq,
                    rhs: value,
                };
                format.equations(&mut out, &[(&stmt, None)]);
            }
        }
    }
    if !rendering.notation.is_empty() {
        format.heading(&mut out, "Notation", false);
        for entry in &rendering.notation {
            format.legend(&mut out, entry);
        }
    }
    if matches!(format, Format::Latex) {
        out.push_str("\\end{document}\n");
    }
    out
}

fn markdown_text(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\\' | '`' | '*' | '_' | '{' | '}' | '[' | ']' | '(' | ')' | '#' | '+' | '-' | '.'
            | '!' | '|' | '$' => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out
}
