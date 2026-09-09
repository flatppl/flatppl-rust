//! Whole-document assembly: a module as an HTML page of MathML.
//!
//! The structure is `NOTATION.md` "Document structure": the module doc is
//! title and abstract; a multi-line doc-comment is a paragraph that ends the
//! current equation block; a one-line doc-comment is the right-column
//! annotation of its row; consecutive rows without prose between them form one
//! block aligned on the relation symbol. Long data literals and a notation
//! summary (parameters, inputs, latent variables, distribution
//! parametrisations) go to appendices.
//!
//! Doc-comment Markdown renders through `pulldown-cmark`; its `$…$` math
//! through `latex2mathml`, falling back to the LaTeX source in `<code>` for
//! anything that converter does not know.

use std::fmt::Write;

use flatppl_core::{CallHead, Doc, Module, Node};
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::ast::Math;
use crate::lower::{self, Lowerer};
use crate::mathml::{self, escape};
use crate::render::Rendering;

/// Render `module` (already rendered into `rendering`) as a standalone HTML
/// page. `fallback_title` names the document when the module doc has no
/// heading (the file stem, typically).
pub fn html(module: &Module, rendering: &Rendering, fallback_title: &str) -> String {
    let (title, abstract_md) = module_doc(module).map_or_else(
        || (fallback_title.to_string(), String::new()),
        |doc| split_title(&doc_text(&doc), fallback_title),
    );

    let mut body = String::new();
    let _ = writeln!(body, "<h1>{}</h1>", escape(&title));
    if !abstract_md.trim().is_empty() {
        let _ = write!(
            body,
            "<section class=\"flatppl-abstract\">\n{}</section>\n",
            markdown(&abstract_md)
        );
    }

    // Rows, grouped into aligned blocks between prose paragraphs.
    let mut block: Vec<String> = Vec::new();
    let mut block_diags: Vec<String> = Vec::new();
    let mut elided: Vec<&str> = Vec::new();
    for b in &rendering.bindings {
        let mut annotation = b.annotation.clone();
        if let Some(doc) = &b.doc {
            if doc.lines.len() > 1 {
                flush_block(&mut body, &mut block, &mut block_diags);
                let _ = write!(
                    body,
                    "<div class=\"flatppl-prose\">\n{}</div>\n",
                    markdown(&doc_text(doc))
                );
            } else if let Some(line) = doc.lines.first() {
                let line = line.trim();
                annotation = Some(match annotation {
                    Some(a) => format!("{line}; {a}"),
                    None => line.to_string(),
                });
            }
        }
        if b.annotation
            .as_deref()
            .is_some_and(|a| a.contains("data appendix"))
        {
            elided.push(&b.name);
        }
        block.push(row_html(b, annotation.as_deref()));
        for d in rendering.diagnostics.iter().filter(|d| d.binding == b.name) {
            block_diags.push(format!("{}: {}", b.name, d.message));
        }
    }
    flush_block(&mut body, &mut block, &mut block_diags);

    let module_diags: Vec<&str> = rendering
        .diagnostics
        .iter()
        .filter(|d| d.binding.is_empty())
        .map(|d| d.message.as_str())
        .collect();
    if !module_diags.is_empty() {
        body.push_str("<section class=\"flatppl-diagnostics\"><h2>Diagnostics</h2><ul>\n");
        for m in module_diags {
            let _ = writeln!(body, "<li>{}</li>", escape(m));
        }
        body.push_str("</ul></section>\n");
    }

    if !elided.is_empty() {
        body.push_str("<section class=\"flatppl-data\"><h2>Data</h2>\n");
        let mut lowerer = Lowerer::new(module);
        for name in elided {
            if let Some((id, _)) = module
                .bindings()
                .find(|(_, b)| module.resolve(b.name) == name)
            {
                let value = lowerer.full_value(id);
                let stmt = crate::ast::Statement {
                    lhs: Math::binding(name),
                    rel: crate::ast::Rel::Eq,
                    rhs: value,
                };
                let _ = writeln!(body, "{}", mathml::fragment(name, &stmt));
            }
        }
        body.push_str("</section>\n");
    }

    body.push_str(&notation_section(module, rendering));

    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>{}</title>\n<style>\n{}</style>\n</head>\n<body>\n<article class=\"flatppl-doc\">\n{}</article>\n</body>\n</html>\n",
        escape(&title),
        CSS,
        body
    )
}

const CSS: &str = "\
.flatppl-doc { max-width: 50em; margin: 2em auto; padding: 0 1em; font-family: system-ui, sans-serif; line-height: 1.5; }
.flatppl-doc h1 { font-size: 1.8em; }
.flatppl-doc math[display=block] { margin: 0.8em 0; }
.flatppl-doc mtable { column-gap: 0.5em; }
.flatppl-doc .flatppl-annot { font-size: 85%; opacity: 0.7; }
.flatppl-doc .flatppl-code { font-family: ui-monospace, monospace; }
.flatppl-doc .flatppl-diagnostics, .flatppl-doc .flatppl-row-diag { color: #a33; font-size: 90%; }
.flatppl-doc .flatppl-notation table { border-collapse: collapse; }
.flatppl-doc .flatppl-notation td, .flatppl-doc .flatppl-notation th { padding: 0.1em 0.8em 0.1em 0; text-align: left; vertical-align: top; }
";

/// The module doc: the doc-comment on `flatppl_compat`.
fn module_doc(module: &Module) -> Option<Doc> {
    module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "flatppl_compat")
        .and_then(|(_, b)| b.doc.clone())
}

fn doc_text(doc: &Doc) -> String {
    doc.lines.join("\n")
}

/// A leading Markdown heading is the title; the rest is the abstract.
fn split_title(md: &str, fallback: &str) -> (String, String) {
    let first = md
        .lines()
        .enumerate()
        .find(|(_, line)| !line.trim().is_empty());
    let rest_start = first.map_or(0, |(i, _)| i + 1);
    match first.map(|(_, line)| line) {
        Some(line) if line.starts_with('#') => {
            let title = line.trim_start_matches('#').trim().to_string();
            let rest: Vec<&str> = md.lines().skip(rest_start).collect();
            (title, rest.join("\n"))
        }
        _ => (fallback.to_string(), md.to_string()),
    }
}

/// Markdown → HTML with `$…$` math as MathML and headings shifted one level
/// down (the document title is the only `<h1>`).
pub fn markdown(md: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_MATH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    let parser = Parser::new_ext(md, options).map(|event| match event {
        Event::InlineMath(src) => Event::InlineHtml(doc_math(&src, false).into()),
        Event::DisplayMath(src) => Event::Html(doc_math(&src, true).into()),
        Event::Start(Tag::Heading {
            level,
            id,
            classes,
            attrs,
        }) => Event::Start(Tag::Heading {
            level: shift_heading(level),
            id,
            classes,
            attrs,
        }),
        Event::End(TagEnd::Heading(level)) => Event::End(TagEnd::Heading(shift_heading(level))),
        other => other,
    });
    let mut out = String::new();
    pulldown_cmark::html::push_html(&mut out, parser);
    out
}

fn shift_heading(level: HeadingLevel) -> HeadingLevel {
    match level {
        HeadingLevel::H1 => HeadingLevel::H2,
        HeadingLevel::H2 => HeadingLevel::H3,
        HeadingLevel::H3 => HeadingLevel::H4,
        HeadingLevel::H4 => HeadingLevel::H5,
        _ => HeadingLevel::H6,
    }
}

/// LaTeX from a doc-comment as MathML; the source in `<code>` when the
/// converter refuses it.
fn doc_math(latex: &str, display: bool) -> String {
    let style = if display {
        latex2mathml::DisplayStyle::Block
    } else {
        latex2mathml::DisplayStyle::Inline
    };
    // The converter reports an unknown command inline as an `[PARSE ERROR …]`
    // text node rather than an `Err`; treat both the same way.
    match latex2mathml::latex_to_mathml(latex, style) {
        Ok(mathml) if !mathml.contains("[PARSE ERROR") => mathml,
        _ => {
            let delim = if display { "$$" } else { "$" };
            format!("<code>{delim}{}{delim}</code>", escape(latex))
        }
    }
}

/// One `<mtr>` of the aligned block: lhs, relation, rhs, annotation.
fn row_html(b: &crate::render::BindingRender, annotation: Option<&str>) -> String {
    let (lhs, rel, rhs) = mathml::statement_parts(&b.statement);
    let annot = annotation
        .map(|a| format!("<mtext class=\"flatppl-annot\">{}</mtext>", escape(a)))
        .unwrap_or_default();
    format!(
        "<mtr data-flatppl-binding=\"{name}\" id=\"flatppl-{name}\"><mtd>{lhs}</mtd><mtd><mo>{rel}</mo></mtd><mtd>{rhs}</mtd><mtd>{annot}</mtd></mtr>",
        name = escape(&b.name)
    )
}

fn flush_block(body: &mut String, block: &mut Vec<String>, diags: &mut Vec<String>) {
    if block.is_empty() {
        return;
    }
    body.push_str(
        "<math display=\"block\" class=\"flatppl-block\"><mtable columnalign=\"right center left left\">\n",
    );
    for row in block.drain(..) {
        body.push_str(&row);
        body.push('\n');
    }
    body.push_str("</mtable></math>\n");
    for d in diags.drain(..) {
        let _ = writeln!(body, "<p class=\"flatppl-row-diag\">{}</p>", escape(&d));
    }
}

/// Parameters, external inputs, latent variables and the distributions used,
/// read off the module's heads and the rendered rows.
fn notation_section(module: &Module, rendering: &Rendering) -> String {
    let mut params = Vec::new();
    let mut inputs = Vec::new();
    let mut latents = Vec::new();
    for b in &rendering.bindings {
        let Some((_, binding)) = module
            .bindings()
            .find(|(_, x)| module.resolve(x.name) == b.name)
        else {
            continue;
        };
        let Node::Call(call) = module.node(binding.rhs) else {
            continue;
        };
        let CallHead::Builtin(head) = call.head else {
            continue;
        };
        let (_, _, rhs) = mathml::statement_parts(&b.statement);
        let lhs = mathml::expr(&b.statement.lhs);
        match module.resolve(head) {
            "elementof" => params.push((lhs, rhs)),
            "external" => inputs.push((lhs, rhs)),
            "draw" => latents.push((lhs, rhs)),
            _ => {}
        }
    }
    let mut dists: Vec<String> = Vec::new();
    for (_, b) in module.bindings() {
        collect_distributions(module, b.rhs, &mut dists);
    }
    dists.sort();
    dists.dedup();

    if params.is_empty() && inputs.is_empty() && latents.is_empty() && dists.is_empty() {
        return String::new();
    }
    let mut out = String::from("<section class=\"flatppl-notation\"><h2>Notation</h2>\n");
    let table = |out: &mut String, title: &str, rows: &[(String, String)], rel: &str| {
        if rows.is_empty() {
            return;
        }
        let _ = writeln!(out, "<h3>{title}</h3>\n<table>");
        for (lhs, rhs) in rows {
            let _ = writeln!(
                out,
                "<tr><td><math>{lhs}</math></td><td><math><mrow><mo>{rel}</mo>{rhs}</mrow></math></td></tr>"
            );
        }
        out.push_str("</table>\n");
    };
    table(&mut out, "Parameters", &params, "∈");
    table(&mut out, "External inputs", &inputs, "∈");
    table(&mut out, "Latent variables", &latents, "∼");
    if !dists.is_empty() {
        out.push_str("<h3>Distributions</h3>\n<table>\n");
        for d in dists {
            let params = flatppl_infer::distribution_param_names(&d)
                .map(|p| p.join(", "))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "<tr><td><math><mi>{}</mi></math></td><td><code>{}({})</code></td></tr>",
                escape(&d),
                escape(&d),
                escape(&params)
            );
        }
        out.push_str("</table>\n");
    }
    out.push_str("</section>\n");
    out
}

fn collect_distributions(module: &Module, node: flatppl_core::NodeId, out: &mut Vec<String>) {
    if let Node::Call(call) = module.node(node)
        && let CallHead::Builtin(head) = call.head
    {
        let name = module.resolve(head);
        if flatppl_infer::builtin_catalogue().base_is_distribution(name)
            && !out.iter().any(|d| d == name)
        {
            out.push(name.to_string());
        }
    }
    // Broadcast heads are bare atoms.
    if let Node::Const(sym) = module.node(node) {
        let name = module.resolve(*sym);
        if flatppl_infer::builtin_catalogue().base_is_distribution(name)
            && !out.iter().any(|d| d == name)
        {
            out.push(name.to_string());
        }
    }
    module.for_each_child(node, |c| collect_distributions(module, c, out));
}

// Keep `lower` referenced for the appendix threshold documentation.
#[allow(dead_code)]
const _: usize = lower::INLINE_ARRAY_LIMIT;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const EIGHT_SCHOOLS: &str = "%%%\n# Eight Schools model\n\nClassic hierarchical Normal model with $\\mu$ and $\\tau$.\n%%%\nflatppl_compat = \"0.1\"\n\ny_data = [28, 8, -3, 7, -1, 1, 18, 12]\nstd_errs_data = [15, 10, 16, 11, 9, 11, 10, 18]\nJ = 8\n\n% the programme mean\nmu ~ Normal(0, 5)\ntau ~ normalize(truncate(Cauchy(0, 5), interval(0, inf)))\ntheta ~ iid(Normal(mu, tau), J)\n\n%%%\n## Observation model\n\nEach school reports its effect with a known standard error.\n%%%\ny ~ Normal.(theta, std_errs_data)\n\nprior = lawof(record(mu = mu, tau = tau, theta = theta))\nforward_kernel = kernelof(record(y = y), mu = mu, tau = tau, theta = theta)\nL = likelihoodof(forward_kernel, record(y = y_data))\nposterior = bayesupdate(L, prior)\n";

    fn page(src: &str) -> String {
        let rendering = crate::render_source(src, "m.flatppl", &HashMap::new()).expect("renders");
        let mut module = flatppl_syntax::parse(src).unwrap();
        flatppl_infer::infer(&mut module);
        html(&module, &rendering, "m")
    }

    #[test]
    fn the_module_doc_gives_title_and_abstract_and_block_docs_split_blocks() {
        let p = page(EIGHT_SCHOOLS);
        assert!(p.contains("<title>Eight Schools model</title>"));
        assert!(p.contains("<h1>Eight Schools model</h1>"));
        assert!(
            p.contains("Classic hierarchical Normal model with <math"),
            "{p}"
        );
        // The prose heading is shifted below the title.
        assert!(p.contains("<h3>Observation model</h3>"));
        // Two equation blocks: before and after the prose.
        assert_eq!(
            p.matches("<math display=\"block\" class=\"flatppl-block\">")
                .count(),
            2
        );
        // One-line doc → annotation.
        assert!(p.contains("<mtext class=\"flatppl-annot\">the programme mean</mtext>"));
        // Rows are addressable.
        assert!(p.contains("<mtr data-flatppl-binding=\"tau\" id=\"flatppl-tau\">"));
        // Notation appendix lists the latent variables and the distributions.
        assert!(p.contains("<h3>Latent variables</h3>"));
        assert!(p.contains("<code>Cauchy(location, scale)</code>"));
    }

    #[test]
    fn long_arrays_land_in_the_data_appendix() {
        let src = "xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]\nm = elementof(reals)";
        let p = page(src);
        assert!(p.contains("<h2>Data</h2>"));
        assert!(p.contains("13 values, see the data appendix"));
        assert!(p.contains("<mn>13</mn><mo>)</mo>"));
        assert!(p.contains("<h3>Parameters</h3>"));
    }

    #[test]
    fn doc_math_that_the_converter_rejects_stays_as_code() {
        let out = markdown("see $\\undefinedmacro{x}$ here");
        assert!(out.contains("<code>$\\undefinedmacro{x}$</code>"), "{out}");
        let out = markdown("see $x^2$ here");
        assert!(out.contains("<math"), "{out}");
    }
}
