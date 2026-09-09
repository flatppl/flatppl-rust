//! Doc-comment rendering and whole-document assembly.
//!
//! [`doc_html`] turns one doc-comment into an HTML fragment and
//! [`module_doc_html`] the module doc into a title and an abstract; both feed
//! the JSON contract (the viewer's Math pane) and [`html`], the standalone
//! page. The page's structure is `NOTATION.md` "Document structure": the
//! module doc is title and abstract; a multi-line doc-comment is a paragraph
//! that ends the current equation block; a one-line doc-comment is the
//! right-column annotation of its row; consecutive rows without prose between
//! them form one block aligned on the relation symbol. Long data literals and a
//! notation summary go to appendices.
//!
//! Markdown renders through `pulldown-cmark` (tables and strikethrough on, a
//! single newline a soft break); `$…$` and `$$…$$` through `math-core`, which
//! writes MathML Core and refuses a command it does not know — the source then
//! shows as `<span class="math-error">` and the row carries a diagnostic.
//! `typ` markup has no renderer here and shows as its source in `<pre>`.
//!
//! The output is inserted into a page or a webview verbatim, so it is
//! sanitised: raw HTML in a doc-comment is shown as text, a link keeps its
//! target only for `http(s)`, `mailto`, a fragment or a relative path and
//! otherwise degrades to its text, an image keeps only an `http(s)` target and
//! otherwise degrades to its alt text. `math-core` writes `style` attributes
//! for `\color` and for the cell alignment of `cases` and `array`; those are
//! the only inline styles in the fragment.

use std::fmt::Write;
use std::sync::OnceLock;

use flatppl_core::{Doc, Markup, Module};
use pulldown_cmark::{CowStr, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::ast::Math;
use crate::lower::Lowerer;
use crate::mathml::{self, escape};
use crate::render::Rendering;

/// One doc-comment as HTML.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocHtml {
    pub html: String,
    /// A multi-line (`%%%`) comment: prose, as opposed to a one-line `%`
    /// caption.
    pub block: bool,
    /// Math the converter refused, one message per expression.
    pub errors: Vec<String>,
}

/// The module doc: its leading heading and the rest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleDoc {
    /// The first Markdown heading's text, when the doc starts with one.
    pub title: Option<String>,
    /// The rest, as HTML.
    pub html: String,
    pub errors: Vec<String>,
}

/// Render a doc-comment.
pub fn doc_html(doc: &Doc) -> DocHtml {
    let block = doc.lines.len() > 1;
    match doc.markup {
        Markup::Md => {
            let (html, errors) = markdown_with_errors(&doc_text(doc));
            DocHtml {
                html,
                block,
                errors,
            }
        }
        Markup::Typ => DocHtml {
            html: typst_source(&doc_text(doc)),
            block,
            errors: Vec::new(),
        },
    }
}

/// Render the module doc (the doc-comment on `flatppl_compat`), splitting a
/// leading Markdown heading off as the title.
pub fn module_doc_html(doc: &Doc) -> ModuleDoc {
    match doc.markup {
        Markup::Md => {
            let (title, rest) = split_title(&doc_text(doc));
            let (html, errors) = markdown_with_errors(&rest);
            ModuleDoc {
                title,
                html,
                errors,
            }
        }
        Markup::Typ => ModuleDoc {
            title: None,
            html: typst_source(&doc_text(doc)),
            errors: Vec::new(),
        },
    }
}

/// Render `module` (already rendered into `rendering`) as a standalone HTML
/// page. `fallback_title` names the document when the module doc has no
/// heading (the file stem, typically).
pub fn html(module: &Module, rendering: &Rendering, fallback_title: &str) -> String {
    let doc = fragment(module, rendering, fallback_title);
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>{}</title>\n<style>\n{}</style>\n</head>\n<body>\n{}\n</body>\n</html>\n",
        escape(&doc.title),
        CSS,
        doc.html
    )
}

/// The same article is embedded by the viewer and wrapped by the HTML export.
/// Its stylesheet is scoped to `.flatppl-doc`, leaving host chrome untouched.
pub(crate) struct HtmlFragment {
    pub title: String,
    pub html: String,
}

pub(crate) fn fragment(
    module: &Module,
    rendering: &Rendering,
    fallback_title: &str,
) -> HtmlFragment {
    let module_doc = rendering.module_doc.as_ref().map(module_doc_html);
    let title = module_doc
        .as_ref()
        .and_then(|d| d.title.clone())
        .unwrap_or_else(|| fallback_title.to_string());

    let mut body = String::new();
    let _ = writeln!(body, "<h1>{}</h1>", escape(&title));
    let mut module_diags: Vec<String> = Vec::new();
    if let Some(d) = &module_doc {
        if !d.html.trim().is_empty() {
            let _ = write!(
                body,
                "<section class=\"flatppl-abstract\">\n{}</section>\n",
                d.html
            );
        }
        module_diags.extend(d.errors.iter().cloned());
    }

    // Rows, grouped into aligned blocks between prose paragraphs.
    let mut block: Vec<String> = Vec::new();
    let mut block_diags: Vec<String> = Vec::new();
    let mut elided: Vec<&str> = Vec::new();
    for b in &rendering.bindings {
        let mut annotation = b.annotation.as_deref().map(escape);
        if let Some(doc) = &b.doc {
            let rendered = doc_html(doc);
            if rendered.block {
                flush_block(&mut body, &mut block, &mut block_diags);
                let _ = write!(
                    body,
                    "<div class=\"flatppl-prose\">\n{}</div>\n",
                    rendered.html
                );
            } else {
                // mtext is an HTML integration point. Keep the sanitised
                // caption markup so inline math and links work in both hosts.
                let caption = format!("<span class=\"flatppl-caption\">{}</span>", rendered.html);
                annotation = Some(match annotation {
                    Some(a) => format!("{caption}; {a}"),
                    None => caption,
                });
            }
            for e in rendered.errors {
                block_diags.push(format!("{}: {e}", b.name));
            }
        }
        if b.elided {
            elided.push(&b.name);
        }
        block.push(row_html(b, annotation.as_deref()));
        for d in rendering
            .diagnostics
            .iter()
            .filter(|d| b.names.contains(&d.binding))
        {
            block_diags.push(format!("{}: {}", b.name, d.message));
        }
    }
    flush_block(&mut body, &mut block, &mut block_diags);

    module_diags.extend(
        rendering
            .diagnostics
            .iter()
            .filter(|d| {
                !rendering
                    .bindings
                    .iter()
                    .any(|b| b.names.contains(&d.binding))
            })
            .map(|d| {
                if d.binding.is_empty() {
                    d.message.clone()
                } else {
                    format!("{}: {}", d.binding, d.message)
                }
            }),
    );
    if !module_diags.is_empty() {
        body.push_str("<section class=\"flatppl-diagnostics\"><h2>Diagnostics</h2><ul>\n");
        for m in module_diags {
            let _ = writeln!(body, "<li>{}</li>", escape(&m));
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
                if let Some(grid) = crate::data::grid(&value) {
                    let _ = writeln!(
                        body,
                        "<h3><math>{}</math></h3><table class=\"flatppl-data-grid\"><thead><tr>",
                        mathml::expr(&Math::binding(name))
                    );
                    for header in &grid.headers {
                        let _ = write!(
                            body,
                            "<th scope=\"col\"><math>{}</math></th>",
                            mathml::expr(header)
                        );
                    }
                    body.push_str("</tr></thead><tbody>\n");
                    for row in &grid.rows {
                        body.push_str("<tr>");
                        for value in row {
                            let _ = write!(body, "<td><math>{}</math></td>", mathml::expr(value));
                        }
                        body.push_str("</tr>\n");
                    }
                    body.push_str("</tbody></table>\n");
                    continue;
                }
                let stmt = crate::ast::Statement {
                    lhs: Math::binding(name),
                    rel: crate::ast::Rel::Eq,
                    rhs: value,
                };
                // Not a `fragment`: the row above already carries the
                // binding's `data-flatppl-binding` hook, and the page must
                // name each binding once.
                let _ = writeln!(
                    body,
                    "<math display=\"block\" class=\"flatppl-data-value\">{}</math>",
                    mathml::statement(&stmt)
                );
            }
        }
        body.push_str("</section>\n");
    }

    body.push_str(&notation_section(rendering));

    HtmlFragment {
        title,
        html: format!("<article class=\"flatppl-doc\">\n{body}</article>"),
    }
}

pub(crate) const CSS: &str = "\
.flatppl-doc { max-width: 50em; margin: 2em auto; padding: 0 1em; font-family: system-ui, sans-serif; line-height: 1.5; }
.flatppl-doc h1 { font-size: 1.8em; }
.flatppl-doc math[display=block] { margin: 0.8em 0; }
.flatppl-doc mtable { column-gap: 0.5em; }
.flatppl-doc .flatppl-annot { font-size: 85%; opacity: 0.7; }
.flatppl-doc .flatppl-caption > * { display: inline; margin: 0; }
.flatppl-doc .flatppl-code { font-family: ui-monospace, monospace; }
.flatppl-doc .math-error { font-family: ui-monospace, monospace; color: #a33; }
.flatppl-doc .flatppl-diagnostics, .flatppl-doc .flatppl-row-diag { color: #a33; font-size: 90%; }
.flatppl-doc .flatppl-notation table { border-collapse: collapse; }
.flatppl-doc .flatppl-notation td, .flatppl-doc .flatppl-notation th { padding: 0.1em 0.8em 0.1em 0; text-align: left; vertical-align: top; }
.flatppl-doc .flatppl-notation-bindings td { padding: 0.1em 0; vertical-align: baseline; }
.flatppl-doc .flatppl-notation-bindings td:first-child { text-align: right; }
.flatppl-doc .flatppl-data-grid { border-collapse: collapse; margin: 0.8em 0; }
.flatppl-doc .flatppl-data-grid th, .flatppl-doc .flatppl-data-grid td { text-align: right; padding: 0.2em 1em; }
.flatppl-doc .flatppl-data-grid thead { border-bottom: 1px solid currentColor; }
.flatppl-doc .flatppl-block { overflow-x: auto; max-width: 100%; }
.flatppl-doc .flatppl-block > mtable > mtr > mtd { text-align: left; }
.flatppl-doc .flatppl-block > mtable > mtr > mtd:first-child { text-align: right; }
.flatppl-doc .flatppl-block > mtable > mtr > mtd:nth-child(2) { text-align: center; }
";

pub(crate) fn doc_text(doc: &Doc) -> String {
    doc.lines.join("\n")
}

fn typst_source(text: &str) -> String {
    format!("<pre class=\"flatppl-typst-src\">{}</pre>", escape(text))
}

/// A leading Markdown heading is the title; the rest is the abstract.
pub(crate) fn split_title(md: &str) -> (Option<String>, String) {
    let first = md
        .lines()
        .enumerate()
        .find(|(_, line)| !line.trim().is_empty());
    match first {
        Some((i, line)) if line.starts_with('#') => {
            let title = line.trim_start_matches('#').trim().to_string();
            let rest: Vec<&str> = md.lines().skip(i + 1).collect();
            (Some(title), rest.join("\n"))
        }
        _ => (None, md.to_string()),
    }
}

/// Markdown → HTML; see [`markdown_with_errors`].
pub fn markdown(md: &str) -> String {
    markdown_with_errors(md).0
}

/// Markdown → HTML with `$…$` math as MathML and headings shifted one level
/// down (the document title is the only `<h1>`), plus one message per math
/// expression the converter refused. Raw HTML in the Markdown is emitted as
/// text; link and image targets pass [`safe_url`], and a link or image whose
/// target fails degrades to its content.
pub fn markdown_with_errors(md: &str) -> (String, Vec<String>) {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_MATH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    let mut errors = Vec::new();
    // A dropped link or image tag must drop its end tag too. Neither nests.
    let mut dropped_link = false;
    let mut dropped_image = false;
    let parser = Parser::new_ext(md, options).filter_map(|event| {
        Some(match event {
            Event::InlineMath(src) => Event::InlineHtml(doc_math(&src, false, &mut errors).into()),
            Event::DisplayMath(src) => Event::Html(doc_math(&src, true, &mut errors).into()),
            // Author HTML is content, not markup.
            Event::Html(raw) | Event::InlineHtml(raw) => Event::Text(raw),
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
            Event::Start(Tag::Link { dest_url, .. }) if !safe_url(&dest_url, false) => {
                dropped_link = true;
                return None;
            }
            Event::End(TagEnd::Link) if dropped_link => {
                dropped_link = false;
                return None;
            }
            Event::Start(Tag::Image { dest_url, .. }) if !safe_url(&dest_url, true) => {
                dropped_image = true;
                return None;
            }
            Event::End(TagEnd::Image) if dropped_image => {
                dropped_image = false;
                return None;
            }
            other => other,
        })
    });
    let mut out = String::new();
    pulldown_cmark::html::push_html(&mut out, parser);
    (out, errors)
}

/// Whether a link (or, with `image`, an image) may keep its target: `http(s)`
/// and, for a link, `mailto`; a fragment or a relative path (no scheme). A
/// target whose part before the first `/`, `?` or `#` holds an entity is
/// refused as ambiguous, since an entity could hide a scheme character.
fn safe_url(url: &CowStr<'_>, image: bool) -> bool {
    let lower = url.trim().to_ascii_lowercase();
    let head = lower.split(['/', '?', '#']).next().unwrap_or("");
    if head.contains('&') {
        return false;
    }
    // A scheme is a non-empty run of letters, digits, `+`, `-`, `.` ending
    // in `:` before any `/`, `?` or `#`.
    let scheme = head.find(':').and_then(|i| {
        let s = &head[..i];
        (!s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)))
        .then_some(s)
    });
    match scheme {
        None => true,
        Some("http" | "https") => true,
        Some("mailto") => !image,
        Some(_) => false,
    }
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

fn converter() -> &'static math_core::LatexToMathML {
    static CONVERTER: OnceLock<math_core::LatexToMathML> = OnceLock::new();
    CONVERTER.get_or_init(|| {
        math_core::LatexToMathML::new(math_core::MathCoreConfig::default())
            .expect("the default configuration defines no macros")
    })
}

/// LaTeX from a doc-comment as MathML; the escaped source in
/// `<span class="math-error">` when the converter refuses it, with the
/// converter's message pushed to `errors`.
fn doc_math(latex: &str, display: bool, errors: &mut Vec<String>) -> String {
    let style = if display {
        math_core::MathDisplay::Block
    } else {
        math_core::MathDisplay::Inline
    };
    match converter().convert_with_local_state(latex, style) {
        Ok(result) => wrap_in_mrow(&result.mathml),
        Err(e) => {
            let delim = if display { "$$" } else { "$" };
            errors.push(format!("math `{latex}`: {e}"));
            format!(
                "<span class=\"math-error\">{delim}{}{delim}</span>",
                escape(latex)
            )
        }
    }
}

/// `<math …>children</math>` → `<math …><mrow>children</mrow></math>`.
/// math-core writes the children directly under `<math>`; Chromium applies
/// the operator dictionary (the spacing around `∼`, `=`, `,`) only inside an
/// `<mrow>`-like element and does not treat the `<math>` root as one, so
/// without the wrap a doc line renders as `θ∼Normal(μ,τ)`. Temml and the row
/// printer both write the wrap; wrapping an already single `<mrow>` again is
/// harmless.
fn wrap_in_mrow(mathml: &str) -> String {
    let Some(open_end) = mathml.find('>') else {
        return mathml.to_string();
    };
    let Some(body) = mathml.strip_suffix("</math>") else {
        return mathml.to_string();
    };
    format!(
        "{}<mrow>{}</mrow></math>",
        &mathml[..=open_end],
        &body[open_end + 1..]
    )
}

/// One `<mtr>` of the aligned block. Annotation is already escaped or sanitised.
fn row_html(b: &crate::render::BindingRender, annotation: Option<&str>) -> String {
    let (lhs, rel, rhs) = mathml::statement_parts(&b.statement);
    let annot = annotation
        .map(|a| format!("<mtext class=\"flatppl-annot\">{a}</mtext>"))
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

/// The notation key: every symbol the rows use, in the notation of its row,
/// with a one-line note. Nothing else: a list of the parameters, inputs or
/// random variables would only repeat the rows above.
fn notation_section(rendering: &Rendering) -> String {
    if rendering.notation.is_empty() {
        return String::new();
    }
    let mut out = String::from("<section class=\"flatppl-notation\"><h2>Notation</h2>\n<table>\n");
    for entry in &rendering.notation {
        let _ = writeln!(
            out,
            "<tr><td><math>{}</math></td><td><p>{}</p></td></tr>",
            mathml::expr(&entry.form),
            entry.note_html()
        );
    }
    out.push_str("</table>\n</section>\n");
    out
}

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
        assert!(p.contains("<span class=\"flatppl-caption\"><p>the programme mean</p>"));
        // Rows are addressable.
        assert!(p.contains("<mtr data-flatppl-binding=\"tau\" id=\"flatppl-tau\">"));
        // The notation key lists the symbols only; nothing repeats the rows.
        assert!(p.contains("<h2>Notation</h2>"));
        assert!(
            !p.contains("Random variables") && !p.contains("<h3>Parameters</h3>"),
            "{p}"
        );
        assert!(
            p.contains("Cauchy distribution with location <math>"),
            "{p}"
        );
        assert!(!p.contains("<code>Cauchy("), "{p}");
    }

    #[test]
    fn wide_literal_data_keeps_values_and_columns_in_the_appendix() {
        let src = "xs = [0.123456789012345, 0.234567890123456, 0.345678901234567, 0.456789012345678, 0.567890123456789]\ndata = table(exposure = [1, 2, 3, 4, 5], efficiency = [6, 7, 8, 9, 10], counts = [11, 12, 13, 14, 15])\ncomputed = table(values = xs)";
        let rendering = crate::render_source(src, "m.flatppl", &HashMap::new()).expect("renders");
        assert!(
            rendering.diagnostics.is_empty(),
            "{:?}",
            rendering.diagnostics
        );
        assert!(rendering.bindings[0].elided);
        assert!(rendering.bindings[1].elided);
        assert!(!rendering.bindings[2].elided);
        let p = page(src);
        let (body, appendix) = p.split_once("<h2>Data</h2>").expect("appendix");
        assert!(!body.contains("0.123456789012345"));
        assert!(appendix.contains("<mn>0.123456789012345</mn>"));
        assert!(appendix.contains("<tr><td><math><mn>1</mn></math></td><td><math><mn>6</mn></math></td><td><math><mn>11</mn></math></td></tr>"));
        let mut module = flatppl_syntax::parse(src).unwrap();
        flatppl_infer::infer(&mut module);
        let md = crate::export::github_markdown(&module, &rendering, "m");
        assert!(
            md.contains("| $\\mathrm{exposure}$ | $\\mathrm{efficiency}$ | $\\mathrm{counts}$ |"),
            "{md}"
        );
        assert!(md.contains("| $5$ | $10$ | $15$ |"), "{md}");
    }

    #[test]
    fn long_arrays_land_in_the_data_appendix() {
        let src = "xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]\nm = elementof(reals)";
        let p = page(src);
        assert!(p.contains("<h2>Data</h2>"));
        assert!(p.contains("13 values, see the data appendix"));
        assert!(p.contains(
            "<tr><td><math><mn>13</mn></math></td><td><math><mn>13</mn></math></td></tr>"
        ));
        assert!(!p.contains("<h3>Parameters</h3>"));
    }

    #[test]
    fn doc_math_that_the_converter_rejects_stays_as_code() {
        let (out, errors) = markdown_with_errors("see $\\undefinedmacro{x}$ here");
        assert!(
            out.contains("<span class=\"math-error\">$\\undefinedmacro{x}$</span>"),
            "{out}"
        );
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("undefinedmacro"), "{errors:?}");
        let out = markdown("see $x^2$ here");
        assert!(out.contains("<math"), "{out}");
        // A literal comparison operator is escaped in the MathML.
        let out = markdown("the condition $a < b$ and $c > d$ holds");
        assert!(out.contains("<mo>&lt;</mo>"), "{out}");
        assert!(out.contains("<mo>&gt;</mo>"), "{out}");
        assert!(
            !out.contains("<mo><</mo>") && !out.contains("<mo>></mo>"),
            "{out}"
        );
    }

    #[test]
    fn doc_comment_html_is_text_and_unsafe_link_targets_are_dropped() {
        let out = markdown("a <script>alert(1)</script> b\n\n<img src=x onerror=alert(1)>\n");
        assert!(!out.contains("<script>"), "{out}");
        assert!(out.contains("&lt;script&gt;"), "{out}");
        assert!(!out.contains("<img"), "{out}");
        let out = markdown(
            "[x](javascript:alert(1)) [y](https://example.org/p?q=1#f) [z](../rel/p.html) [w](#frag) [v](mailto:a@b.c) [u](DATA:text/html,hi)",
        );
        assert!(!out.contains("javascript:"), "{out}");
        assert!(!out.to_ascii_lowercase().contains("data:"), "{out}");
        assert!(
            out.contains("href=\"https://example.org/p?q=1#f\""),
            "{out}"
        );
        assert!(out.contains("href=\"../rel/p.html\""), "{out}");
        assert!(out.contains("href=\"#frag\""), "{out}");
        assert!(out.contains("href=\"mailto:a@b.c\""), "{out}");
        // A colon inside a path or query is not a scheme.
        let out = markdown("[p](dir/a:b.html) [q](?k=a:b)");
        assert!(out.contains("href=\"dir/a:b.html\""), "{out}");
        assert!(out.contains("href=\"?k=a:b\""), "{out}");
        // Images go through the same filter.
        let out = markdown(
            "![i](javascript:alert(1)) ![j](https://example.org/i.png) ![k](mailto:a@b.c)",
        );
        assert!(!out.contains("javascript:"), "{out}");
        assert!(out.contains("src=\"https://example.org/i.png\""), "{out}");
        // A refused image degrades to its alt text; `mailto` is not an image.
        assert_eq!(out.matches("<img").count(), 1, "{out}");
        assert!(out.contains("i ") && out.contains(" k"), "{out}");
    }

    #[test]
    fn the_data_appendix_names_each_binding_once() {
        let src = "xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]";
        let p = page(src);
        assert_eq!(p.matches("data-flatppl-binding=\"xs\"").count(), 1, "{p}");
        assert!(p.contains("class=\"flatppl-data-grid\""));
    }

    #[test]
    fn doc_comments_render_with_the_block_flag_and_the_module_doc_splits_its_title() {
        let src = "%%%\n# Eight Schools\n\nWith $\\mu$ and $\\tau$.\n%%%\nflatppl_compat = \"0.1\"\n% the mean, $\\mu \\in \\mathbb{R}$\nmu ~ Normal(0, 5)\n%%%\nTwo lines\nof prose with $\\bad{x}$.\n%%%\ntau = elementof(posreals)";
        let r = crate::render_source(src, "m.flatppl", &HashMap::new()).expect("renders");
        let m = module_doc_html(r.module_doc.as_ref().expect("module doc"));
        assert_eq!(m.title.as_deref(), Some("Eight Schools"));
        assert!(m.html.starts_with("<p>With <math>"), "{}", m.html);
        assert!(m.errors.is_empty());
        let mu = doc_html(r.bindings[0].doc.as_ref().unwrap());
        assert!(!mu.block);
        assert!(mu.html.contains("<math><mrow><mi>μ</mi>"), "{}", mu.html);
        let tau = doc_html(r.bindings[1].doc.as_ref().unwrap());
        assert!(tau.block);
        assert!(
            tau.html
                .contains("<span class=\"math-error\">$\\bad{x}$</span>"),
            "{}",
            tau.html
        );
        assert_eq!(tau.errors.len(), 1);
        // The page lists the refused math under the row.
        let mut module = flatppl_syntax::parse(src).unwrap();
        flatppl_infer::infer(&mut module);
        let page = html(&module, &r, "m");
        assert!(page.contains("<h1>Eight Schools</h1>"));
        assert!(
            page.contains("<p class=\"flatppl-row-diag\">tau: math `\\bad{x}`"),
            "{page}"
        );
    }

    #[test]
    fn doc_math_children_sit_in_one_mrow_under_the_math_root() {
        let out = markdown("$\\theta \\sim \\mathrm{Normal}(\\mu, \\tau)$ and $$x = 1$$");
        assert!(out.contains("<math><mrow><mi>θ</mi>"), "{out}");
        assert!(out.contains("</mrow></math>"), "{out}");
        assert!(
            out.contains("<math display=\"block\"><mrow><mi>x</mi>"),
            "{out}"
        );
        assert_eq!(
            out.matches("<math").count(),
            out.matches("</mrow></math>").count(),
            "{out}"
        );
    }

    #[test]
    fn the_legend_explains_the_set_letter_only_beside_a_set_function_row() {
        let with = page(
            "mu = elementof(reals)\ny ~ Normal(mu, 1)\nK = kernelof(y, mu = mu)\nL = likelihoodof(K, 0.3)\nprior = Normal(0, 10)\npost = bayesupdate(L, prior)\ng ~ Gamma(2, 3)",
        );
        assert!(
            with.contains("<td><math><mi>𝘈</mi></math></td><td><p>A measurable set. A row <math>"),
            "{with}"
        );
        assert!(with.contains("defines the measure <math><mi>ν</mi></math> by its value on every set <math><mi>𝘈</mi></math>."), "{with}");
        // The rate family pins its convention by the mean, in math.
        assert!(with.contains("Gamma distribution with shape <math><mi>α</mi></math> and rate <math><mi>β</mi></math>."), "{with}");
        // The legend explains the notation; it never shows the code spelling.
        assert!(!with.contains("<code>Gamma("), "{with}");
        assert!(!with.contains("Normal(mu, sigma)"), "{with}");
        let without = page("x ~ Normal(0, 1)");
        assert!(!without.contains("<mi>𝘈</mi>"), "{without}");
        assert!(without.contains("Normal distribution with mean <math><mi>μ</mi></math> and variance <math><msup><mi>σ</mi><mn>2</mn></msup></math>."), "{without}");
    }

    #[test]
    fn a_typst_doc_comment_shows_its_source() {
        let src = "%%%typ\n= Title\nSome _prose_.\n%%%\nflatppl_compat = \"0.1\"\nx = 1";
        let r = crate::render_source(src, "m.flatppl", &HashMap::new()).expect("renders");
        let m = module_doc_html(r.module_doc.as_ref().expect("module doc"));
        assert_eq!(m.title, None);
        assert_eq!(
            m.html,
            "<pre class=\"flatppl-typst-src\">= Title\nSome _prose_.</pre>"
        );
    }

    #[test]
    fn doc_math_and_row_math_share_their_markup_conventions() {
        // The two generators feed one renderer; the conventions that decide
        // how the same expression looks must agree: rigid parentheses around
        // plain content, upright multi-letter names, italic single letters.
        let doc = markdown("$\\mathrm{Normal}(\\mu, \\tau)$");
        let row = mathml::expr(&Math::call(
            "Normal",
            vec![Math::binding("mu"), Math::binding("tau")],
        ));
        for needle in [
            "<mo stretchy=\"false\">(</mo>",
            "<mo stretchy=\"false\">)</mo>",
            "<mi>Normal</mi>",
            "<mi>μ</mi>",
            "<mi>τ</mi>",
        ] {
            assert!(doc.contains(needle), "doc math lacks {needle}: {doc}");
            assert!(
                row.replace(" data-flatppl-ref=\"mu\"", "")
                    .replace(" data-flatppl-ref=\"tau\"", "")
                    .contains(needle),
                "row math lacks {needle}: {row}"
            );
        }
        assert!(!doc.contains("mathvariant"), "{doc}");
    }
}
