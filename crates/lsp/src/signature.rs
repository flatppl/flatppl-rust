//! `textDocument/signatureHelp` over a call head.
//!
//! Two halves. The call site is found **textually** — signature help fires while
//! the user is mid-call, when the file usually does not parse, so a forward scan
//! that tracks nesting, strings and comments is what can answer at all. The
//! parameter roster then comes from whoever owns it: `flatppl_infer`'s catalogue
//! for built-ins (§07/§08) and standard-module members (§09), and the
//! `functionof` boundary of a user `FunctionDefinition` for a local callable.
//! No argument names are declared here.

use flatppl_core::{CallHead, Inputs, Node};

use crate::db::{Catalogues, FileSet, SourceFile};
use crate::queries::parsed_catalogues;

/// What a call's head names.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Head {
    /// A bare name: a built-in, or a callable bound in this module.
    Bare(String),
    /// `alias.member` — a standard-module member, or a member of a loaded file.
    Member { alias: String, member: String },
}

/// Which parameter the cursor is on.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Active {
    /// The nth comma-separated argument, counted from the open paren.
    Positional(usize),
    /// A keyword argument `name = …` (spec §05 `KeywordArg ::= Name "=" Expression`).
    Named(String),
}

/// The call the cursor sits inside.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct CallSite {
    pub head: Head,
    pub active: Active,
}

/// Is `b` a byte that may appear in a FlatPPL identifier (§05 `Name`)?
#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// One open bracket the forward scan is inside.
struct Frame {
    /// The bracket byte, so a `(` call frame is distinguishable from `[` / `{`.
    delim: u8,
    /// Offset just past the bracket.
    body: u32,
    /// Comma count at this frame's own level.
    commas: usize,
    /// Offset just past the most recent separator at this level (the bracket or
    /// a comma) — the start of the argument the cursor is in.
    arg_start: u32,
}

/// Find the call the cursor at `byte` is inside, or `None` when it is not in a
/// call's argument list.
///
/// Scans forward from the start of the file rather than backwards from the
/// cursor: `%` comments and `"` strings can only be recognized reliably in
/// reading order, and a backwards scan would count a comma inside either. Each
/// bracket gets its own frame, so commas nested in `[…]` or in an inner call do
/// not advance the outer argument index.
pub(crate) fn call_site_at(text: &str, byte: u32) -> Option<CallSite> {
    let bytes = text.as_bytes();
    let end = (byte as usize).min(bytes.len());
    let mut stack: Vec<Frame> = Vec::new();
    let mut i = 0usize;
    while i < end {
        match bytes[i] {
            // A `%` comment runs to the end of the line (spec §05).
            b'%' => {
                while i < end && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'"' => {
                i += 1;
                while i < end {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
            open @ (b'(' | b'[' | b'{') => {
                i += 1;
                stack.push(Frame {
                    delim: open,
                    body: i as u32,
                    commas: 0,
                    arg_start: i as u32,
                });
            }
            b')' | b']' | b'}' => {
                stack.pop();
                i += 1;
            }
            b',' => {
                i += 1;
                if let Some(f) = stack.last_mut() {
                    f.commas += 1;
                    f.arg_start = i as u32;
                }
            }
            _ => i += 1,
        }
    }

    // The innermost `(` frame: the cursor may be nested in a `[…]` inside an
    // argument, and the call is still the enclosing paren.
    let frame = stack.iter().rev().find(|f| f.delim == b'(')?;
    let head = head_before(text, frame.body - 1)?;
    let active = active_arg(text, frame.arg_start, byte, frame.commas);
    Some(CallSite { head, active })
}

/// Read the call head immediately left of the open paren at `open`.
fn head_before(text: &str, open: u32) -> Option<Head> {
    let bytes = text.as_bytes();
    let mut i = open as usize;
    while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\r' | b'\n') {
        i -= 1;
    }
    let name_end = i;
    while i > 0 && is_ident_byte(bytes[i - 1]) {
        i -= 1;
    }
    if i == name_end {
        return None; // a grouping paren, not a call
    }
    let name = text.get(i..name_end)?.to_string();
    // `alias.name` — look for a dot and a second identifier.
    let mut j = i;
    while j > 0 && matches!(bytes[j - 1], b' ' | b'\t' | b'\r' | b'\n') {
        j -= 1;
    }
    if j == 0 || bytes[j - 1] != b'.' {
        return Some(Head::Bare(name));
    }
    j -= 1;
    while j > 0 && matches!(bytes[j - 1], b' ' | b'\t' | b'\r' | b'\n') {
        j -= 1;
    }
    let alias_end = j;
    while j > 0 && is_ident_byte(bytes[j - 1]) {
        j -= 1;
    }
    if j == alias_end {
        return Some(Head::Bare(name));
    }
    Some(Head::Member {
        alias: text.get(j..alias_end)?.to_string(),
        member: name,
    })
}

/// Classify the argument spanning `[arg_start, byte)`: a keyword argument when it
/// opens `name =`, otherwise the positional slot given by the comma count.
///
/// A `==` is not a keyword argument, so the byte after the `=` must not be `=`.
fn active_arg(text: &str, arg_start: u32, byte: u32, commas: usize) -> Active {
    let bytes = text.as_bytes();
    let mut i = arg_start as usize;
    let end = (byte as usize).min(bytes.len());
    while i < end && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    let name_start = i;
    while i < end && is_ident_byte(bytes[i]) {
        i += 1;
    }
    let name_end = i;
    while i < end && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    let is_kwarg = name_end > name_start
        && bytes.get(i) == Some(&b'=')
        && bytes.get(i + 1) != Some(&b'=')
        && i < end;
    if is_kwarg {
        return Active::Named(text[name_start..name_end].to_string());
    }
    Active::Positional(commas)
}

/// Read something off `file`'s module, tolerating a buffer that does not parse.
///
/// Signature help fires *inside* an open call, so the file is normally
/// unparseable at exactly the moment it is asked. Like member completion, this
/// retries on the text with its last non-empty line stripped, which recovers the
/// bindings declared above the call being typed.
///
/// `parse`, not `analyze`: every caller reads structure only (binding names,
/// call heads, reification boundaries), so no inference is forced.
fn with_module<T>(
    db: &dyn salsa::Database,
    file: SourceFile,
    f: impl Fn(&flatppl_core::Module) -> Option<T>,
) -> Option<T> {
    if let Some(module) = crate::queries::parse(db, file).module(db) {
        return f(module);
    }
    let text = file.text(db);
    let repaired = crate::capabilities::strip_last_nonempty_line(text);
    if repaired.is_empty() || repaired == text {
        return None;
    }
    f(&flatppl_syntax::parse(repaired).ok()?)
}

/// A resolved signature: the head as written, its ordered parameter names when
/// declared, and a one-line note (the arity and the spec section it comes from).
struct Roster {
    label_head: String,
    params: Vec<String>,
    note: Option<String>,
}

/// The ordered parameter names of a user callable bound as `name` in `module`.
///
/// A `FunctionDefinition` (`f(a, b) = …`) lowers to a `functionof` reification
/// whose `%specinputs` boundary carries the surface argument names in order, so
/// the roster is read off the boundary rather than re-derived from the syntax.
fn user_param_names(module: &flatppl_core::Module, name: &str) -> Option<Vec<String>> {
    let (_, binding) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)?;
    let Node::Call(call) = module.node(binding.rhs) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    let head_name = module.resolve(head);
    if head_name != "functionof" && head_name != "kernelof" {
        return None;
    }
    match call.inputs.as_ref()? {
        Inputs::Spec(entries) => Some(
            entries
                .iter()
                .map(|(sym, _)| module.resolve(*sym).to_string())
                .collect(),
        ),
        // `%autoinputs` is a keyword-only callable whose boundary is inference
        // metadata, not an ordered surface list, so there is no positional
        // roster to show.
        Inputs::Auto => None,
    }
}

/// Resolve `head` to a parameter roster, or `None` when nothing declares one.
fn roster(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    head: &Head,
) -> Option<Roster> {
    let builtin = flatppl_infer::builtin_catalogue();
    let external = parsed_catalogues(db, cats);

    match head {
        Head::Member { alias, member } => {
            let label_head = format!("{alias}.{member}");
            // A `standard_module` alias: the roster is a §09 catalogue row.
            if let Some(mod_name) =
                crate::capabilities::find_standard_module_name(db, file, fs, cats, alias)
            {
                let names = builtin
                    .module_param_names(&mod_name, member)
                    .map(|n| n.to_vec())
                    .or_else(|| {
                        external.as_slice().iter().find_map(|c| {
                            c.module_param_names(&mod_name, member).map(|n| n.to_vec())
                        })
                    });
                let arity = builtin.module_arity(&mod_name, member).or_else(|| {
                    external
                        .as_slice()
                        .iter()
                        .find_map(|c| c.module_arity(&mod_name, member))
                });
                let note = arity.map(|a| format!("§09 `{mod_name}` · {}", a.describe()));
                return match (names, note) {
                    (None, None) => None,
                    (names, note) => Some(Roster {
                        label_head,
                        params: names.unwrap_or_default(),
                        note,
                    }),
                };
            }
            // A `load_module` alias: the member is a binding in the loaded file.
            let dep = loaded_dep_of(db, file, fs, alias)?;
            let params = with_module(db, dep, |m| user_param_names(m, member))?;
            Some(Roster {
                label_head,
                params,
                note: Some("module member".to_string()),
            })
        }
        Head::Bare(name) => {
            // A binding in this module shadows the built-in of the same name
            // (spec §04 "Name resolution"), so it is consulted first.
            if let Some(params) = with_module(db, file, |m| user_param_names(m, name)) {
                return Some(Roster {
                    label_head: name.clone(),
                    params,
                    note: Some("user-defined callable".to_string()),
                });
            }
            let params = builtin
                .base_param_names(name)
                .map(|n| n.to_vec())
                .or_else(|| {
                    external
                        .as_slice()
                        .iter()
                        .find_map(|c| c.base_param_names(name).map(|n| n.to_vec()))
                });
            let arity = builtin
                .base_arity(name)
                .or_else(|| external.as_slice().iter().find_map(|c| c.base_arity(name)));
            if params.is_none() && arity.is_none() {
                return None;
            }
            let note =
                arity.map(|a| format!("{} · {}", builtin.base_param_section(name), a.describe()));
            Some(Roster {
                label_head: name.clone(),
                params: params.unwrap_or_default(),
                note,
            })
        }
    }
}

/// The workspace file loaded by the `load_module` binding named `alias` in
/// `file`, or `None` when `alias` is not such a binding or the path does not
/// resolve.
fn loaded_dep_of(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    alias: &str,
) -> Option<SourceFile> {
    let directive = with_module(db, file, |module| {
        let (_, binding) = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == alias)?;
        let Node::Call(call) = module.node(binding.rhs) else {
            return None;
        };
        let CallHead::Builtin(head) = call.head else {
            return None;
        };
        if module.resolve(head) != "load_module" {
            return None;
        }
        match call.args.first().map(|&a| module.node(a)) {
            Some(Node::Lit(flatppl_core::Scalar::Str(s))) => Some(s.to_string()),
            _ => None,
        }
    })?;
    crate::queries::resolve_path(db, file, &directive, fs)
}

/// Build the `textDocument/signatureHelp` response for the cursor at
/// `byte_offset`, or `None` when the cursor is not inside a call whose head
/// declares a signature.
///
/// The active parameter is the cursor's comma-separated slot, or — when the
/// argument opens `name =` (§05 `KeywordArgs` / `MixedArgs`) — the index of that
/// name in the declared roster. A keyword naming no declared parameter leaves
/// the active parameter unset rather than pointing at an arbitrary slot.
pub fn signature_help(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
) -> Option<lsp_types::SignatureHelp> {
    let site = call_site_at(file.text(db), byte_offset)?;
    let r = roster(db, file, fs, cats, &site.head)?;

    // Lay the label out as `head(p1, p2)` and record each parameter's range so
    // the client can highlight the active one. The label is ASCII, so a byte
    // offset is also a UTF-16 offset.
    let mut label = format!("{}(", r.label_head);
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for (i, p) in r.params.iter().enumerate() {
        if i > 0 {
            label.push_str(", ");
        }
        let start = label.len() as u32;
        label.push_str(p);
        ranges.push((start, label.len() as u32));
    }
    if r.params.is_empty() {
        label.push('…');
    }
    label.push(')');

    let parameters: Vec<lsp_types::ParameterInformation> = r
        .params
        .iter()
        .zip(&ranges)
        .map(|(_, &(s, e))| lsp_types::ParameterInformation {
            label: lsp_types::ParameterLabel::LabelOffsets([s, e]),
            documentation: None,
        })
        .collect();

    let active_parameter = match &site.active {
        Active::Positional(n) => (*n < r.params.len()).then_some(*n as u32),
        Active::Named(name) => r.params.iter().position(|p| p == name).map(|i| i as u32),
    };

    #[allow(deprecated)] // SignatureHelp.active_parameter is deprecated in LSP 3.18
    let help = lsp_types::SignatureHelp {
        signatures: vec![lsp_types::SignatureInformation {
            label,
            documentation: r.note.map(|n| {
                lsp_types::Documentation::MarkupContent(lsp_types::MarkupContent {
                    kind: lsp_types::MarkupKind::Markdown,
                    value: n,
                })
            }),
            parameters: Some(parameters),
            active_parameter,
        }],
        active_signature: Some(0),
        active_parameter,
    };
    Some(help)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn site(text: &str) -> Option<CallSite> {
        // The cursor is marked with `|` in the fixture.
        let byte = text.find('|').expect("fixture marks the cursor with `|`") as u32;
        let stripped = text.replace('|', "");
        call_site_at(&stripped, byte)
    }

    // ── call_site_at ─────────────────────────────────────────────────────────

    #[test]
    fn call_site_reads_the_head_and_first_slot() {
        let s = site("x = Normal(|").expect("inside a call");
        assert_eq!(s.head, Head::Bare("Normal".to_string()));
        assert_eq!(s.active, Active::Positional(0));
    }

    #[test]
    fn call_site_counts_commas_for_the_active_slot() {
        assert_eq!(
            site("x = Normal(0.0, |").expect("in a call").active,
            Active::Positional(1)
        );
        assert_eq!(
            site("x = f(a, b, |").expect("in a call").active,
            Active::Positional(2)
        );
    }

    #[test]
    fn call_site_ignores_commas_nested_in_an_inner_call() {
        let s = site("x = Normal(add(1, 2), |").expect("in a call");
        assert_eq!(s.head, Head::Bare("Normal".to_string()));
        assert_eq!(
            s.active,
            Active::Positional(1),
            "the inner call's comma must not advance the outer slot"
        );
    }

    #[test]
    fn call_site_reports_the_inner_call_when_the_cursor_is_in_it() {
        let s = site("x = Normal(add(1, |").expect("in the inner call");
        assert_eq!(s.head, Head::Bare("add".to_string()));
        assert_eq!(s.active, Active::Positional(1));
    }

    #[test]
    fn call_site_ignores_commas_nested_in_a_vector() {
        let s = site("x = f([1, 2, 3], |").expect("in a call");
        assert_eq!(s.active, Active::Positional(1));
    }

    #[test]
    fn call_site_keeps_the_enclosing_call_from_inside_a_vector_argument() {
        let s = site("x = f(a, [1, |").expect("in the enclosing call");
        assert_eq!(s.head, Head::Bare("f".to_string()));
        assert_eq!(
            s.active,
            Active::Positional(1),
            "the bracket is inside the second argument"
        );
    }

    #[test]
    fn call_site_ignores_a_comma_in_a_string_or_comment() {
        assert_eq!(
            site("x = load_module(\"a,b.flatppl\", |")
                .expect("in a call")
                .active,
            Active::Positional(1),
            "the comma inside the string literal must not count"
        );
        assert_eq!(
            site("x = f(a, % note, more\n  |")
                .expect("in a call")
                .active,
            Active::Positional(1),
            "the comma inside the comment must not count"
        );
    }

    #[test]
    fn call_site_reads_a_member_head() {
        let s = site("x = pp.kallen(|").expect("in a call");
        assert_eq!(
            s.head,
            Head::Member {
                alias: "pp".to_string(),
                member: "kallen".to_string()
            }
        );
    }

    #[test]
    fn call_site_detects_a_keyword_argument() {
        assert_eq!(
            site("x = Normal(mu = |").expect("in a call").active,
            Active::Named("mu".to_string())
        );
        assert_eq!(
            site("x = Normal(0.0, sigma = |").expect("in a call").active,
            Active::Named("sigma".to_string())
        );
    }

    #[test]
    fn call_site_does_not_read_a_comparison_as_a_keyword() {
        assert_eq!(
            site("x = f(a == |").expect("in a call").active,
            Active::Positional(0)
        );
    }

    #[test]
    fn call_site_none_outside_a_call() {
        assert!(site("x = |").is_none());
        assert!(site("x = add(1, 2)\ny = |").is_none(), "the call is closed");
        assert!(site("x = (|").is_none(), "a grouping paren has no head");
    }

    // ── signature_help ───────────────────────────────────────────────────────

    fn help(src: &str) -> Option<lsp_types::SignatureHelp> {
        let byte = src.find('|').expect("fixture marks the cursor") as u32;
        let text = src.replace('|', "");
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), text);
        let fs = FileSet::new(&db, vec![f]);
        let cats = Catalogues::new(&db, vec![]);
        signature_help(&db, f, fs, cats, byte)
    }

    /// The label text a `ParameterLabel::LabelOffsets` entry selects.
    fn param_text(h: &lsp_types::SignatureHelp, i: usize) -> String {
        let sig = &h.signatures[0];
        let p = &sig.parameters.as_ref().expect("parameters")[i];
        let lsp_types::ParameterLabel::LabelOffsets([s, e]) = p.label else {
            panic!("expected label offsets");
        };
        sig.label[s as usize..e as usize].to_string()
    }

    #[test]
    fn signature_help_names_a_builtin_distributions_parameters() {
        // §08 declares `Normal`'s parameters as `mu`, `sigma`; the roster comes
        // from the infer catalogue, not from this crate.
        let h = help("x = Normal(|").expect("signature for Normal");
        let sig = &h.signatures[0];
        assert_eq!(
            sig.label, "Normal(mu, sigma)",
            "the label must carry the catalogue's parameter names"
        );
        assert_eq!(param_text(&h, 0), "mu");
        assert_eq!(param_text(&h, 1), "sigma");
        assert_eq!(h.active_parameter, Some(0));
    }

    #[test]
    fn signature_help_advances_the_active_parameter_with_the_cursor() {
        let h = help("x = Normal(0.0, |").expect("signature");
        assert_eq!(h.active_parameter, Some(1));
    }

    #[test]
    fn signature_help_maps_a_keyword_argument_to_its_declared_slot() {
        // §05 `KeywordArg ::= Name "=" Expression`: the active parameter follows
        // the NAME, not the comma count.
        let h = help("x = Normal(sigma = |").expect("signature");
        assert_eq!(
            h.active_parameter,
            Some(1),
            "`sigma` is the second declared parameter, though it is the first written"
        );
    }

    #[test]
    fn signature_help_leaves_an_unknown_keyword_unhighlighted() {
        let h = help("x = Normal(nope = |").expect("signature");
        assert_eq!(h.active_parameter, None);
    }

    #[test]
    fn signature_help_leaves_an_overrun_positional_unhighlighted() {
        let h = help("x = Normal(0.0, 1.0, |").expect("signature");
        assert_eq!(
            h.active_parameter, None,
            "a third argument to a two-parameter row highlights nothing"
        );
    }

    #[test]
    fn signature_help_documents_the_arity_and_spec_section() {
        let h = help("x = Normal(|").expect("signature");
        let doc = match h.signatures[0].documentation.as_ref().expect("note") {
            lsp_types::Documentation::MarkupContent(m) => m.value.clone(),
            lsp_types::Documentation::String(s) => s.clone(),
        };
        assert!(
            doc.contains("§08") && doc.contains("2 arguments"),
            "the note must carry the spec section and the arity; got {doc:?}"
        );
    }

    #[test]
    fn signature_help_covers_a_builtin_function() {
        let h = help("x = atan2(|").expect("signature for atan2");
        assert_eq!(h.signatures[0].label, "atan2(y, x)");
    }

    #[test]
    fn signature_help_covers_a_user_function_definition() {
        // `f(a, b) = …` lowers to a `functionof` whose boundary carries `a`, `b`.
        let h = help("f(a, b) = add(a, b)\nz = f(|").expect("signature for f");
        assert_eq!(h.signatures[0].label, "f(a, b)");
        assert_eq!(h.active_parameter, Some(0));
    }

    #[test]
    fn signature_help_none_over_an_undeclared_name() {
        assert!(
            help("x = notacallable_zz(|").is_none(),
            "a name no catalogue and no binding declares has no signature"
        );
    }

    #[test]
    fn signature_help_none_outside_a_call() {
        assert!(help("x = |").is_none());
    }

    #[test]
    fn signature_help_covers_a_standard_module_member() {
        let ron = r#"Catalogue(base: [], modules: [Module(name:"myext",version:"0.1",bindings:[Binding(name:"MyDist", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized, params: ["loc","scale"]))])])"#;
        let src = "e = standard_module(\"myext\",\"0.1\")\nx = e.MyDist(";
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let cats = Catalogues::new(&db, vec![ron.to_string()]);
        let h = signature_help(&db, f, fs, cats, src.len() as u32)
            .expect("signature for the external module member");
        assert_eq!(h.signatures[0].label, "e.MyDist(loc, scale)");
        assert_eq!(h.active_parameter, Some(0));
    }

    #[test]
    fn signature_help_covers_a_member_of_a_loaded_file() {
        let db = Database::default();
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "scale(v, k) = mul(v, k)".to_string(),
        );
        let src = "h = load_module(\"helpers.flatppl\")\nz = h.scale(1.0, ";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let h = signature_help(&db, model, fs, cats, src.len() as u32)
            .expect("signature for the cross-file member");
        assert_eq!(h.signatures[0].label, "h.scale(v, k)");
        assert_eq!(h.active_parameter, Some(1));
    }

    #[test]
    fn signature_help_works_while_the_file_does_not_parse() {
        // The whole point of a textual call-site scan: the buffer mid-call is
        // not parseable, and help must still resolve the built-in roster.
        let h = help("mu = 0.0\nx = Normal(mu, |").expect("signature on unparseable text");
        assert_eq!(h.signatures[0].label, "Normal(mu, sigma)");
        assert_eq!(h.active_parameter, Some(1));
    }
}
