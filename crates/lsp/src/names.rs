//! Name geometry and rename legality — the pure-text half of find-references
//! and rename.
//!
//! The IR spans nodes, not names. A binding carries no span for its own name,
//! and a `Ref` node's span covers the whole sub-expression the reference
//! started (`h.shifted` for a member ref, `f(3)` for a call callee — see
//! `flatppl_syntax`'s `alloc_spanned`). Rewriting a name therefore needs the
//! *name* range recovered from the source text, which is what this module does.
//! It also holds the spec §04/§05 rules a new name must satisfy, so the legality
//! decision has one home and is testable without a database.

/// A `Ref` node names either one identifier (`x`) or two (`alias.member`).
/// Which of the two a rename rewrites.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefPart {
    /// The identifier at the start of the node's span: a bare `SelfMod` /
    /// `Local` reference, or the alias half of `alias.member`.
    Head,
    /// The identifier after the `.` in `alias.member`.
    Member,
}

/// Why a name cannot be renamed. The message is user-facing: it goes into the
/// `prepareRename` / `rename` error the editor shows.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal(pub String);

/// Is `b` a byte that may appear in a FlatPPL identifier?
///
/// §05 "Formal grammar" gives `Name ::= (Letter | "_") (Letter | Digit | "_")*`.
/// Only ASCII is recognized, matching the lexer.
#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The half-open byte range of the identifier starting exactly at `at`, or
/// `None` when `at` is not on an identifier start.
fn ident_at(text: &str, at: u32) -> Option<(u32, u32)> {
    let bytes = text.as_bytes();
    let start = at as usize;
    if start >= bytes.len() {
        return None;
    }
    if !(bytes[start].is_ascii_alphabetic() || bytes[start] == b'_') {
        return None;
    }
    let mut end = start;
    while end < bytes.len() && is_ident_byte(bytes[end]) {
        end += 1;
    }
    Some((start as u32, end as u32))
}

/// Is the identifier occupying `[start, end)` a whole word — neither neighbour
/// an identifier byte? Guards against matching `x` inside `xy`.
fn is_whole_word(text: &str, start: u32, end: u32) -> bool {
    let bytes = text.as_bytes();
    let before_ok = start == 0 || !is_ident_byte(bytes[start as usize - 1]);
    let after_ok = end as usize >= bytes.len() || !is_ident_byte(bytes[end as usize]);
    before_ok && after_ok
}

/// The first non-whitespace byte at or after `from`, with its offset.
fn next_significant(text: &str, from: u32) -> Option<(u32, u8)> {
    let bytes = text.as_bytes();
    let mut i = from as usize;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    bytes.get(i).map(|&b| (i as u32, b))
}

/// The end offset of the identifier at `at` — where the alias half of an
/// `alias.member` reference stops.
pub fn head_ident_end(text: &str, at: u32) -> Option<u32> {
    ident_at(text, at).map(|(_, e)| e)
}

/// The byte range of the `name` an occurrence of a reference designates, given
/// the `Ref` node's span and which half to take.
///
/// Fails closed: the extracted text must equal `name`, so a span the parser
/// shaped differently than assumed yields `None` instead of a wrong edit.
pub fn ref_name_range(
    text: &str,
    span_start: u32,
    part: RefPart,
    name: &str,
) -> Option<(u32, u32)> {
    let head = ident_at(text, span_start)?;
    let (start, end) = match part {
        RefPart::Head => head,
        RefPart::Member => {
            let (dot, b) = next_significant(text, head.1)?;
            if b != b'.' {
                return None;
            }
            let (member_start, _) = next_significant(text, dot + 1)?;
            ident_at(text, member_start)?
        }
    };
    (&text[start as usize..end as usize] == name).then_some((start, end))
}

/// The byte range of a binding's own name at its definition site.
///
/// `rhs_start` is the start of the binding's RHS node span. Two shapes occur:
///
/// - A `~` binding's RHS is the `draw` call, whose span starts at the binding
///   name itself (`x ~ Normal(…)` spans from `x`), so the name sits at
///   `rhs_start`.
/// - Otherwise the name precedes the RHS, separated by `=` (plus a parameter
///   list for a `FunctionDefinition`). Take the LAST whole-word occurrence of
///   `name` before `rhs_start` that is followed by `=`, `~` or `(` — the
///   definition site is always the closest such occurrence to its own RHS, so
///   an earlier keyword argument, comment, or string of the same spelling
///   cannot win. The `(` case admits `f(a) = …`; requiring a following
///   operator is also what rejects a parameter that shares the function's name
///   (`f(f) = …`, where the parameter is followed by `)`).
pub fn def_name_range(text: &str, rhs_start: u32, name: &str) -> Option<(u32, u32)> {
    if let Some((s, e)) = ident_at(text, rhs_start) {
        if &text[s as usize..e as usize] == name {
            return Some((s, e));
        }
    }
    let window = text.get(..rhs_start as usize)?;
    let mut best = None;
    let mut from = 0usize;
    while let Some(rel) = window[from..].find(name) {
        let start = (from + rel) as u32;
        let end = start + name.len() as u32;
        from = start as usize + 1;
        if !is_whole_word(text, start, end) {
            continue;
        }
        if matches!(next_significant(text, end), Some((_, b'=' | b'~' | b'('))) {
            best = Some((start, end));
        }
    }
    best
}

/// Reserved names a binding may not take.
///
/// §05 "Note on reserved words": "The keywords `in`, `true`, `false`, `all`,
/// and `only` are recognized before `Name` and cannot be used as bindings. The
/// top-level binding names `inputs` and `outputs` are reserved for the
/// determinization signature." §04 "Name resolution" adds: "`self` and `base`
/// themselves are reserved names and cannot be bound to something else."
const RESERVED: &[&str] = &[
    "in", "true", "false", "all", "only", "inputs", "outputs", "self", "base",
];

/// Does `name` match §04's public-binding form?
///
/// §04 "Binding names": "Names that do not begin with an underscore are public:
/// they form the interface of a FlatPPL module. They must match the regular
/// expression `^[A-Za-z][A-Za-z0-9_]*$`."
fn is_public_form(name: &str) -> bool {
    let mut cs = name.bytes();
    cs.next().is_some_and(|b| b.is_ascii_alphabetic()) && cs.all(is_ident_byte)
}

/// Does `name` match §04's private-binding form?
///
/// §04 "Binding names": "Binding names that begin with a single underscore and
/// do not end with an underscore (regular expression
/// `^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?$`), e.g. `_tmp`, are private to a
/// module."
///
/// Written out rather than regexed: a leading `_`, then a letter, then
/// identifier bytes, and the last byte is alphanumeric. The single-underscore
/// requirement excludes the `__`-prefixed auto-generated form, and the
/// no-trailing-underscore requirement excludes the `_name_` placeholder form.
fn is_private_form(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('_') else {
        return false;
    };
    let mut cs = rest.bytes();
    if !cs.next().is_some_and(|b| b.is_ascii_alphabetic()) {
        return false;
    }
    rest.bytes().all(is_ident_byte)
        && rest
            .bytes()
            .next_back()
            .is_some_and(|b| b.is_ascii_alphanumeric())
}

/// Is `name` private, i.e. outside the module's public interface?
///
/// §04 "Binding names" splits on the leading underscore alone, which is also
/// how the parser sets `Binding::public`.
pub fn is_private(name: &str) -> bool {
    name.starts_with('_')
}

/// Check `new_name` as a replacement binding name, per §04 "Binding names" and
/// §05 "Note on reserved words". `Ok(())` when it is a legal public or private
/// binding name.
///
/// Rejected shapes and why: the bare `_` discard and the `_name_` placeholder
/// form are reserved by §04 for the discard and for `functionof`/`kernelof`
/// respectively; the `__`-prefixed form is reserved for engine-generated names.
/// None of the three is a name an author may bind, so none is a rename target.
pub fn check_new_name(new_name: &str) -> Result<(), Refusal> {
    if new_name.is_empty() {
        return Err(Refusal("a binding name cannot be empty".to_string()));
    }
    if RESERVED.contains(&new_name) {
        return Err(Refusal(format!(
            "`{new_name}` is a reserved name and cannot be bound (spec §04/§05)"
        )));
    }
    if is_public_form(new_name) || is_private_form(new_name) {
        return Ok(());
    }
    Err(Refusal(format!(
        "`{new_name}` is not a legal binding name: spec §04 admits public names \
         matching `^[A-Za-z][A-Za-z0-9_]*$` and private names matching \
         `^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?$`"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ident_at / whole-word ────────────────────────────────────────────────

    #[test]
    fn ident_at_reads_the_identifier_and_stops() {
        assert_eq!(ident_at("abc + d", 0), Some((0, 3)));
        assert_eq!(ident_at("_priv = 1", 0), Some((0, 5)));
        assert_eq!(ident_at("a1_b, c", 0), Some((0, 4)));
        // A digit cannot start a §05 `Name`.
        assert_eq!(ident_at("1abc", 0), None);
        assert_eq!(ident_at("x", 5), None);
    }

    #[test]
    fn whole_word_rejects_a_substring_of_a_longer_name() {
        // `x` inside `xy` is not an occurrence of `x`.
        assert!(!is_whole_word("xy", 0, 1));
        assert!(is_whole_word("x y", 0, 1));
        assert!(is_whole_word("x", 0, 1));
    }

    // ── ref_name_range ───────────────────────────────────────────────────────

    #[test]
    fn ref_name_range_head_takes_the_leading_identifier() {
        // A `Ref` node for a call callee spans the whole call (`f(3)`), so the
        // name range must be the head identifier only.
        let text = "z = f(3)";
        assert_eq!(ref_name_range(text, 4, RefPart::Head, "f"), Some((4, 5)));
    }

    #[test]
    fn ref_name_range_member_takes_the_identifier_after_the_dot() {
        // The `Ref` node for `h.shifted` spans all of `h.shifted`.
        let text = "v = h.shifted";
        assert_eq!(
            ref_name_range(text, 4, RefPart::Member, "shifted"),
            Some((6, 13))
        );
        assert_eq!(ref_name_range(text, 4, RefPart::Head, "h"), Some((4, 5)));
    }

    #[test]
    fn ref_name_range_member_tolerates_space_around_the_dot() {
        let text = "v = h . shifted";
        assert_eq!(
            ref_name_range(text, 4, RefPart::Member, "shifted"),
            Some((8, 15))
        );
    }

    #[test]
    fn ref_name_range_fails_closed_on_a_name_mismatch() {
        // Guards the whole rename path: if the recovered text is not the name
        // we expect, no edit is produced.
        let text = "v = h.shifted";
        assert_eq!(ref_name_range(text, 4, RefPart::Member, "other"), None);
        assert_eq!(ref_name_range(text, 4, RefPart::Member, "h"), None);
    }

    #[test]
    fn ref_name_range_member_needs_a_dot() {
        let text = "v = add(h, 1)";
        assert_eq!(ref_name_range(text, 8, RefPart::Member, "h"), None);
    }

    // ── def_name_range ───────────────────────────────────────────────────────

    #[test]
    fn def_name_range_plain_binding() {
        let text = "x = 1";
        assert_eq!(def_name_range(text, 4, "x"), Some((0, 1)));
    }

    #[test]
    fn def_name_range_tilde_binding_starts_at_the_name() {
        // The `draw` RHS span starts at the binding name for a `~` binding.
        let text = "mu = 0.0\nx ~ Normal(mu, 1.0)";
        assert_eq!(def_name_range(text, 9, "x"), Some((9, 10)));
    }

    #[test]
    fn def_name_range_function_definition() {
        // `f(a, b) = add(a, b)`: the RHS span starts at `add`, and the name is
        // followed by `(`, not `=`.
        let text = "f(a, b) = add(a, b)";
        assert_eq!(def_name_range(text, 10, "f"), Some((0, 1)));
    }

    #[test]
    fn def_name_range_ignores_a_parameter_of_the_same_name() {
        // The parameter `f` is followed by `)`, so only the definition site
        // (followed by `(`) qualifies.
        let text = "f(f) = add(f, 1)";
        assert_eq!(def_name_range(text, 7, "f"), Some((0, 1)));
    }

    #[test]
    fn def_name_range_ignores_a_longer_name_containing_it() {
        let text = "xy = 1\nx = 2";
        assert_eq!(def_name_range(text, 11, "x"), Some((7, 8)));
    }

    #[test]
    fn def_name_range_ignores_the_name_in_an_earlier_comment_or_string() {
        let text = "% see x = 9\np = load_module(\"x.flatppl\")\nx = 1";
        // The definition is the last qualifying occurrence before its own RHS.
        assert_eq!(def_name_range(text, 45, "x"), Some((41, 42)));
    }

    #[test]
    fn def_name_range_ignores_an_earlier_keyword_argument() {
        let text = "g = Normal(mu = 0.0, sigma = 1.0)\nmu = 0.0";
        assert_eq!(def_name_range(text, 39, "mu"), Some((34, 36)));
    }

    #[test]
    fn def_name_range_none_when_the_name_is_absent() {
        assert_eq!(def_name_range("x = 1", 4, "zzz"), None);
    }

    // ── name legality ────────────────────────────────────────────────────────

    #[test]
    fn public_and_private_forms_are_accepted() {
        for ok in ["x", "mu", "Normal2", "a_b_c", "_tmp", "_a", "_a_b", "_x9"] {
            assert!(
                check_new_name(ok).is_ok(),
                "`{ok}` must be a legal binding name"
            );
        }
    }

    #[test]
    fn reserved_words_are_refused() {
        for bad in [
            "in", "true", "false", "all", "only", "inputs", "outputs", "self", "base",
        ] {
            let err = check_new_name(bad).expect_err("reserved name must refuse");
            assert!(
                err.0.contains("reserved"),
                "refusal must say why; got {}",
                err.0
            );
        }
    }

    #[test]
    fn placeholder_discard_and_generated_forms_are_refused() {
        // `_` is the §04 discard; `_x_` is the placeholder form reserved for
        // `functionof`/`kernelof`; `__x` is the engine-generated form.
        for bad in ["_", "__", "_x_", "_a_b_", "__gen", "__1"] {
            assert!(
                check_new_name(bad).is_err(),
                "`{bad}` is reserved by spec §04 and must refuse"
            );
        }
    }

    #[test]
    fn malformed_names_are_refused() {
        for bad in ["", "1x", "x-y", "x y", "x.y", "é"] {
            assert!(check_new_name(bad).is_err(), "`{bad}` must refuse");
        }
    }

    #[test]
    fn is_private_splits_on_the_leading_underscore() {
        assert!(is_private("_tmp"));
        assert!(!is_private("tmp"));
    }
}
