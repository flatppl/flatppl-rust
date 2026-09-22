//! Binding names → display names (`NOTATION.md` "Names").
//!
//! The rules are deliberately small and rule-based: a Greek or single-letter
//! head takes subscripts from its trailing digits and underscore segments;
//! every other name prints whole as upright text. Nothing here reads the
//! module — the same function serves binding names, lambda parameters, record
//! fields and axis names.

/// A rendered identifier: a head plus subscript parts, optionally squared or
/// under a root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayName {
    pub head: Atom,
    /// Subscript parts, comma-separated when printed. Empty for a plain head.
    pub subs: Vec<Atom>,
    /// A marker the name carries: `_sq` squares the symbol (`sigma_sq` → σ²),
    /// `_sqrt` puts it under a root (`s_sqrt` → √s), `log_` takes its
    /// logarithm (`log_sigma` → log σ).
    pub wrap: Option<Wrap>,
}

/// What a name marker does to the symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    Squared,
    Sqrt,
    Log,
}

/// One piece of a display name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Atom {
    /// A Greek letter (already resolved to its Unicode code point).
    Greek(char),
    /// A single Latin letter, printed in italics.
    Letter(char),
    /// A single Latin capital in script style (`ℒ`): the head of a binding
    /// the lowering restyles by its inferred type — a likelihood named `L`.
    Script(char),
    /// A digit string (a trailing `1` in `theta1`, or a `_12` segment).
    Digits(String),
    /// A word, printed upright.
    Word(String),
}

impl DisplayName {
    /// A head with its subscript parts, unmarked.
    pub fn new(head: Atom, subs: Vec<Atom>) -> Self {
        DisplayName {
            head,
            subs,
            wrap: None,
        }
    }

    /// A plain upright word with no subscripts (operator names, roman heads).
    pub fn word(text: impl Into<String>) -> Self {
        DisplayName::new(Atom::Word(text.into()), Vec::new())
    }

    /// The same symbol under a marker's effect.
    pub fn wrapped(mut self, wrap: Wrap) -> Self {
        self.wrap = Some(wrap);
        self
    }
}

/// A name marker: a fixed affix and what it does to the symbol the rest of
/// the name spells. The table is the whole rule; adding a marker is one row.
struct Marker {
    affix: Affix,
    text: &'static str,
    wrap: Wrap,
}

enum Affix {
    Prefix,
    Suffix,
}

const MARKERS: &[Marker] = &[
    Marker {
        affix: Affix::Suffix,
        text: "_sq",
        wrap: Wrap::Squared,
    },
    Marker {
        affix: Affix::Suffix,
        text: "_sqrt",
        wrap: Wrap::Sqrt,
    },
    Marker {
        affix: Affix::Prefix,
        text: "log_",
        wrap: Wrap::Log,
    },
];

/// `name` without its marker, and the marker's effect, when it carries one.
/// The remainder must be a name of its own (non-empty, not starting or
/// ending in `_`).
fn strip_marker(name: &str) -> Option<(&str, Wrap)> {
    MARKERS.iter().find_map(|m| {
        let inner = match m.affix {
            Affix::Prefix => name.strip_prefix(m.text)?,
            Affix::Suffix => name.strip_suffix(m.text)?,
        };
        (!inner.is_empty() && !inner.starts_with('_') && !inner.ends_with('_'))
            .then_some((inner, m.wrap))
    })
}

/// Render `name` by the `NOTATION.md` name rules: strip one marker, apply the
/// segment rules to the rest, wrap. A name with two markers, a leading
/// underscore, a word head or a doubled underscore prints as written.
pub fn display_name(name: &str) -> DisplayName {
    if name.starts_with('_') || name.is_empty() {
        return DisplayName::word(name);
    }
    match strip_marker(name) {
        Some((inner, wrap)) => match strip_marker(inner) {
            Some(_) => DisplayName::word(name),
            // A word inside a marker is still marked: `rate_sq` is rate².
            None => segments(inner)
                .unwrap_or_else(|| DisplayName::word(inner))
                .wrapped(wrap),
        },
        None => segments(name).unwrap_or_else(|| DisplayName::word(name)),
    }
}

/// The segment rules: a Greek or single-letter head takes subscripts from its
/// trailing digits and underscore segments. `None` for a word head or a
/// doubled underscore, which print as written.
fn segments(name: &str) -> Option<DisplayName> {
    let mut segments = name.split('_');
    let head_seg = segments.next().unwrap_or("");
    let (head_alpha, head_digits) = split_trailing_digits(head_seg);
    let head = classify_head(head_alpha)?;
    let mut subs = Vec::new();
    if !head_digits.is_empty() {
        subs.push(Atom::Digits(head_digits.to_string()));
    }
    for seg in segments {
        if seg.is_empty() {
            // `a__b`: a doubled underscore is not a subscript separator.
            return None;
        }
        subs.push(classify_segment(seg));
    }
    Some(DisplayName::new(head, subs))
}

/// Split `theta1` into `("theta", "1")`; a segment with no trailing digits keeps
/// an empty digit part. An all-digit segment counts as digits with an empty
/// alphabetic part.
fn split_trailing_digits(seg: &str) -> (&str, &str) {
    let cut = seg
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_digit())
        .last()
        .map(|(i, _)| i)
        .unwrap_or(seg.len());
    seg.split_at(cut)
}

/// A head that may carry subscripts: a Greek name or one Latin letter.
fn classify_head(alpha: &str) -> Option<Atom> {
    if let Some(g) = greek(alpha) {
        return Some(Atom::Greek(g));
    }
    let mut chars = alpha.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if c.is_ascii_alphabetic() => Some(Atom::Letter(c)),
        _ => None,
    }
}

/// A subscript segment: Greek, letter or digits by the head rules, else a word.
fn classify_segment(seg: &str) -> Atom {
    if seg.chars().all(|c| c.is_ascii_digit()) {
        return Atom::Digits(seg.to_string());
    }
    classify_head(seg).unwrap_or_else(|| Atom::Word(seg.to_string()))
}

/// The Greek letter a spelled-out name denotes, if any. Lowercase and
/// capitalised spellings, plus the LaTeX `var` forms.
pub fn greek(name: &str) -> Option<char> {
    Some(match name {
        "alpha" => 'α',
        "beta" => 'β',
        "gamma" => 'γ',
        "delta" => 'δ',
        "epsilon" => 'ε',
        "varepsilon" => 'ϵ',
        "zeta" => 'ζ',
        "eta" => 'η',
        "theta" => 'θ',
        "vartheta" => 'ϑ',
        "iota" => 'ι',
        "kappa" => 'κ',
        "lambda" => 'λ',
        "mu" => 'μ',
        "nu" => 'ν',
        "xi" => 'ξ',
        "omicron" => 'ο',
        "pi" => 'π',
        "varpi" => 'ϖ',
        "rho" => 'ρ',
        "varrho" => 'ϱ',
        "sigma" => 'σ',
        "varsigma" => 'ς',
        "tau" => 'τ',
        "upsilon" => 'υ',
        "phi" => 'φ',
        "varphi" => 'ϕ',
        "chi" => 'χ',
        "psi" => 'ψ',
        "omega" => 'ω',
        "Alpha" => 'Α',
        "Beta" => 'Β',
        "Gamma" => 'Γ',
        "Delta" => 'Δ',
        "Epsilon" => 'Ε',
        "Zeta" => 'Ζ',
        "Eta" => 'Η',
        "Theta" => 'Θ',
        "Iota" => 'Ι',
        "Kappa" => 'Κ',
        "Lambda" => 'Λ',
        "Mu" => 'Μ',
        "Nu" => 'Ν',
        "Xi" => 'Ξ',
        "Omicron" => 'Ο',
        "Pi" => 'Π',
        "Rho" => 'Ρ',
        "Sigma" => 'Σ',
        "Tau" => 'Τ',
        "Upsilon" => 'Υ',
        "Phi" => 'Φ',
        "Chi" => 'Χ',
        "Psi" => 'Ψ',
        "Omega" => 'Ω',
        _ => return None,
    })
}

/// The Unicode Mathematical Script capital for an ASCII capital (`L` → `ℒ`).
/// Eight of the twenty-six live in the Letterlike Symbols block, the rest in
/// the Mathematical Alphanumeric Symbols block.
pub fn script_capital(c: char) -> char {
    match c {
        'B' => 'ℬ',
        'E' => 'ℰ',
        'F' => 'ℱ',
        'H' => 'ℋ',
        'I' => 'ℐ',
        'L' => 'ℒ',
        'M' => 'ℳ',
        'R' => 'ℛ',
        'A'..='Z' => char::from_u32(0x1D49C + (c as u32 - 'A' as u32)).unwrap_or(c),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::Atom::*;
    use super::*;

    fn dn(name: &str) -> DisplayName {
        display_name(name)
    }

    #[test]
    fn greek_heads_become_letters() {
        assert_eq!(dn("mu"), DisplayName::new(Greek('μ'), vec![]));
        assert_eq!(dn("Gamma"), DisplayName::new(Greek('Γ'), vec![]));
        assert_eq!(dn("varphi").head, Greek('ϕ'));
    }

    #[test]
    fn single_letters_are_italic_letters() {
        assert_eq!(dn("J"), DisplayName::new(Letter('J'), vec![]));
        assert_eq!(dn("x").head, Letter('x'));
    }

    #[test]
    fn trailing_digits_subscript_a_greek_or_letter_head() {
        assert_eq!(
            dn("theta1"),
            DisplayName::new(Greek('θ'), vec![Digits("1".into())])
        );
        assert_eq!(
            dn("s12"),
            DisplayName::new(Letter('s'), vec![Digits("12".into())])
        );
        assert_eq!(dn("c0").subs, vec![Digits("0".into())]);
    }

    #[test]
    fn underscore_segments_subscript_a_greek_or_letter_head() {
        assert_eq!(dn("mu_a"), DisplayName::new(Greek('μ'), vec![Letter('a')]));
        assert_eq!(dn("sigma_B").subs, vec![Letter('B')]);
        assert_eq!(dn("S_mu").subs, vec![Greek('μ')]);
        assert_eq!(dn("y_data").subs, vec![Word("data".into())]);
        assert_eq!(
            dn("E1_data").subs,
            vec![Digits("1".into()), Word("data".into())]
        );
        assert_eq!(
            dn("Z0_12").subs,
            vec![Digits("0".into()), Digits("12".into())]
        );
    }

    #[test]
    fn word_heads_keep_the_whole_name() {
        for name in [
            "prior",
            "alphaxy",
            "forward_kernel",
            "rcp_max",
            "yS",
            "mcstat_0",
            "poly6",
        ] {
            assert_eq!(dn(name), DisplayName::word(name), "{name}");
        }
    }

    #[test]
    fn the_sigma_squared_wart_is_a_subscript_by_rule() {
        // NOTATION.md names this case: the rule cannot know σ² was meant.
        assert_eq!(dn("sigma2").subs, vec![Digits("2".into())]);
    }

    #[test]
    fn private_and_odd_names_are_words() {
        assert_eq!(dn("_tmp"), DisplayName::word("_tmp"));
        assert_eq!(dn("__0x1f"), DisplayName::word("__0x1f"));
        assert_eq!(dn("a__b"), DisplayName::word("a__b"));
        assert_eq!(dn("x_"), DisplayName::word("x_"));
    }
}
