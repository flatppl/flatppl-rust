//! Binding names → display names (`NOTATION.md` "Names").
//!
//! The rules are deliberately small and rule-based: a Greek or single-letter
//! head takes subscripts from its trailing digits and underscore segments;
//! every other name prints whole as upright text. Nothing here reads the
//! module — the same function serves binding names, lambda parameters, record
//! fields and axis names.

/// A rendered identifier: a head plus subscript parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayName {
    pub head: Atom,
    /// Subscript parts, comma-separated when printed. Empty for a plain head.
    pub subs: Vec<Atom>,
}

/// One piece of a display name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Atom {
    /// A Greek letter (already resolved to its Unicode code point).
    Greek(char),
    /// A single Latin letter, printed in italics.
    Letter(char),
    /// A digit string (a trailing `1` in `theta1`, or a `_12` segment).
    Digits(String),
    /// A word, printed upright.
    Word(String),
}

impl DisplayName {
    /// A plain upright word with no subscripts (operator names, roman heads).
    pub fn word(text: impl Into<String>) -> Self {
        DisplayName {
            head: Atom::Word(text.into()),
            subs: Vec::new(),
        }
    }
}

/// Render `name` by the `NOTATION.md` name rules.
pub fn display_name(name: &str) -> DisplayName {
    // A leading underscore marks a private or generated name: no segment rules.
    if name.starts_with('_') || name.is_empty() {
        return DisplayName::word(name);
    }
    let mut segments = name.split('_');
    let head_seg = segments.next().unwrap_or("");
    let (head_alpha, head_digits) = split_trailing_digits(head_seg);
    let head = match classify_head(head_alpha) {
        Some(atom) => atom,
        // A word head keeps the whole name, underscores and digits included.
        None => return DisplayName::word(name),
    };
    let mut subs = Vec::new();
    if !head_digits.is_empty() {
        subs.push(Atom::Digits(head_digits.to_string()));
    }
    for seg in segments {
        if seg.is_empty() {
            // `a__b`: a doubled underscore is not a subscript separator.
            return DisplayName::word(name);
        }
        subs.push(classify_segment(seg));
    }
    DisplayName { head, subs }
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

#[cfg(test)]
mod tests {
    use super::Atom::*;
    use super::*;

    fn dn(name: &str) -> DisplayName {
        display_name(name)
    }

    #[test]
    fn greek_heads_become_letters() {
        assert_eq!(
            dn("mu"),
            DisplayName {
                head: Greek('μ'),
                subs: vec![]
            }
        );
        assert_eq!(
            dn("Gamma"),
            DisplayName {
                head: Greek('Γ'),
                subs: vec![]
            }
        );
        assert_eq!(dn("varphi").head, Greek('ϕ'));
    }

    #[test]
    fn single_letters_are_italic_letters() {
        assert_eq!(
            dn("J"),
            DisplayName {
                head: Letter('J'),
                subs: vec![]
            }
        );
        assert_eq!(dn("x").head, Letter('x'));
    }

    #[test]
    fn trailing_digits_subscript_a_greek_or_letter_head() {
        assert_eq!(
            dn("theta1"),
            DisplayName {
                head: Greek('θ'),
                subs: vec![Digits("1".into())]
            }
        );
        assert_eq!(
            dn("s12"),
            DisplayName {
                head: Letter('s'),
                subs: vec![Digits("12".into())]
            }
        );
        assert_eq!(dn("c0").subs, vec![Digits("0".into())]);
    }

    #[test]
    fn underscore_segments_subscript_a_greek_or_letter_head() {
        assert_eq!(
            dn("mu_a"),
            DisplayName {
                head: Greek('μ'),
                subs: vec![Letter('a')]
            }
        );
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
