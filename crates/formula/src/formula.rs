use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Formula {
    pub response: String,
    pub intercept: bool,
    pub fixed: Vec<String>,
    pub groups: Vec<String>,
}

/// Parse additive numeric terms and `(1 | group)` random intercepts.
///
/// Interactions, calls, response additions and factor coding are not yet supported.
/// Group terms are ordered using C collation, matching the pinned brms fixtures.
pub fn parse_formula(text: &str) -> Result<Formula> {
    let (response, rhs) = text
        .split_once('~')
        .ok_or_else(|| Error("formula must contain `response ~ terms`".into()))?;
    let response = name(response.trim())?;
    let mut out = Formula {
        response,
        intercept: true,
        fixed: Vec::new(),
        groups: Vec::new(),
    };
    let mut depth = 0i32;
    let mut start = 0;
    let mut add = true;
    let mut groups = Vec::new();
    let mut term = |raw: &str, add: bool| -> Result<()> {
        let raw = raw.trim();
        match raw {
            "1" => out.intercept = add,
            "0" => out.intercept = !add,
            _ if raw.starts_with('(') && raw.ends_with(')') => {
                let group = raw[1..raw.len() - 1]
                    .split_once('|')
                    .filter(|(effect, _)| effect.trim() == "1")
                    .ok_or_else(|| {
                        Error(format!(
                            "unsupported formula term `{raw}`; expected `(1 | group)`"
                        ))
                    })?;
                // Their laws agree for one coefficient, but R formula removal
                // distinguishes `|` and `||` terms before brms checks duplicates.
                let tail = group.1.trim();
                let independent = tail.starts_with('|');
                let group = name(tail.strip_prefix('|').unwrap_or(tail).trim())?;
                update(&mut groups, (group, independent), add);
            }
            _ => update(&mut out.fixed, name(raw)?, add),
        }
        Ok(())
    };
    for (at, ch) in rhs.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            '+' | '-' if depth == 0 => {
                if !rhs[start..at].trim().is_empty() {
                    term(&rhs[start..at], add)?;
                    add = ch == '+';
                } else if ch == '-' {
                    add = !add;
                }
                start = at + 1;
            }
            _ => {}
        }
        if depth < 0 {
            return Err(Error("unbalanced parentheses in formula".into()));
        }
    }
    if depth != 0 {
        return Err(Error("unbalanced parentheses in formula".into()));
    }
    term(&rhs[start..], add)?;
    groups.sort();
    for (group, _) in groups {
        if out.groups.last() == Some(&group) {
            return Err(Error(format!(
                "duplicated group-level intercept for `{group}`"
            )));
        }
        out.groups.push(group);
    }
    Ok(out)
}

fn update<T: PartialEq>(terms: &mut Vec<T>, name: T, add: bool) {
    if !add {
        terms.retain(|term| term != &name);
    } else if !terms.contains(&name) {
        terms.push(name);
    }
}

fn name(text: &str) -> Result<String> {
    let valid = text.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    if !valid
        || [
            "Intercept",
            "TRUE",
            "FALSE",
            "NULL",
            "Inf",
            "NaN",
            "NA",
            "NA_integer_",
            "NA_real_",
            "NA_complex_",
            "NA_character_",
        ]
        .contains(&text)
    {
        return Err(Error(format!(
            "unsupported formula term `{text}`; expected a plain column name"
        )));
    }
    Ok(text.into())
}
