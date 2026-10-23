//! Spec §04 "Additional arguments": the one binder for the additional arguments
//! of a callback.
//!
//! Some operations call a function or kernel with leading arguments of their own
//! (`weighted(f, M, ...)` calls `f(x, ...)`, `scan(f, init, ...)` calls
//! `f(acc, ...)`). The additional arguments `...` bind the callback's remaining
//! inputs under the ordinary calling convention, "positionally or by name". A sole
//! additional record or table "passes whole if the remaining inputs accept one
//! positional argument. Otherwise, its fields or columns bind the remaining inputs
//! by name. No input may be left unbound or bound twice."
//!
//! Inference and the determinizer both read their callbacks through this function,
//! so there is one statement of the rule.

/// The remaining inputs of a callback, after the leading arguments the operation
/// supplies itself.
pub struct Remaining<'a> {
    /// Input names in declaration order.
    pub names: &'a [String],
    /// Whether the callback's inputs are ordered (positional binding allowed).
    pub ordered: bool,
}

/// Where one remaining input takes its value from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The `k`-th positional additional argument.
    Positional(usize),
    /// The `j`-th keyword additional argument.
    Named(usize),
    /// The `i`-th field or column of the sole additional record or table.
    Field(usize),
}

/// Why the additional arguments do not bind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A positional argument was given to a callback with keyword-only inputs.
    KeywordOnly,
    /// More positional arguments than remaining inputs.
    TooMany { given: usize, inputs: usize },
    /// A keyword or field names no remaining input.
    Unknown(String),
    /// An input is bound both positionally and by keyword.
    Twice(String),
    /// A remaining input receives no argument.
    Unbound(String),
}

impl Remaining<'_> {
    /// True when the remaining inputs accept one positional argument, in which
    /// case a sole additional record or table is passed whole.
    pub fn accepts_one_positional(&self) -> bool {
        self.ordered && self.names.len() == 1
    }
}

/// Bind `positional` positional arguments, the keyword arguments `named`, and,
/// when the sole additional argument is a record or table, its field names
/// `aggregate`, to `remaining`. Returns the source of every remaining input in
/// declaration order.
pub fn bind(
    remaining: &Remaining<'_>,
    positional: usize,
    named: &[&str],
    aggregate: Option<&[String]>,
) -> Result<Vec<Source>, Error> {
    let find = |name: &str| remaining.names.iter().position(|n| n == name);
    let mut slots: Vec<Option<Source>> = vec![None; remaining.names.len()];
    if let Some(fields) = aggregate
        && positional == 1
        && named.is_empty()
        && !remaining.accepts_one_positional()
    {
        for (i, field) in fields.iter().enumerate() {
            let at = find(field).ok_or_else(|| Error::Unknown(field.clone()))?;
            slots[at] = Some(Source::Field(i));
        }
    } else {
        if positional > 0 && !remaining.ordered {
            return Err(Error::KeywordOnly);
        }
        if positional > remaining.names.len() {
            return Err(Error::TooMany {
                given: positional,
                inputs: remaining.names.len(),
            });
        }
        for (k, slot) in slots.iter_mut().take(positional).enumerate() {
            *slot = Some(Source::Positional(k));
        }
        for (j, name) in named.iter().enumerate() {
            let at = find(name).ok_or_else(|| Error::Unknown((*name).to_string()))?;
            if slots[at].replace(Source::Named(j)).is_some() {
                return Err(Error::Twice((*name).to_string()));
            }
        }
    }
    slots
        .into_iter()
        .zip(remaining.names)
        .map(|(slot, name)| slot.ok_or_else(|| Error::Unbound(name.clone())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn positional_keyword_and_record_forms_bind_the_same_inputs() {
        let step = names(&["drift", "scale"]);
        let remaining = Remaining {
            names: &step,
            ordered: true,
        };
        use Source::*;
        assert_eq!(
            bind(&remaining, 2, &[], None),
            Ok(vec![Positional(0), Positional(1)])
        );
        assert_eq!(
            bind(&remaining, 0, &["scale", "drift"], None),
            Ok(vec![Named(1), Named(0)])
        );
        let fields = names(&["scale", "drift"]);
        assert_eq!(
            bind(&remaining, 1, &[], Some(&fields)),
            Ok(vec![Field(1), Field(0)])
        );
    }

    #[test]
    fn a_sole_record_passes_whole_when_one_positional_input_remains() {
        let one = names(&["p"]);
        let remaining = Remaining {
            names: &one,
            ordered: true,
        };
        let fields = names(&["a", "b"]);
        assert_eq!(
            bind(&remaining, 1, &[], Some(&fields)),
            Ok(vec![Source::Positional(0)])
        );
        // Keyword-only inputs never take a positional argument, so a record binds by name.
        let remaining = Remaining {
            names: &one,
            ordered: false,
        };
        let fields = names(&["p"]);
        assert_eq!(
            bind(&remaining, 1, &[], Some(&fields)),
            Ok(vec![Source::Field(0)])
        );
    }

    #[test]
    fn unbound_twice_unknown_and_surplus_are_errors() {
        let step = names(&["a", "b"]);
        let remaining = Remaining {
            names: &step,
            ordered: true,
        };
        assert_eq!(
            bind(&remaining, 1, &[], None),
            Err(Error::Unbound("b".into()))
        );
        assert_eq!(
            bind(&remaining, 1, &["a"], None),
            Err(Error::Twice("a".into()))
        );
        assert_eq!(
            bind(&remaining, 0, &["a", "c"], None),
            Err(Error::Unknown("c".into()))
        );
        assert_eq!(
            bind(&remaining, 3, &[], None),
            Err(Error::TooMany {
                given: 3,
                inputs: 2
            })
        );
        let keyword_only = Remaining {
            names: &step,
            ordered: false,
        };
        assert_eq!(bind(&keyword_only, 1, &[], None), Err(Error::KeywordOnly));
    }
}
