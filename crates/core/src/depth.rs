//! A recursion-depth budget shared by every recursive descent in the workspace.
//!
//! Deeply nested input used to abort the process with a stack overflow rather
//! than refuse: the FlatPPL source parser died above depth 5500, the FlatPIR
//! s-expression reader above 8000, and the HS3 expression parser above 10000. A
//! shallow document can reach the same place, because a reduction over a wide
//! operand array builds a tree as deep as the array is long.
//!
//! Two guards existed before this, disagreeing: `flatpir`'s JSON reader capped
//! at 128 and one determiniser walk capped at 64. This is the single mechanism
//! they are replaced by.
//!
//! It is DELIBERATELY not a visited set. A visited set answers a different
//! question — whether a DAG is being re-walked as a tree, which is a complexity
//! problem rather than a stack-safety one — and mixing the two hides which is
//! doing the work at any site.
//!
//! # Choosing [`DEFAULT_MAX_DEPTH`]
//!
//! Measured over every corpus in the workspace: the deepest real model is 14
//! levels (a `flatppl-testsuite` corpus model; the examples reach 10, the pyhf
//! import corpus 9, the HS3 fixtures fewer). The default is the smallest power
//! of two at or above eight times that, and it sits well under the shallowest
//! measured crash floor, so it refuses long before the stack is at risk while
//! leaving nine times the headroom any real model has ever needed.

/// Maximum recursion depth a walk may reach before it refuses.
///
/// 128: eight times the deepest model in any workspace corpus (14), rounded up
/// to a power of two, and far below the shallowest measured stack-overflow
/// floor (5500, the FlatPPL source parser).
pub const DEFAULT_MAX_DEPTH: usize = 128;

/// A recursion budget carried BY VALUE through a recursive descent.
///
/// Each step calls [`Depth::deeper`] and passes the result to its children, so
/// nesting is charged and breadth is not: siblings each receive a copy at the
/// same level. Passing by value rather than holding a guard is deliberate — an
/// RAII guard borrowing the budget cannot express nesting at all, since a
/// parent's guard is still alive while its child takes one.
///
/// ```
/// use flatppl_core::depth::Depth;
/// let root = Depth::with_limit(2);
/// let one = root.deeper("expression").unwrap();
/// let two = one.deeper("expression").unwrap();
/// assert!(two.deeper("expression").is_err());   // depth 3 exceeds the limit
/// assert!(one.deeper("expression").is_ok());    // a sibling is unaffected
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Depth {
    current: usize,
    limit: usize,
}

/// The refusal a walk returns when it is already as deep as it may go.
///
/// Carries the limit so a message can state the number rather than describe it,
/// and the construct so the reader knows which descent ran out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooDeep {
    /// What was being walked, for the message (`"expression"`, `"node"`, …).
    pub construct: String,
    /// The limit that was reached.
    pub limit: usize,
}

impl std::fmt::Display for TooDeep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} nesting is deeper than the limit of {}; this is a resource guard, \
             not a language rule — the deepest model in any FlatPPL corpus is 14 levels",
            self.construct, self.limit
        )
    }
}

impl std::error::Error for TooDeep {}

impl Default for Depth {
    fn default() -> Self {
        Self::with_limit(DEFAULT_MAX_DEPTH)
    }
}

impl Depth {
    /// A budget with a caller-chosen limit. Prefer [`Depth::default`].
    pub fn with_limit(limit: usize) -> Self {
        Depth { current: 0, limit }
    }

    /// The budget one level down, or a refusal naming `construct`.
    pub fn deeper(self, construct: &str) -> Result<Depth, TooDeep> {
        if self.current >= self.limit {
            return Err(TooDeep {
                construct: construct.to_string(),
                limit: self.limit,
            });
        }
        Ok(Depth {
            current: self.current + 1,
            limit: self.limit,
        })
    }

    /// How deep this budget already is.
    pub fn current(self) -> usize {
        self.current
    }

    /// The configured limit.
    pub fn limit(self) -> usize {
        self.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_limit_is_eight_times_the_deepest_corpus_model_as_a_power_of_two() {
        // The deepest model measured across the examples, testsuite, HS3 and
        // pyhf corpora is 14 levels. Pinning the derivation makes a future
        // change to the constant restate the reasoning.
        let deepest_real_model = 14usize;
        let target = 8 * deepest_real_model;
        assert_eq!(
            DEFAULT_MAX_DEPTH,
            target.next_power_of_two(),
            "8 x 14 rounded up"
        );
        const {
            assert!(
                DEFAULT_MAX_DEPTH < 5500,
                "must stay below the shallowest measured crash floor"
            )
        };
    }

    #[test]
    fn nesting_is_charged_but_breadth_is_not() {
        let root = Depth::with_limit(2);
        // 1000 siblings all at depth 1: free.
        for _ in 0..1000 {
            assert!(root.deeper("node").is_ok());
        }
        let a = root.deeper("node").unwrap();
        let b = a.deeper("node").unwrap();
        assert!(b.deeper("node").is_err(), "depth 3 exceeds the limit of 2");
        assert!(a.deeper("node").is_ok(), "a sibling of b is unaffected");
    }

    #[test]
    fn refusal_names_the_construct_and_states_the_limit() {
        let err = Depth::with_limit(0)
            .deeper("expression")
            .expect_err("limit 0 refuses immediately");
        assert_eq!((err.limit, err.construct.as_str()), (0, "expression"));
        let msg = err.to_string();
        assert!(
            msg.contains("expression") && msg.contains("limit of 0"),
            "message must state both: {msg}"
        );
    }
}
