//! Per-profile boundary facts and the cost terms they add to extraction.

use flatppl_core::{BindingId, Module, Phase};

use crate::Profile;

// More than any whole query can save, so a boundary keeps its written form.
const REWRITE_ACROSS_BOUNDARY: u64 = 1 << 40;

/// What extraction knows about one e-class and one of its members.
#[derive(Clone, Copy)]
pub(super) struct Choice<'a> {
    /// The operation of the member, if it is a call.
    pub(super) head: Option<&'a str>,
    /// The class holds a value the target reads by name.
    pub(super) data_boundary: bool,
    /// The member is a term the user wrote.
    pub(super) written: bool,
}

impl Profile {
    /// Whether the target reads a named fixed value by its name, as data.
    fn reads_named_data(self) -> bool {
        matches!(self, Profile::Stan)
    }

    /// The profile's addition to the shared extraction cost of a member.
    pub(super) fn extra_cost(self, choice: Choice) -> u64 {
        let boundary = if self.reads_named_data() && choice.data_boundary && !choice.written {
            REWRITE_ACROSS_BOUNDARY
        } else {
            0
        };
        // Stan has no axis insertion. It writes `to_matrix` of a transposed
        // vector, and a literal vector adds one more transpose.
        let operations = match (self, choice.head) {
            (Profile::Stan, Some("addaxes")) => 2,
            _ => 0,
        };
        boundary + operations
    }
}

/// A named value the target reads by its name. Neither its value nor its
/// reads may be replaced.
pub(super) fn data_boundary(m: &Module, profile: Profile, binding: BindingId) -> bool {
    profile.reads_named_data()
        && m.query_pinned_rhs(binding).is_none()
        && m.phase_of(m.binding(binding).rhs) == Some(Phase::Fixed)
}
