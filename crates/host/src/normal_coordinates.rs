//! Compiled companions for one explicitly selected Normal coordinate.

use crate::Export;

/// All queries use the requested emission precision and target. Every output is
/// `(value, valid)` plus any separate native runtime status. Check both before
/// using a value. Invalid coordinates are not support-zero log densities.
pub struct NormalCoordinateExports {
    pub score: Export,
    pub forward: Export,
    pub inverse: Export,
    pub log_jacobian: Export,
    /// Replaces only the selected input slot in score, forward and Jacobian.
    /// The inverse retains the original input name.
    pub coordinate_input: String,
}
