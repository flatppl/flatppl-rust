//! Compiled companions for one explicitly selected Normal coordinate.

use std::sync::Arc;

use crate::{Export, LoadedModule};

/// Query modules whose validity constants use one explicit arithmetic precision.
/// Compile every companion with that precision. Each returns `(value, valid)`.
pub struct PreparedNormalCoordinate {
    pub score: Arc<LoadedModule>,
    pub forward: Arc<LoadedModule>,
    pub inverse: Arc<LoadedModule>,
    pub log_jacobian: Arc<LoadedModule>,
    pub coordinate_input: String,
}

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
