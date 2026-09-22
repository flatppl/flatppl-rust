//! Operation rules can suspend while a callable body is inferred in its own scope.
//!
//! Continuations capture only rule-local state. The trace driver owns nested
//! inference scopes and restores the caller before invoking its continuation.

use flatppl_core::{NodeId, Type, ValueSet};

use crate::modules::Resolved;
use crate::trace::Inferencer;

pub(crate) type Resume<T> =
    Box<dyn for<'m, 's> FnOnce(&mut Inferencer<'m, 's>, (Type, ValueSet)) -> RuleStep<T>>;

pub(crate) enum RuleStep<T> {
    Ready(T),
    InferBody {
        body: NodeId,
        seeds: Vec<(NodeId, Resolved)>,
        resume: Resume<T>,
    },
}

impl<T: 'static> RuleStep<T> {
    /// Compose a bounded chain of rule helpers without entering a body walk.
    pub(crate) fn map<U: 'static>(
        self,
        inf: &mut Inferencer<'_, '_>,
        f: impl for<'m, 's> FnOnce(&mut Inferencer<'m, 's>, T) -> U + 'static,
    ) -> RuleStep<U> {
        match self {
            Self::Ready(value) => RuleStep::Ready(f(inf, value)),
            Self::InferBody {
                body,
                seeds,
                resume,
            } => RuleStep::InferBody {
                body,
                seeds,
                resume: Box::new(move |inf, result| resume(inf, result).map(inf, f)),
            },
        }
    }

    pub(crate) fn and_then<U: 'static>(
        self,
        inf: &mut Inferencer<'_, '_>,
        f: impl for<'m, 's> FnOnce(&mut Inferencer<'m, 's>, T) -> RuleStep<U> + 'static,
    ) -> RuleStep<U> {
        match self {
            Self::Ready(value) => f(inf, value),
            Self::InferBody {
                body,
                seeds,
                resume,
            } => RuleStep::InferBody {
                body,
                seeds,
                resume: Box::new(move |inf, result| resume(inf, result).and_then(inf, f)),
            },
        }
    }
}
