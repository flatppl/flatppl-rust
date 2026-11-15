//! Explicit dependency invalidation for host-owned scalar term caches.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::{AnalysisError, LoadedModule};

/// One ordinary query and the input names that invalidate its cached result.
pub struct IncrementalTermQuery {
    pub query: Arc<LoadedModule>,
    pub dependencies: Vec<String>,
    pub cache_input: String,
}

/// Immutable term queries and an ordinary combiner. Keep cached values in the
/// executor and recombine them through `combine`; never subtract old totals.
pub struct IncrementalScoreQueries {
    terms: Vec<IncrementalTermQuery>,
    combine: Arc<LoadedModule>,
    dependencies: Arc<DependencyIndex>,
}

struct DependencyIndex {
    inputs: Vec<String>,
    affected: HashMap<String, Vec<usize>>,
}

impl DependencyIndex {
    fn affected_terms(&self, changed: &[String]) -> Result<Vec<usize>, AnalysisError> {
        let mut affected = BTreeSet::new();
        for name in changed {
            let terms = self
                .affected
                .get(name)
                .ok_or_else(|| AnalysisError(format!("unknown incremental input {name:?}")))?;
            affected.extend(terms);
        }
        Ok(affected.into_iter().copied().collect())
    }
}

impl IncrementalScoreQueries {
    pub(crate) fn new(
        terms: Vec<IncrementalTermQuery>,
        combine: Arc<LoadedModule>,
        inputs: Vec<String>,
    ) -> Self {
        let mut affected = inputs
            .iter()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<HashMap<_, _>>();
        for (index, term) in terms.iter().enumerate() {
            for name in &term.dependencies {
                affected
                    .get_mut(name)
                    .expect("term dependencies come from the analyzed boundary")
                    .push(index);
            }
        }
        Self {
            terms,
            combine,
            dependencies: Arc::new(DependencyIndex { inputs, affected }),
        }
    }

    pub fn terms(&self) -> &[IncrementalTermQuery] {
        &self.terms
    }
    pub fn combine(&self) -> &Arc<LoadedModule> {
        &self.combine
    }
    pub fn inputs(&self) -> &[String] {
        &self.dependencies.inputs
    }

    pub fn affected_terms(&self, changed: &[String]) -> Result<Vec<usize>, AnalysisError> {
        self.dependencies.affected_terms(changed)
    }

    /// Create one coordinator per compiled plan, precision and target. Notify it
    /// before changing any input, including in-place array edits and rollbacks.
    /// New models, imports, shapes or compiled artifacts require a new coordinator.
    pub fn revisions(&self) -> ScoreRevisions {
        ScoreRevisions {
            identity: Arc::new(()),
            dependencies: self.dependencies.clone(),
            generations: vec![0; self.terms.len()],
            current: vec![false; self.terms.len()],
        }
    }
}

/// A result's term revision, captured before asynchronous evaluation begins.
/// Check native runtime status before offering its value to `accept`.
pub struct ScoreTicket {
    identity: Arc<()>,
    term: usize,
    generation: u64,
}

/// Revision state only. No data values, array identities or evaluator caches.
pub struct ScoreRevisions {
    identity: Arc<()>,
    dependencies: Arc<DependencyIndex>,
    generations: Vec<u64>,
    current: Vec<bool>,
}

impl ScoreRevisions {
    /// Invalidate all affected terms atomically. Repeated names invalidate once.
    /// Unknown input names and revision overflow leave every revision unchanged.
    pub fn invalidate(&mut self, changed: &[String]) -> Result<Vec<usize>, AnalysisError> {
        let affected = self.dependencies.affected_terms(changed)?;
        if affected.iter().any(|&i| self.generations[i] == u64::MAX) {
            return Err(AnalysisError(
                "incremental revision overflow requires a new coordinator".into(),
            ));
        }
        for &index in &affected {
            self.generations[index] += 1;
            self.current[index] = false;
        }
        Ok(affected)
    }

    pub fn ticket(&self, term: usize) -> Result<ScoreTicket, AnalysisError> {
        let generation = *self
            .generations
            .get(term)
            .ok_or_else(|| AnalysisError("incremental term index is out of range".into()))?;
        Ok(ScoreTicket {
            identity: self.identity.clone(),
            term,
            generation,
        })
    }

    /// Accept only a result from this coordinator's current input revisions.
    /// A status-zero infinite log density is valid. Failed statuses never commit.
    /// Commit the executor's term value only when this method returns true.
    pub fn accept(&mut self, ticket: ScoreTicket) -> bool {
        if !Arc::ptr_eq(&ticket.identity, &self.identity)
            || self.generations.get(ticket.term) != Some(&ticket.generation)
        {
            return false;
        }
        self.current[ticket.term] = true;
        true
    }

    pub fn is_current(&self, term: usize) -> Result<bool, AnalysisError> {
        self.current
            .get(term)
            .copied()
            .ok_or_else(|| AnalysisError("incremental term index is out of range".into()))
    }
}
