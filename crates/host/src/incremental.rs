//! Explicit dependency invalidation for host-owned scalar term caches.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use crate::{AnalysisError, LoadedModule};
use flatppl_determinizer::{InputArray, InputCell, InputPath};

/// One ordinary query and the input names that invalidate its cached result.
pub struct IncrementalTermQuery {
    pub query: Arc<LoadedModule>,
    pub dependencies: Vec<String>,
    pub input_paths: Vec<InputPath>,
    pub input_cells: Vec<InputCell>,
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
    valid: HashSet<InputPath>,
    own: HashMap<InputPath, Vec<usize>>,
    descendants: HashMap<InputPath, Vec<usize>>,
    cells: HashMap<InputCell, Vec<usize>>,
    arrays: HashMap<InputPath, Vec<u64>>,
}

impl DependencyIndex {
    fn affected_cells(&self, changed: &[InputCell]) -> Result<Vec<usize>, AnalysisError> {
        for cell in changed {
            let valid = self.arrays.get(&cell.path).is_some_and(|shape| {
                shape.len() == cell.indices.len()
                    && cell
                        .indices
                        .iter()
                        .zip(shape)
                        .all(|(index, size)| index < size)
            });
            if !valid {
                return Err(AnalysisError(format!(
                    "invalid incremental input cell {cell:?}"
                )));
            }
        }
        let mut affected = BTreeSet::new();
        for cell in changed {
            affected.extend(self.cells.get(cell).into_iter().flatten());
            for count in 0..=cell.path.fields.len() {
                let parent = InputPath {
                    input: cell.path.input.clone(),
                    fields: cell.path.fields[..count].to_vec(),
                };
                affected.extend(self.own.get(&parent).into_iter().flatten());
            }
        }
        Ok(affected.into_iter().copied().collect())
    }

    fn affected_terms(&self, changed: &[String]) -> Result<Vec<usize>, AnalysisError> {
        self.affected_paths(
            &changed
                .iter()
                .map(|input| InputPath {
                    input: input.clone(),
                    fields: vec![],
                })
                .collect::<Vec<_>>(),
        )
    }

    fn affected_paths(&self, changed: &[InputPath]) -> Result<Vec<usize>, AnalysisError> {
        for path in changed {
            if !self.valid.contains(path) {
                return Err(AnalysisError(format!(
                    "unknown incremental input path {path:?}"
                )));
            }
        }
        let mut affected = BTreeSet::new();
        for path in changed {
            affected.extend(self.descendants.get(path).into_iter().flatten());
            for count in 0..path.fields.len() {
                let parent = InputPath {
                    input: path.input.clone(),
                    fields: path.fields[..count].to_vec(),
                };
                affected.extend(self.own.get(&parent).into_iter().flatten());
            }
        }
        Ok(affected.into_iter().copied().collect())
    }
}

impl IncrementalScoreQueries {
    pub(crate) fn new(
        terms: Vec<IncrementalTermQuery>,
        combine: Arc<LoadedModule>,
        inputs: Vec<String>,
        input_paths: Vec<InputPath>,
        input_arrays: Vec<InputArray>,
    ) -> Self {
        let valid = input_paths.into_iter().collect();
        let mut own: HashMap<InputPath, Vec<usize>> = HashMap::new();
        let mut descendants: HashMap<InputPath, Vec<usize>> = HashMap::new();
        let mut cells: HashMap<InputCell, Vec<usize>> = HashMap::new();
        for (index, term) in terms.iter().enumerate() {
            for path in &term.input_paths {
                own.entry(path.clone()).or_default().push(index);
            }
            for cell in &term.input_cells {
                cells.entry(cell.clone()).or_default().push(index);
            }
            for path in term
                .input_paths
                .iter()
                .chain(term.input_cells.iter().map(|cell| &cell.path))
            {
                for count in 0..=path.fields.len() {
                    descendants
                        .entry(InputPath {
                            input: path.input.clone(),
                            fields: path.fields[..count].to_vec(),
                        })
                        .or_default()
                        .push(index);
                }
            }
        }
        Self {
            terms,
            combine,
            dependencies: Arc::new(DependencyIndex {
                inputs,
                valid,
                own,
                descendants,
                cells,
                arrays: input_arrays
                    .into_iter()
                    .map(|array| (array.path, array.shape))
                    .collect(),
            }),
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

    pub fn affected_paths(&self, changed: &[InputPath]) -> Result<Vec<usize>, AnalysisError> {
        self.dependencies.affected_paths(changed)
    }

    /// Scalar array edits use full-rank zero-based coordinates. Whole-array and
    /// ancestor record readers remain affected. Unread valid cells are allowed.
    pub fn affected_cells(&self, changed: &[InputCell]) -> Result<Vec<usize>, AnalysisError> {
        self.dependencies.affected_cells(changed)
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
        self.invalidate_terms(affected)
    }

    /// Field edits invalidate overlapping prefixes and whole-record readers.
    /// Validate every path before changing any revision.
    pub fn invalidate_paths(&mut self, changed: &[InputPath]) -> Result<Vec<usize>, AnalysisError> {
        let affected = self.dependencies.affected_paths(changed)?;
        self.invalidate_terms(affected)
    }

    /// Validate every cell before invalidating any result revision.
    pub fn invalidate_cells(&mut self, changed: &[InputCell]) -> Result<Vec<usize>, AnalysisError> {
        let affected = self.dependencies.affected_cells(changed)?;
        self.invalidate_terms(affected)
    }

    fn invalidate_terms(&mut self, affected: Vec<usize>) -> Result<Vec<usize>, AnalysisError> {
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
