//! Equality saturation over the projected query, followed by shared-DAG extraction.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use egglog::sort::S;
use egglog::{Core, EGraph, Error, FullState, RawValues, Read, Value, Write};

use crate::{Dtype, Profile, Target};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) enum Term {
    Leaf(usize),
    Boolean(bool),
    Integer(i64),
    Real(i64),
    Call(String, Vec<usize>),
}

pub(super) struct Fact {
    pub name: &'static str,
    pub args: Vec<FactArg>,
}

pub(super) enum FactArg {
    Expr(usize),
    Int(i64),
    Bool(bool),
    Str(&'static str),
    Call(&'static str, Vec<FactArg>),
}

const CONSTRUCTORS: [&str; 7] = ["Leaf", "Boolean", "Integer", "Real", "U", "B", "T"];
const MAX_INPUT_ROWS: usize = 32_768;
const MAX_NEW_ROWS: usize = 16_384;
const MAX_MATCHES: usize = 131_072;
const MAX_ROUNDS: usize = 24;
const MAX_TRIAL_VISITS: usize = 1_000_000;

#[derive(Clone, Copy)]
struct Catalogue {
    profile: Profile,
    measures: bool,
    predicates_and_reductions: bool,
    fixed_poisson: bool,
}

impl Catalogue {
    fn for_query(profile: Profile, terms: &[Term], facts: &[Fact]) -> Self {
        // Measure rules require original constructor fields or scoped scoring
        // witnesses. The predicate/reduction rules introduce no new heads
        // reachable from the remaining catalogues.
        let measures = facts.iter().any(|fact| {
            fact.name.starts_with("gaussian-")
                || fact.name.starts_with("conjugate-")
                || fact.name.starts_with("measure-")
        });
        let predicates_and_reductions = matches!(profile, Profile::StableHlo { .. })
            && terms.iter().any(|term| {
                matches!(term, Term::Call(name, _) if matches!(name.as_str(),
                    "land" | "lor" | "lxor" | "lnot" | "equal" | "unequal" | "ifelse"
                    | "logsumexp" | "logsumexp.pair" | "special-functions.log_ndtr"))
            });
        Self {
            profile,
            measures,
            predicates_and_reductions,
            fixed_poisson: facts
                .iter()
                .any(|fact| fact.name == "fixed-poisson-partition"),
        }
    }

    fn index(self) -> usize {
        let first = match self.profile {
            Profile::Portable => 0,
            Profile::Stan => 2,
            Profile::StableHlo { .. } => 4,
        };
        first
            + usize::from(self.measures)
            + 2 * usize::from(self.predicates_and_reductions)
            + 8 * usize::from(self.fixed_poisson)
    }
}

thread_local! {
    // Egglog snapshots share their intern pools, so retire the template after
    // a bounded amount of input and derived data have passed through it.
    static CATALOGUES: RefCell<[Option<(EGraph, usize)>; 16]> = const {
        RefCell::new([const { None }; 16])
    };
}

fn catalogue(config: Catalogue, rows: usize) -> EGraph {
    CATALOGUES.with_borrow_mut(|catalogues| {
        let cached = &mut catalogues[config.index()];
        if cached
            .as_ref()
            .is_none_or(|(_, used)| used + rows > 4 * MAX_INPUT_ROWS)
        {
            *cached = Some((build_catalogue(config), 0));
        }
        let (template, used) = cached.as_mut().expect("initialized rule catalogue");
        *used += rows;
        template.clone()
    })
}

fn build_catalogue(config: Catalogue) -> EGraph {
    let mut graph = EGraph::default().with_num_threads(1);
    let mut load = |name: &str, program: &str| {
        graph
            .parse_and_run_program(Some(format!("{name}.egg")), program)
            .unwrap_or_else(|error| panic!("invalid embedded egglog catalogue {name}: {error}"));
    };
    load("schema", include_str!("rules/schema.egg"));
    load("portable", include_str!("rules/portable.egg"));
    load("concat", include_str!("rules/concat.egg"));
    let mut portable =
        String::from("(unstable-combined-ruleset query-portable analysis portable concat");
    if config.measures {
        load("gaussian", include_str!("rules/gaussian.egg"));
        load("conjugacy", include_str!("rules/conjugacy.egg"));
        load("measure_algebra", include_str!("rules/measure_algebra.egg"));
        portable.push_str(" gaussian conjugacy measure-algebra");
    }
    portable.push(')');
    load("portable-schedule", &portable);
    match config.profile {
        Profile::Portable => {}
        Profile::Stan => {
            load("stan", include_str!("rules/stan.egg"));
            load(
                "stan-schedule",
                "(unstable-combined-ruleset query-stan query-portable stan)",
            );
        }
        Profile::StableHlo { .. } => {
            load("mathematical", include_str!("rules/mathematical.egg"));
            load("sharing", include_str!("rules/sharing.egg"));
            load("hoisting", include_str!("rules/hoisting.egg"));
            let mut schedule = String::from(
                "(unstable-combined-ruleset query-gpu query-portable mathematical selector-facts sharing",
            );
            if config.predicates_and_reductions {
                load("statistical", include_str!("rules/statistical.egg"));
                load("simplification", include_str!("rules/simplification.egg"));
                schedule.push_str(" statistical simplification");
            }
            schedule.push_str(")\n(unstable-combined-ruleset query-cpu query-gpu hoisting");
            if config.fixed_poisson {
                load("fixed_poisson", include_str!("rules/fixed_poisson.egg"));
                schedule.push_str(" fixed-poisson");
            }
            schedule.push(')');
            load("stablehlo-schedule", &schedule);
        }
    }
    graph
}

pub(super) fn optimise(
    terms: &mut Vec<Term>,
    roots: &[usize],
    facts: &[Fact],
    profile: Profile,
) -> Option<Vec<usize>> {
    if roots.is_empty()
        || terms.len() + facts.len() > MAX_INPUT_ROWS
        || !has_candidates(terms, facts, profile)
    {
        return None;
    }
    optimise_query(terms, roots, facts, profile).ok().flatten()
}

/// Run an emitter catalogue over bounded integer facts supplied by the
/// lowering. The caller owns the interpretation of the result relation.
pub(super) fn integer_plan_rows(
    filename: &str,
    program: &str,
    ruleset: &str,
    facts: &[(&str, &[i64])],
    result: &str,
) -> Option<Vec<Vec<i64>>> {
    if facts.len() > MAX_INPUT_ROWS {
        return None;
    }
    let mut graph = EGraph::default().with_num_threads(1);
    graph
        .parse_and_run_program(Some(filename.to_owned()), program)
        .unwrap_or_else(|error| panic!("invalid embedded emitter catalogue {filename}: {error}"));
    graph
        .update(|mut state| {
            for &(relation, values) in facts {
                let key = values
                    .iter()
                    .map(|&value| state.base_to_value(value))
                    .collect();
                state.add(relation, RawValues(key))?;
            }
            Ok(())
        })
        .ok()?;
    let mut matches = 0;
    if !saturate(&mut graph, ruleset, &mut matches).ok()? {
        return None;
    }
    graph
        .read(|state| {
            let mut rows = vec![];
            state.constructor_enodes(result, |row| {
                rows.push(
                    row.children
                        .iter()
                        .map(|&value| state.value_to_base::<i64>(value))
                        .collect(),
                );
            })?;
            Ok::<_, Error>(rows)
        })
        .ok()
}

fn saturate(graph: &mut EGraph, rules: &str, matches: &mut usize) -> Result<bool, Error> {
    let initial_rows = graph.num_tuples();
    if initial_rows > MAX_INPUT_ROWS {
        return Ok(false);
    }
    for _ in 0..MAX_ROUNDS {
        let report = graph.step_rules(rules)?;
        for &count in report.num_matches_per_rule.values() {
            *matches = matches.saturating_add(count);
        }
        if graph.num_tuples() > initial_rows + MAX_NEW_ROWS || *matches > MAX_MATCHES {
            return Ok(false);
        }
        if report.can_stop {
            return Ok(true);
        }
    }
    Ok(false)
}

fn optimise_query(
    terms: &mut Vec<Term>,
    roots: &[usize],
    facts: &[Fact],
    profile: Profile,
) -> Result<Option<Vec<usize>>, Error> {
    let config = Catalogue::for_query(profile, terms, facts);
    let mut graph = catalogue(config, terms.len() + facts.len());
    graph.update(|mut state| {
        let mut values = Vec::with_capacity(terms.len());
        for term in terms.iter() {
            let (name, args) = encode_term(&state, term, &values);
            values.push(state.add(name, RawValues(args))?);
        }
        for fact in facts {
            let args = encode_args(&mut state, &fact.args, &values)?;
            state.add(fact.name, RawValues(args))?;
        }
        Ok(())
    })?;

    // The emitter's GPU fusion can make a full negation and gather cheaper
    // than an independently materialized narrow map. Keep that schedule on GPU.
    let rules = match profile {
        Profile::Portable => "query-portable",
        Profile::Stan => "query-stan",
        Profile::StableHlo {
            target: Target::Cpu,
            ..
        } => "query-cpu",
        Profile::StableHlo {
            target: Target::Gpu,
            ..
        } => "query-gpu",
    };
    let mut matches = 0usize;
    let saturated = saturate(&mut graph, rules, &mut matches);
    CATALOGUES.with_borrow_mut(|catalogues| {
        if let Some((_, used)) = &mut catalogues[config.index()] {
            *used = used.saturating_add(matches.min(MAX_MATCHES));
        }
    });
    if !saturated? {
        return Ok(None);
    }
    let choices = graph.read(|state| ExecutionDag::read(&state, terms, config))?;
    Ok(choices.and_then(|dag| dag.improve(terms, roots, MAX_TRIAL_VISITS)))
}

fn has_candidates(terms: &[Term], facts: &[Fact], profile: Profile) -> bool {
    use FactArg::{Call, Expr, Int};

    let hlo = matches!(profile, Profile::StableHlo { .. });
    let cpu = matches!(
        profile,
        Profile::StableHlo {
            target: Target::Cpu,
            ..
        }
    );
    let stan = matches!(profile, Profile::Stan);
    let mut singleton_vectors = HashSet::new();
    let mut singleton_matrices = HashSet::new();
    let mut empty_vectors = HashSet::new();
    let mut unit_counts = HashSet::new();
    let mut discardable = HashSet::new();
    for fact in facts {
        match (fact.name, fact.args.as_slice()) {
            (name, _) if name.starts_with("conjugate-") => return true,
            (_, _) if crate::gaussian::has_candidate(fact, terms) => return true,
            ("portable-exclusive-transposed-cell" | "concat-read" | "fixed-singleton-stack", _) => {
                return true;
            }
            ("measure-alias", _) => return true,
            ("gaussian-fixed-cdf", [_, Int(mu), _, Int(x)])
                if x == mu || f64::from_bits(*x as u64).is_infinite() =>
            {
                return true;
            }
            ("exclusive-gather-source" | "composed-selector", _) if hlo => return true,
            ("commutable-negation", _) if cpu => return true,
            ("fixed-poisson-partition", _) if cpu => return true,
            ("real-vector", [Expr(id), Int(0)]) => {
                singleton_vectors.insert(*id);
                empty_vectors.insert(*id);
            }
            ("real-vector", [Expr(id), Int(1)]) => {
                singleton_vectors.insert(*id);
            }
            ("real-matrix", [Expr(id), Int(1), Int(1)]) => {
                singleton_matrices.insert(*id);
            }
            ("math-block-shape", [Expr(id), Call("MathDim", dims), _])
                if matches!(dims.as_slice(), [Int(1), Call("MathNil", _)]) =>
            {
                singleton_vectors.insert(*id);
            }
            ("math-int-list", [Expr(id), dims]) if unit_dimensions(dims) => {
                unit_counts.insert(*id);
            }
            ("portable-discardable", [Expr(id)]) => {
                discardable.insert(*id);
            }
            _ => {}
        }
    }
    let op = |id: usize| match &terms[id] {
        Term::Call(name, _) => name.as_str(),
        _ => "",
    };
    let endpoint = |id: usize| match &terms[id] {
        Term::Real(bits) => f64::from_bits(*bits as u64).is_infinite(),
        Term::Call(name, _) => matches!(name.as_str(), "neg" | "ifelse"),
        _ => false,
    };
    let transpose = |name: &str| matches!(name, "transpose" | "adjoint");
    let is_transposed = |id: usize, source: usize| matches!(&terms[id], Term::Call(name, args) if transpose(name) && args == &[source]);
    let boolean = |id: usize| matches!(terms[id], Term::Boolean(_));
    let integer = |id: usize| matches!(terms[id], Term::Integer(_));
    terms.iter().any(|term| {
        let Term::Call(name, args) = term else {
            return false;
        };
        match (name.as_str(), args.as_slice()) {
            ("identity", [_]) => true,
            ("lengthof", [x]) => op(*x) == "vector" || discardable.contains(x),
            ("logdensityof" | "densityof" | "totalmass", [m, ..]) => {
                op(*m) == "normalize" || matches!(&terms[*m], Term::Call(head, args)
                    if head == "truncate" && args.first().is_some_and(|&base| op(base) == "truncate"))
            }
            ("reverse", [x]) => op(*x) == "reverse" || singleton_vectors.contains(x),
            ("lnot", [x]) => op(*x) == "lnot" || boolean(*x),
            ("transpose" | "adjoint", [x]) => {
                transpose(op(*x))
                    || (hlo && (matches!(op(*x), "rowstack" | "colstack" | "addaxes" | "diagmat")
                        || singleton_matrices.contains(x)))
            }
            ("get" | "get0", [x, k]) => {
                integer(*k) && matches!(op(*x), "reverse" | "vector")
            }
            ("land" | "lor" | "lxor" | "equal" | "unequal", [x, y]) => {
                boolean(*x) || boolean(*y)
                    || (hlo && (*x == *y || related_predicates(terms, *x, *y)))
                    || (matches!(name.as_str(), "land" | "lor") && (
                        matches!(&terms[*x], Term::Call(op, args) if op == "lnot" && args == &[*y])
                        || matches!(&terms[*y], Term::Call(op, args) if op == "lnot" && args == &[*x])
                        || (comparison(op(*x)) && comparison(op(*y)) && related_predicates(terms, *x, *y))
                    ))
            }
            ("ifelse", [c, a, b]) => a == b || boolean(*c) || (boolean(*a) && boolean(*b)),
            ("mul", [x, y]) if hlo || stan => is_transposed(*x, *y) || is_transposed(*y, *x),
            _ if !hlo => false,
            ("tile", [_, counts]) => matches!(terms[*counts], Term::Integer(1)) || unit_counts.contains(counts),
            ("addaxes", [x, left, right]) => {
                op(*x) == "addaxes" || (matches!(terms[*left], Term::Integer(0)) && matches!(terms[*right], Term::Integer(0)))
            }
            ("joinblocks", [x]) => op(*x) == "splitblocks",
            ("splitblocks", [x, _]) => op(*x) == "joinblocks",
            ("row_gram" | "col_gram", [x]) => transpose(op(*x)),
            ("sum", [x]) => op(*x) == "diag",
            ("special-functions.log_ndtr", [x]) => endpoint(*x),
            ("logsumexp" | "logsumexp.pair", [x]) => match &terms[*x] {
                Term::Call(head, items) if head == "vector" => items.iter().any(|&id| endpoint(id)),
                _ => empty_vectors.contains(x),
            },
            _ => false,
        }
    })
}

fn unit_dimensions(mut dims: &FactArg) -> bool {
    loop {
        match dims {
            FactArg::Call("MathNil", _) => return true,
            FactArg::Call("MathDim", args) => match args.as_slice() {
                [FactArg::Int(1), tail] => dims = tail,
                _ => return false,
            },
            _ => return false,
        }
    }
}

fn ordered(op: &str) -> bool {
    matches!(op, "lt" | "le" | "gt" | "ge")
}

fn comparison(op: &str) -> bool {
    ordered(op) || matches!(op, "equal" | "unequal")
}

fn related_predicates(terms: &[Term], x: usize, y: usize) -> bool {
    let (Term::Call(left, a), Term::Call(right, b)) = (&terms[x], &terms[y]) else {
        return [&terms[x], &terms[y]].iter().any(|term| {
            matches!(term, Term::Call(op, _) if matches!(op.as_str(), "lnot" | "land" | "lor"))
        });
    };
    let compound = |op: &str| matches!(op, "lnot" | "land" | "lor");
    if compound(left) || compound(right) {
        return true;
    }
    let finiteness = |op: &str| matches!(op, "isfinite" | "isinf" | "isnan");
    let bound = |x: usize| matches!(terms[x], Term::Integer(_));
    match (a.as_slice(), b.as_slice()) {
        ([p], [q]) => p == q && finiteness(left) && finiteness(right),
        ([a, b], [c, d]) if comparison(left) && comparison(right) => {
            (a == c && b == d)
                || (a == d && b == c)
                || ((ordered(left) || ordered(right))
                    && ((a == c && bound(*b) && bound(*d))
                        || (a == d && bound(*b) && bound(*c))
                        || (b == c && bound(*a) && bound(*d))
                        || (b == d && bound(*a) && bound(*c))))
        }
        _ => false,
    }
}

fn encode_term<'a, 'db: 'a>(
    state: &impl Core<'a, 'db>,
    term: &Term,
    values: &[Value],
) -> (&'static str, Vec<Value>) {
    match term {
        Term::Leaf(id) => ("Leaf", vec![state.base_to_value(*id as i64)]),
        Term::Boolean(value) => ("Boolean", vec![state.base_to_value(*value)]),
        Term::Integer(value) => ("Integer", vec![state.base_to_value(*value)]),
        Term::Real(bits) => ("Real", vec![state.base_to_value(*bits)]),
        Term::Call(name, args) => {
            let mut key = Vec::with_capacity(args.len() + 1);
            key.push(state.base_to_value(S::new(name.clone())));
            key.extend(args.iter().map(|&arg| values[arg]));
            (CONSTRUCTORS[3 + args.len()], key)
        }
    }
}

fn encode_args(
    state: &mut FullState<'_, '_>,
    args: &[FactArg],
    values: &[Value],
) -> Result<Vec<Value>, Error> {
    args.iter()
        .map(|arg| match arg {
            FactArg::Expr(id) => Ok(values[*id]),
            FactArg::Int(value) => Ok(state.base_to_value(*value)),
            FactArg::Bool(value) => Ok(state.base_to_value(*value)),
            FactArg::Str(value) => Ok(state.base_to_value(S::new((*value).into()))),
            FactArg::Call(name, args) => {
                let args = encode_args(state, args, values)?;
                state.add(name, RawValues(args))
            }
        })
        .collect()
}

fn decode_term<'a, 'db: 'a>(
    state: &impl Read<'a, 'db>,
    constructor: usize,
    args: &[Value],
    classes: &HashMap<Value, usize>,
) -> Option<Term> {
    let term = match constructor {
        0 => Term::Leaf(state.value_to_base::<i64>(args[0]) as usize),
        1 => Term::Boolean(state.value_to_base(args[0])),
        2 => Term::Integer(state.value_to_base(args[0])),
        3 => Term::Real(state.value_to_base(args[0])),
        _ => Term::Call(
            state.value_to_base::<S>(args[0]).into_inner(),
            args[1..]
                .iter()
                .map(|x| classes.get(x).copied())
                .collect::<Option<_>>()?,
        ),
    };
    Some(term)
}

/// Each e-class owns its alternatives. For projected originals, `original`
/// records the exact alternative for each term, including shared structure.
struct ExecutionDag {
    choices: Vec<Vec<Term>>,
    costs: Vec<Vec<u64>>,
    original: Vec<(usize, usize)>,
}

impl ExecutionDag {
    fn read<'a, 'db: 'a>(
        state: &impl Read<'a, 'db>,
        terms: &[Term],
        config: Catalogue,
    ) -> Result<Option<Self>, Error> {
        let mut original_values = Vec::with_capacity(terms.len());
        for term in terms {
            let (name, key) = encode_term(state, term, &original_values);
            let Some(value) = state.eclass_of(name, RawValues(key))? else {
                return Ok(None);
            };
            original_values.push(value);
        }
        let mut rows = Vec::new();
        for (constructor, name) in CONSTRUCTORS.iter().enumerate() {
            state.constructor_enodes(name, |row| {
                if !row.subsumed {
                    rows.push((row.eclass, constructor, row.children.to_vec()));
                }
            })?;
        }
        // Keep original classes in projection order. Give newly created
        // classes IDs in structural order, independent of egglog's interned
        // values and of which other models previously used the template.
        let mut classes = HashMap::new();
        for &value in &original_values {
            let next = classes.len();
            classes.entry(value).or_insert(next);
        }
        let mut pending: Vec<_> = rows
            .iter()
            .filter(|(id, _, _)| !classes.contains_key(id))
            .collect();
        while !pending.is_empty() {
            let mut ready = Vec::new();
            for &&(id, constructor, ref args) in &pending {
                if let Some(term) = decode_term(state, constructor, args, &classes) {
                    ready.push((term, id));
                }
            }
            if ready.is_empty() {
                return Ok(None);
            }
            ready.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
            for (_, id) in ready {
                let next = classes.len();
                classes.entry(id).or_insert(next);
            }
            pending.retain(|(id, _, _)| !classes.contains_key(id));
        }
        let mut choices = vec![Vec::new(); classes.len()];
        for (class, constructor, args) in rows {
            let Some(term) = decode_term(state, constructor, &args, &classes) else {
                return Ok(None);
            };
            choices[classes[&class]].push(term);
        }
        if choices.iter().all(|row| row.len() < 2) {
            return Ok(None);
        }
        for row in &mut choices {
            row.sort_unstable();
            row.dedup();
        }
        let mut original: Vec<(usize, usize)> = Vec::with_capacity(terms.len());
        for (term, value) in terms.iter().zip(original_values) {
            let class = classes[&value];
            let projected = match term {
                Term::Call(name, args) => Term::Call(
                    name.clone(),
                    args.iter().map(|&arg| original[arg].0).collect(),
                ),
                _ => term.clone(),
            };
            let Ok(choice) = choices[class].binary_search(&projected) else {
                return Ok(None);
            };
            original.push((class, choice));
        }
        let costs = CostModel::read(state, &classes, config)?;
        let costs = choices
            .iter()
            .enumerate()
            .map(|(class, row)| row.iter().map(|term| costs.cost(class, term)).collect())
            .collect();
        Ok(Some(Self {
            choices,
            costs,
            original,
        }))
    }

    fn improve(
        &self,
        terms: &mut Vec<Term>,
        roots: &[usize],
        mut budget: usize,
    ) -> Option<Vec<usize>> {
        let original_cost = self.original_cost(terms, roots);
        let roots: Vec<_> = roots.iter().map(|&id| self.original[id].0).collect();
        let cheapest = self.cheapest_tree_choices();

        // A size-biased tree extraction can duplicate valuable shared work.
        // Start both from that plan and from the first original term in each
        // class, and try alternative representatives against the whole DAG.
        let mut original = cheapest.clone();
        let mut has_original = vec![false; original.len()];
        for &(class, choice) in &self.original {
            if !has_original[class] {
                original[class] = choice;
                has_original[class] = true;
            }
        }
        let mut best = original_cost;
        let mut result = None;
        for mut selected in [original, cheapest] {
            let (mut cost, mut order) = self.plan(&selected, &roots)?;
            // Visit only live classes. The next pass explores alternatives in
            // classes made live by a previous change.
            loop {
                let mut improved = false;
                for class in order.clone() {
                    let mut previous = selected[class];
                    for next in 0..self.choices[class].len() {
                        if next == previous {
                            continue;
                        }
                        budget = budget.checked_sub(self.choices.len())?;
                        selected[class] = next;
                        if let Some((trial_cost, trial_order)) = self.plan(&selected, &roots)
                            && trial_cost < cost
                        {
                            previous = next;
                            cost = trial_cost;
                            order = trial_order;
                            improved = true;
                        }
                    }
                    selected[class] = previous;
                }
                if !improved {
                    break;
                }
            }
            if cost < best {
                best = cost;
                result = Some((selected, order));
            }
        }
        let (selected, order) = result?;
        let mut interned: HashMap<_, _> = terms
            .iter()
            .enumerate()
            .map(|(id, term)| (term.clone(), id))
            .collect();
        let mut ids = vec![0; self.choices.len()];
        for class in order {
            let term = match &self.choices[class][selected[class]] {
                Term::Call(name, args) => {
                    Term::Call(name.clone(), args.iter().map(|&arg| ids[arg]).collect())
                }
                term => term.clone(),
            };
            ids[class] = *interned.entry(term.clone()).or_insert_with(|| {
                terms.push(term);
                terms.len() - 1
            });
        }
        Some(roots.into_iter().map(|class| ids[class]).collect())
    }

    fn original_cost(&self, terms: &[Term], roots: &[usize]) -> u64 {
        let mut seen = vec![false; terms.len()];
        let mut pending = roots.to_vec();
        let mut cost = 0u64;
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut seen[id], true) {
                continue;
            }
            let (class, choice) = self.original[id];
            cost = cost.saturating_add(self.costs[class][choice]);
            pending.extend(children(&terms[id]));
        }
        cost
    }

    /// Choose an acyclic seed plan by additive tree cost, finalizing a choice
    /// only once all its child classes have been finalized. Real sharing is
    /// measured later and can change which choices survive.
    fn cheapest_tree_choices(&self) -> Vec<usize> {
        let mut parents = vec![Vec::new(); self.choices.len()];
        let mut remaining = vec![Vec::new(); self.choices.len()];
        let mut costs = self.costs.clone();
        let mut ready = BinaryHeap::new();
        for (class, row) in self.choices.iter().enumerate() {
            for (choice, term) in row.iter().enumerate() {
                let args = children(term);
                remaining[class].push(args.len());
                if args.is_empty() {
                    ready.push(Reverse((costs[class][choice], class, choice)));
                }
                for &arg in args {
                    parents[arg].push((class, choice));
                }
            }
        }
        let mut selected = vec![usize::MAX; self.choices.len()];
        while let Some(Reverse((cost, class, choice))) = ready.pop() {
            if selected[class] != usize::MAX {
                continue;
            }
            selected[class] = choice;
            for &(parent, alternative) in &parents[class] {
                let remaining = &mut remaining[parent][alternative];
                *remaining -= 1;
                let total = &mut costs[parent][alternative];
                *total = total.saturating_add(cost);
                if *remaining == 0 && selected[parent] == usize::MAX {
                    ready.push(Reverse((*total, parent, alternative)));
                }
            }
        }
        selected
    }

    /// Return the cost of the union of cones, and its child-first order.
    /// Terms such as `reverse(reverse(x))` can introduce cyclic e-classes,
    /// so a trial representative must first yield an acyclic query plan.
    fn plan(&self, selected: &[usize], roots: &[usize]) -> Option<(u64, Vec<usize>)> {
        let mut seen = vec![0; selected.len()];
        let mut stack: Vec<_> = roots.iter().map(|&root| (root, false)).collect();
        let mut order = Vec::new();
        let mut cost = 0u64;
        while let Some((class, finish)) = stack.pop() {
            if finish {
                seen[class] = 2;
                order.push(class);
                continue;
            }
            match seen[class] {
                2 => continue,
                1 => return None,
                _ => seen[class] = 1,
            }
            let row = selected[class];
            let term = self.choices[class].get(row)?;
            cost = cost.saturating_add(self.costs[class][row]);
            stack.push((class, true));
            stack.extend(children(term).iter().rev().map(|&arg| (arg, false)));
        }
        Some((cost, order))
    }
}

fn children(term: &Term) -> &[usize] {
    match term {
        Term::Call(_, args) => args,
        _ => &[],
    }
}

struct CostModel {
    elements: Vec<u64>,
    selectors: Vec<u64>,
    measure_work: Vec<u64>,
    folded_native: Vec<bool>,
    removed_transports: HashSet<usize>,
    poisson_rows: Vec<u64>,
    element_bytes: u64,
}

impl CostModel {
    fn read<'a, 'db: 'a>(
        state: &impl Read<'a, 'db>,
        classes: &HashMap<Value, usize>,
        config: Catalogue,
    ) -> Result<Self, Error> {
        let mut elements = vec![1; classes.len()];
        let mut selectors = vec![0; classes.len()];
        let mut measure_work = vec![0; classes.len()];
        let mut folded_native = vec![false; classes.len()];
        let mut removed_transports = HashSet::new();
        let mut poisson_rows = vec![];
        if config.measures {
            state.constructor_enodes("gaussian-folded-native", |row| {
                if let Some(&class) = classes.get(&row.children[0]) {
                    folded_native[class] = true;
                }
            })?;
            state.constructor_enodes("gaussian-removed-unit-transport", |row| {
                if let Ok(site) = usize::try_from(state.value_to_base::<i64>(row.children[1])) {
                    removed_transports.insert(site);
                }
            })?;
            for name in ["conjugate-original-measure", "measure-observation-work"] {
                state.constructor_enodes(name, |row| {
                    if let Some(&class) = classes.get(&row.children[0]) {
                        let work = state.value_to_base::<i64>(row.children[1]).max(0) as u64;
                        measure_work[class] = measure_work[class].max(work);
                    }
                })?;
            }
        }
        let Profile::StableHlo { dtype, .. } = config.profile else {
            return Ok(Self {
                elements,
                selectors,
                measure_work,
                folded_native,
                removed_transports,
                poisson_rows,
                element_bytes: 0,
            });
        };
        for name in ["real-vector", "math-real-row", "real-matrix"] {
            state.constructor_enodes(name, |row| {
                if let Some(&class) = classes.get(&row.children[0]) {
                    let size = row.children[1..].iter().fold(1u64, |size, &dim| {
                        size.saturating_mul(state.value_to_base::<i64>(dim).max(0) as u64)
                    });
                    elements[class] = elements[class].max(size);
                }
            })?;
        }
        state.constructor_enodes("fixed-selector", |row| {
            if let Some(&class) = classes.get(&row.children[1]) {
                let width = state.value_to_base::<i64>(row.children[2]);
                selectors[class] = selectors[class].max(width.max(0) as u64);
            }
        })?;
        if config.fixed_poisson {
            poisson_rows.resize(classes.len(), 0);
            state.constructor_enodes("fixed-poisson-partition", |row| {
                for (child, width) in [(0, 3), (2, 4)] {
                    if let Some(&class) = classes.get(&row.children[child]) {
                        let lanes = state.value_to_base::<i64>(row.children[width]).max(0) as u64;
                        poisson_rows[class] = poisson_rows[class].max(lanes);
                    }
                }
            })?;
        }
        // The emitter uses the integer width corresponding to the real dtype
        // for selector constants and indexing.
        let element_bytes = match dtype {
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        };
        Ok(Self {
            elements,
            selectors,
            measure_work,
            folded_native,
            removed_transports,
            poisson_rows,
            element_bytes,
        })
    }

    fn cost(&self, class: usize, term: &Term) -> u64 {
        let selector_bytes = self.selectors[class].saturating_mul(self.element_bytes);
        let moving = |lanes: u64, streams: u64| lanes.saturating_mul(streams * self.element_bytes);
        let work = match term {
            Term::Call(op, args) => match (op.as_str(), args.as_slice()) {
                ("get" | "get0", [_, selector]) if self.selectors[*selector] > 0 => {
                    // Read values and selectors, write selected values.
                    moving(self.selectors[*selector], 3)
                }
                ("get" | "get0", _) => moving(self.elements[class], 2),
                ("neg" | "reverse" | "transpose" | "adjoint" | "colstack", _) => {
                    moving(self.elements[class], 2)
                }
                ("logsumexp" | "logsumexp.pair", [x]) => moving(self.elements[*x], 2),
                ("sum" | "poisson.grouped-score", [.., plan]) if !self.poisson_rows.is_empty() => {
                    // Account for both the opaque native lanes and the hidden
                    // compact plan, including constants and reverse gathers.
                    moving(self.poisson_rows[*plan], 6)
                }
                // A named measure can be shared while each observation still
                // evaluates its marginal likelihood or restriction predicates.
                ("logdensityof" | "densityof" | "totalmass", [measure, ..]) => {
                    self.measure_work[*measure]
                }
                _ => 0,
            }
            .saturating_add(1),
            Term::Leaf(site) if self.removed_transports.contains(site) => 2,
            Term::Leaf(_) if self.folded_native[class] => 1,
            _ => 0,
        };
        work.saturating_add(selector_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_query_facts_retain_the_original() {
        let original = vec![Term::Leaf(0), Term::Call("identity".into(), vec![0])];
        let mut terms = original.clone();
        let invalid = Fact {
            name: "type-id",
            args: vec![FactArg::Expr(0)],
        };
        assert!(optimise(&mut terms, &[1], &[invalid], Profile::Portable).is_none());
        assert_eq!(terms, original);
    }

    #[test]
    fn equal_cost_retains_the_original() {
        let mut terms = vec![Term::Leaf(0)];
        let dag = ExecutionDag {
            choices: vec![vec![Term::Leaf(0), Term::Integer(0)]],
            costs: vec![vec![0, 0]],
            original: vec![(0, 0)],
        };
        assert!(dag.improve(&mut terms, &[0], MAX_TRIAL_VISITS).is_none());
        assert_eq!(terms, [Term::Leaf(0)]);
    }

    #[test]
    fn a_second_output_keeps_shared_work_live() {
        let input = Term::Leaf(0);
        let full = Term::Call("full".into(), vec![0]);
        let late = Term::Call("late".into(), vec![1]);
        let early = Term::Call("early".into(), vec![0]);
        let original = vec![input.clone(), full.clone(), late.clone()];
        let mut dag = ExecutionDag {
            choices: vec![vec![input], vec![full], vec![late, early.clone()]],
            costs: vec![vec![0], vec![10], vec![2, 5]],
            original: vec![(0, 0), (1, 0), (2, 0)],
        };
        let mut single = original.clone();
        let winner = dag.improve(&mut single, &[2], MAX_TRIAL_VISITS).unwrap();
        assert_eq!(single[winner[0]], early);

        let mut shared = original.clone();
        assert!(
            dag.improve(&mut shared, &[2, 1], MAX_TRIAL_VISITS)
                .is_none()
        );
        assert_eq!(shared, original);

        dag.costs[2][1] = 1;
        let winners = dag.improve(&mut shared, &[2, 1], MAX_TRIAL_VISITS).unwrap();
        assert_eq!(shared[winners[0]], early);
        assert_eq!(winners[1], 1);
    }

    #[test]
    fn a_cycle_and_an_exhausted_budget_preserve_a_finite_plan() {
        let input = Term::Leaf(0);
        let costly = Term::Call("costly".into(), vec![0]);
        let cycle = Term::Call("cycle".into(), vec![1]);
        let original = vec![input.clone(), costly.clone()];
        let mut dag = ExecutionDag {
            choices: vec![vec![input], vec![costly, cycle]],
            costs: vec![vec![0], vec![2, 0]],
            original: vec![(0, 0), (1, 0)],
        };
        let mut terms = original.clone();
        assert!(dag.improve(&mut terms, &[1], MAX_TRIAL_VISITS).is_none());
        assert_eq!(terms, original);

        dag.choices[1].push(Term::Integer(1));
        dag.costs[1].push(1);
        assert!(dag.improve(&mut terms, &[1], 1).is_none());
        assert_eq!(terms, original);
        let winner = dag.improve(&mut terms, &[1], MAX_TRIAL_VISITS).unwrap();
        assert_eq!(terms[winner[0]], Term::Integer(1));
    }
}
