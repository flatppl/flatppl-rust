use std::collections::{BTreeSet, HashMap};

use egglog::ast::Literal;
use egglog::{CommandOutput, EGraph, Term, TermDag, TermId};
use flatppl_core::graph::{build_call, expect_builtin_call, special_functions_alias};
use flatppl_core::{Call, CallHead, Module, Node, NodeId, Ref, RefNs};

use crate::{Emitter, Error, Step, affine_scan, contract, math, real, record, tensor};

pub(super) type RuleKey<'a> = (&'static str, &'a str, usize, usize);

pub(super) struct Rules<'a>(HashMap<RuleKey<'a>, Template>);

pub(super) struct Template {
    dag: TermDag,
    root: TermId,
}

impl<'a> Rules<'a> {
    pub(super) fn load(keys: BTreeSet<RuleKey<'a>>) -> Result<Self, Error> {
        if keys.is_empty() {
            return Ok(Self(HashMap::new()));
        }
        let mut program = concat!(
            include_str!("pullbacks.egg"),
            "\n",
            include_str!("transforms.egg"),
            "\n",
            include_str!("math.egg"),
            "\n",
            include_str!("array_math.egg"),
            "\n",
            include_str!("complex.egg"),
            "\n",
            include_str!("glm.egg"),
            "\n",
            include_str!("vector_scan.egg"),
            "\n",
            include_str!("spectral.egg")
        )
        .to_owned();
        for (catalog, op, arity, slot) in &keys {
            program.push_str(&format!("({catalog} {op:?} {arity} {slot})\n"));
        }
        if keys.iter().any(|(_, op, _, _)| {
            op.ends_with(" GLM log density")
                || op.starts_with("rowwise Categorical")
                || matches!(*op, "LKJ log density" | "LKJCholesky log density")
        }) {
            program.push_str("(run-schedule (saturate (run pullbacks)))\n");
        } else {
            program.push_str("(run pullbacks 4)\n");
        }
        for (catalog, op, arity, slot) in &keys {
            program.push_str(&format!("(extract ({catalog} {op:?} {arity} {slot}))\n"));
        }
        let outputs = EGraph::default()
            .parse_and_run_program(Some("pullbacks.egg".into()), &program)
            .map_err(|e| Error::new(None, format!("egglog pullbacks: {e}")))?;
        let templates: Vec<_> = outputs
            .into_iter()
            .filter_map(|output| match output {
                CommandOutput::ExtractBest(dag, _, root) => Some(Template { dag, root }),
                _ => None,
            })
            .collect();
        if templates.len() != keys.len() {
            return Err(Error::new(
                None,
                "egglog returned an incomplete pullback table",
            ));
        }
        Ok(Self(keys.into_iter().zip(templates).collect()))
    }

    pub(super) fn get(&self, key: RuleKey<'a>) -> Option<&Template> {
        let template = self.0.get(&key)?;
        match template.dag.get(template.root) {
            Term::App(head, _) if head == "Missing" => None,
            _ => Some(template),
        }
    }
}

impl Template {
    pub(super) fn instantiate(
        &self,
        emitter: &mut Emitter,
        step: &Step<'_>,
        bar: NodeId,
        masked: bool,
    ) -> Result<NodeId, Error> {
        if matches!(self.dag.get(self.root), Term::App(head, _) if head == "GammaShape") {
            return crate::gamma_shape::contribution(emitter, step, bar);
        }
        let mut values = Values {
            step,
            bar,
            masked,
            pointwise: step.pointwise(&emitter.module),
            terms: HashMap::new(),
        };
        self.emit(self.root, emitter, &mut values).ok_or_else(|| {
            Error::new(
                Some(step.node),
                format!(
                    "cannot instantiate the `{}` pullback for these operands",
                    step.op
                ),
            )
        })
    }

    fn emit(&self, term: TermId, emitter: &mut Emitter, values: &mut Values<'_>) -> Option<NodeId> {
        if let Some(&node) = values.terms.get(&term) {
            return Some(node);
        }
        let Term::App(head, operands) = self.dag.get(term) else {
            return None;
        };
        let node = match (head.as_str(), operands.as_slice()) {
            ("WindowTranspose", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                crate::array_math::window_transpose(emitter, values.step, values.bar, slot)?
            }
            ("OrderPartial" | "GridWeight", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                if head == "OrderPartial" {
                    crate::array_math::order_partial(emitter, values.step, slot)?
                } else {
                    crate::array_math::grid_weight(emitter, values.step, slot)?
                }
            }
            ("BernsteinWeights" | "BernsteinSlope", []) => {
                crate::array_math::bernstein(emitter, values.step, head == "BernsteinSlope")?
            }
            ("StepMask", []) => crate::array_math::step_mask(emitter, values.step)?,
            ("DiagonalScatter", []) => {
                crate::array_math::diagonal_scatter(emitter, values.step, values.bar)?
            }
            ("Arg", [index]) => {
                let index = usize::try_from(self.int(*index)?).ok()?;
                emitter.primal(*values.step.args.get(index)?)
            }
            ("Out", []) => emitter.primal(values.step.node),
            ("ParticleScore", [index]) => {
                let index = usize::try_from(self.int(*index)?).ok()?;
                crate::particle::partial(emitter, values.step, index)?
            }
            ("Bar", []) => values.bar,
            ("ZeroOut", []) => emitter.zero_like(values.step.node).ok()?,
            ("Int", [value]) => tensor::int(&mut emitter.module, self.int(*value)?),
            ("Real", [value]) => {
                let Term::Lit(Literal::Float(value)) = self.dag.get(*value) else {
                    return None;
                };
                real(&mut emitter.module, value.0)
            }
            (
                "Unary" | "Binary" | "Ternary" | "MapUnary" | "MapBinary" | "MapTernary",
                [op, operands @ ..],
            ) => {
                let Term::Lit(Literal::String(op)) = self.dag.get(*op) else {
                    return None;
                };
                let children = operands
                    .iter()
                    .map(|id| self.emit(*id, emitter, values))
                    .collect::<Option<Vec<_>>>()?;
                if values.pointwise
                    || matches!(head.as_str(), "MapUnary" | "MapBinary" | "MapTernary")
                {
                    tensor::broadcast(&mut emitter.module, op, &children)
                } else {
                    build_call(&mut emitter.module, op, &children)
                }
            }
            ("SpecialUnary" | "SpecialBinary" | "MapSpecialUnary", [member, operands @ ..]) => {
                let Term::Lit(Literal::String(member)) = self.dag.get(*member) else {
                    return None;
                };
                let args = operands
                    .iter()
                    .map(|id| self.emit(*id, emitter, values))
                    .collect::<Option<Vec<_>>>()?;
                let broadcast = values.pointwise || head == "MapSpecialUnary";
                special_call(&mut emitter.module, member, args, broadcast)
            }
            ("Fill" | "MeanFill", [bar, slot]) => {
                let index = usize::try_from(self.int(*slot)?).ok()?;
                let primal = *values.step.args.get(index)?;
                let bar = self.emit(*bar, emitter, values)?;
                if head == "Fill" {
                    tensor::fill_like(emitter, bar, primal)?
                } else {
                    tensor::mean_pullback(emitter, bar, primal)?
                }
            }
            ("Pick", [bar, index]) => {
                let bar = self.emit(*bar, emitter, values)?;
                let index = tensor::int(&mut emitter.module, self.int(*index)?);
                build_call(&mut emitter.module, "get", &[bar, index])
            }
            ("RecordPick", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                record::pick(emitter, values.step.node, values.bar, slot, values.step.op)?
            }
            ("CatSlice" | "CatEmbed", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                if head == "CatSlice" {
                    tensor::cat_slice(emitter, values.bar, &values.step.args, slot)?
                } else {
                    tensor::cat_embed(emitter, values.bar, &values.step.args, slot)?
                }
            }
            ("StackPick", [index, columns]) => tensor::stack_pick(
                &mut emitter.module,
                values.bar,
                self.int(*index)?,
                self.int(*columns)? != 0,
            ),
            ("StackInsert", [slot, columns]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let zero = emitter.zero_like(*values.step.args.get(slot)?).ok()?;
                tensor::stack_insert(
                    &mut emitter.module,
                    values.bar,
                    zero,
                    slot,
                    values.step.args.len(),
                    self.int(*columns)? != 0,
                )
            }
            ("Unstack", [columns]) => {
                let &[primal] = values.step.args.as_slice() else {
                    return None;
                };
                let (count, _) = tensor::nested_vector(&emitter.module, primal)?;
                tensor::unstack(emitter, values.bar, count, self.int(*columns)? != 0)
            }
            ("DropAxes", [bar]) => {
                let bar = self.emit(*bar, emitter, values)?;
                let [primal, leading, _] = *values.step.args.as_slice() else {
                    return None;
                };
                tensor::drop_added_axes(emitter, bar, primal, leading, values.step.node)?
            }
            ("PackRows", []) => {
                let (rows, _) = tensor::nested_vector(&emitter.module, values.step.node)?;
                crate::callbacks::pack_rows(&mut emitter.module, values.bar, rows)
            }
            ("PackColumn", [slot]) => {
                let m = &mut emitter.module;
                let matrix = build_call(m, "rowstack", &[values.bar]);
                tensor::stack_pick(m, matrix, self.int(*slot)? + 1, true)
            }
            ("PackSeed", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let primal = *values.step.args.get(slot)?;
                let zero = emitter.zero_like(primal).ok()?;
                let mut columns = vec![zero; values.step.args.len()];
                columns[slot] = values.bar;
                crate::callbacks::pack_vectors(&mut emitter.module, &columns)
            }
            ("ArrayPullback", [bar]) => {
                let bar = self.emit(*bar, emitter, values)?;
                tensor::array_pullback(emitter, bar, values.step)?
            }
            ("Scatter", [bar, base]) => {
                let bar = self.emit(*bar, emitter, values)?;
                let (&primal, selectors) = values.step.args.split_first()?;
                tensor::scatter(emitter, bar, primal, selectors, self.int(*base)?)?
            }
            ("ArrayShapePushforward", []) => {
                crate::array_transforms::pushforward(emitter, values.step, values.bar)?
            }
            ("ArrayShapePullback", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                crate::array_transforms::pullback(emitter, values.step, values.bar, slot)?
            }
            ("BetaShape", [slot]) => crate::beta_shape::contribution(
                emitter,
                values.step,
                values.bar,
                self.int(*slot)? == 1,
            )?,
            ("AirySlope", [argument]) => {
                let argument = self.emit(*argument, emitter, values)?;
                let module = &mut emitter.module;
                if values.pointwise {
                    let mut entries = Vec::new();
                    let x = crate::callbacks::callback_input(module, "airy_argument", &mut entries);
                    let body = crate::special::airy_slope(module, x);
                    let function = crate::callbacks::reify(module, body, entries);
                    build_call(module, "broadcast", &[function, argument])
                } else {
                    crate::special::airy_slope(module, argument)
                }
            }
            ("BesselOrder", [member, order, argument]) => {
                let Term::Lit(Literal::String(member)) = self.dag.get(*member) else {
                    return None;
                };
                let order = self.emit(*order, emitter, values)?;
                let argument = self.emit(*argument, emitter, values)?;
                crate::special::bessel_order(&mut emitter.module, order, argument, member)
            }
            ("BesselKOrder", [member, order, argument]) => {
                let Term::Lit(Literal::String(member)) = self.dag.get(*member) else {
                    return None;
                };
                let order = self.emit(*order, emitter, values)?;
                let argument = self.emit(*argument, emitter, values)?;
                if values.pointwise {
                    return None;
                }
                crate::special::bessel_k_order(&mut emitter.module, order, argument, member)
            }
            ("BesselYOrder", [order, argument]) => {
                let order = self.emit(*order, emitter, values)?;
                let argument = self.emit(*argument, emitter, values)?;
                if values.pointwise {
                    return None;
                }
                crate::special::bessel_y_order(&mut emitter.module, order, argument)
            }
            ("CumulativeProduct", [forward]) => {
                let [input] = *values.step.args.as_slice() else {
                    return None;
                };
                crate::cumulative::product(
                    emitter,
                    input,
                    values.bar,
                    self.int(*forward)? != 0,
                    values.masked,
                )?
            }
            ("CumulativeExtreme" | "CumulativeExtremeForward", [maximum]) => {
                let [input] = *values.step.args.as_slice() else {
                    return None;
                };
                tensor::cumulative_extreme(
                    emitter,
                    values.bar,
                    input,
                    values.step.node,
                    self.int(*maximum)? == 1,
                    head == "CumulativeExtremeForward",
                )?
            }
            ("LogSumExp", [forward]) => {
                let [input] = *values.step.args.as_slice() else {
                    return None;
                };
                tensor::logsumexp(
                    emitter,
                    input,
                    values.step.node,
                    values.bar,
                    self.int(*forward)? != 0,
                )?
            }
            ("ForwardCall", [name, slot]) => {
                let Term::Lit(Literal::String(name)) = self.dag.get(*name) else {
                    return None;
                };
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let mut args: Vec<_> = values
                    .step
                    .args
                    .iter()
                    .map(|&id| emitter.primal(id))
                    .collect();
                *args.get_mut(slot)? = values.bar;
                build_call(&mut emitter.module, name, &args)
            }
            ("RowReduce", [op, matrix]) => {
                let Term::Lit(Literal::String(op)) = self.dag.get(*op) else {
                    return None;
                };
                let matrix = self.emit(*matrix, emitter, values)?;
                tensor::reduce_rows(&mut emitter.module, op, matrix)
            }
            ("FlattenRows" | "RestoreRows", [value, slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let primal = *values.step.args.get(slot)?;
                let value = self.emit(*value, emitter, values)?;
                if let Some((rows, _)) = tensor::nested_vector(&emitter.module, primal) {
                    if head == "FlattenRows" {
                        build_call(&mut emitter.module, "rowstack", &[value])
                    } else {
                        crate::callbacks::pack_rows(&mut emitter.module, value, rows)
                    }
                } else {
                    value
                }
            }
            ("ProbabilityPropagation", [slot, forward, state, argument]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let forward = self.int(*forward)? != 0;
                let state = self.emit(*state, emitter, values)?;
                let argument = self.emit(*argument, emitter, values)?;
                crate::probability_scan::propagate(
                    emitter,
                    values.step,
                    values.bar,
                    state,
                    argument,
                    slot,
                    forward,
                )?
            }
            ("MarkovScan", [slot, forward]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                if self.int(*forward)? == 0 {
                    crate::vector_scan::pullback(
                        emitter,
                        values.step,
                        values.bar,
                        slot,
                        values.masked,
                    )?
                } else {
                    crate::vector_scan::pushforward(emitter, values.step, values.bar, slot)?
                }
            }
            ("AffineScan", [initial, sequence, scale, guarded]) => {
                let initial = self.emit(*initial, emitter, values)?;
                let sequence = self.emit(*sequence, emitter, values)?;
                let scale = self.emit(*scale, emitter, values)?;
                let guarded = self.int(*guarded)? != 0;
                affine_scan::scan(&mut emitter.module, initial, sequence, scale, guarded)
            }
            ("LaggedScan", [initial, states]) => {
                let shape = tensor::real_shape(&emitter.module, values.step.node)?;
                let &[flatppl_core::Dim::Static(length)] = shape else {
                    return None;
                };
                let initial = self.emit(*initial, emitter, values)?;
                let states = self.emit(*states, emitter, values)?;
                affine_scan::lag(&mut emitter.module, initial, states, length)
            }
            ("MathDot", [left, right]) => {
                let left = self.emit(*left, emitter, values)?;
                let right = self.emit(*right, emitter, values)?;
                math::dot(&mut emitter.module, left, right)
            }
            ("MatrixPullback", [slot, regular]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                if values.masked {
                    contract::matrix_pullback(emitter, values.step, values.bar, slot)?
                } else {
                    self.emit(*regular, emitter, values)?
                }
            }
            ("IndexedProduct", [slot, forward]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let forward = self.int(*forward)? != 0;
                let call = expect_builtin_call(&emitter.module, values.step.node, "aggregate")?;
                let contraction = contract::SumProduct::read(&emitter.module, &call.args)?;
                contraction.contribution(emitter, values.bar, slot, forward, values.masked)?
            }
            ("PolynomialSlope", []) => {
                let [coefficients, x] = *values.step.args.as_slice() else {
                    return None;
                };
                math::polynomial_slope(emitter, coefficients, x)?
            }
            ("PolynomialBasis", []) => {
                let [coefficients, x] = *values.step.args.as_slice() else {
                    return None;
                };
                math::polynomial_basis(emitter, coefficients, x)?
            }
            ("SampleDegrees", []) => {
                let [primal] = *values.step.args.as_slice() else {
                    return None;
                };
                let degrees = math::sample_degrees(&emitter.module, primal)? as f64;
                real(&mut emitter.module, degrees)
            }
            ("ProductPartials", []) => {
                let [primal] = *values.step.args.as_slice() else {
                    return None;
                };
                math::product_partials(emitter, primal)?
            }
            ("IdentityLike", [slot]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                math::identity(emitter, *values.step.args.get(slot)?)?
            }
            ("ComplexExponentialPart", [slot, forward]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let forward = self.int(*forward)? != 0;
                crate::linalg::complex_exponential_part(
                    emitter,
                    values.step,
                    values.bar,
                    slot,
                    forward,
                )?
            }
            ("EigenProjectionPart", [forward]) => {
                let forward = self.int(*forward)? != 0;
                crate::linalg::projection_part(emitter, values.step, values.bar, forward)?
            }
            ("SpectrumPart", [forward]) => {
                let forward = self.int(*forward)? != 0;
                crate::linalg::spectrum_part(emitter, values.step, values.bar, forward)?
            }
            ("FactorizationPart", [forward]) => {
                let forward = self.int(*forward)? != 0;
                crate::linalg::factorization_part(emitter, values.step, values.bar, forward)?
            }
            ("KronPart" | "FactorSolvePart", [slot, forward]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let forward = self.int(*forward)? != 0;
                if head == "KronPart" {
                    crate::linalg::kron_part(emitter, values.step, values.bar, slot, forward)?
                } else {
                    crate::linalg::solve_part(emitter, values.step, values.bar, slot, forward)?
                }
            }
            ("MatrixExpFrechet", [base, direction]) => {
                let base = self.emit(*base, emitter, values)?;
                let direction = self.emit(*direction, emitter, values)?;
                math::matrix_exp_frechet(
                    emitter,
                    values.step.node,
                    values.step.args[0],
                    base,
                    direction,
                )?
            }
            ("SpectralProduct", [slot, forward]) => {
                let slot = usize::try_from(self.int(*slot)?).ok()?;
                let forward = self.int(*forward)? != 0;
                let result =
                    crate::spectral::product(emitter, values.step, values.bar, slot, forward)?;
                if values.masked {
                    crate::spectral::mask(emitter, values.step, slot, values.bar, result)?
                } else {
                    result
                }
            }
            ("DetCofactors", []) => math::cofactors(emitter, values.step.args[0])?,
            ("TriangleHalf", [value]) => {
                let value = self.emit(*value, emitter, values)?;
                math::triangle_half(emitter, values.step.node, value)?
            }
            ("Symmetric", [value]) => {
                let value = self.emit(*value, emitter, values)?;
                math::symmetric(&mut emitter.module, value)
            }
            ("Insert", [index]) => {
                let index = self.int(*index)?;
                if emitter.module.type_of(values.step.node).is_some_and(|ty| {
                    tensor::nested_vector_type(ty).is_some()
                        || tensor::vector_of_arrays(ty).is_some()
                }) {
                    let slot = usize::try_from(index - 1).ok()?;
                    let zero = emitter.zero_like(*values.step.args.get(slot)?).ok()?;
                    tensor::insert_nested(
                        &mut emitter.module,
                        values.bar,
                        zero,
                        slot,
                        values.step.args.len(),
                    )
                } else {
                    tensor::insert(emitter, values.bar, index, values.step.args.len())
                }
            }
            _ => return None,
        };
        values.terms.insert(term, node);
        Some(node)
    }

    fn int(&self, term: TermId) -> Option<i64> {
        match self.dag.get(term) {
            Term::Lit(Literal::Int(value)) => Some(*value),
            _ => None,
        }
    }
}

struct Values<'a> {
    step: &'a Step<'a>,
    bar: NodeId,
    masked: bool,
    pointwise: bool,
    terms: HashMap<TermId, NodeId>,
}

pub(super) fn special_call(
    module: &mut Module,
    member: &str,
    mut args: Vec<NodeId>,
    broadcast: bool,
) -> NodeId {
    let alias = special_functions_alias(module);
    let name = module.intern(member);
    let callee = module.alloc(Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }));
    if broadcast {
        args.insert(0, callee);
        return build_call(module, "broadcast", &args);
    }
    module.alloc(Node::Call(Call {
        head: CallHead::User(callee),
        args: args.into(),
        named: Vec::new().into(),
        inputs: None,
    }))
}
