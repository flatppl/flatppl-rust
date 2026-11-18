use crate::{
    DataDocument, DataValue, Design, DesignComponent, Error, Family, ModelInfo, Parameter,
    ParameterRole, Recipe, ResolvedPrior, Result, Spec, Trials, metadata,
    prior_model::ParameterPrior,
    priors::{self, PriorNode},
};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs};

pub(crate) fn model(recipe: &mut Recipe) -> Result<Module> {
    let mut module = Module::new();
    let mut b = Builder::new(&mut module);
    b.stamp_compat();
    let count_set = b.call_head("posintegers");
    let size = b.call("external", &[count_set]);
    b.bind("N", size);
    external_vector(
        &mut b,
        "y_data",
        if recipe.family.discrete() {
            "nonnegintegers"
        } else {
            "reals"
        },
    );
    if matches!(recipe.formula.trials, Some(Trials::Column(_))) {
        external_vector(&mut b, "trials_data", "nonnegintegers");
    }
    if recipe.formula.weights.is_some() {
        external_vector(&mut b, "weights_data", "nonnegreals");
    }
    if recipe.formula.se.is_some() {
        external_vector(&mut b, "se_data", "nonnegreals");
    }
    crate::response::declare(&mut b, &recipe.formula);
    if let Some(nonlinear) = &recipe.nonlinear {
        for i in 0..nonlinear.columns.len() {
            external_vector(&mut b, &crate::nonlinear::data_name(i), "reals");
        }
    }
    let predictor_names: Vec<_> = recipe
        .predictors()
        .map(|(name, _)| name.to_owned())
        .collect();
    for name in &predictor_names {
        let suffix = recipe.predictor_suffix(name);
        predictor_data(&mut b, &suffix, recipe.predictors.get_mut(name).unwrap());
    }
    let mut parameters = Vec::new();
    let thresholds = recipe.thresholds();
    for name in &predictor_names {
        let suffix = recipe.predictor_suffix(name);
        predictor_parameters(
            &mut b,
            &suffix,
            recipe.predictors.get_mut(name).unwrap(),
            recipe.family,
            if name == "mu" { thresholds } else { 0 },
            &mut parameters,
        )?;
    }
    for &name in recipe.family.auxiliaries() {
        if recipe.predictors.contains_key(name) {
            continue;
        }
        if name == "sigma" && !recipe.auxiliary_priors.contains_key(name) {
            let zero = b.lit_real(0.0);
            b.bind(name, zero);
            continue;
        }
        let support = if matches!(name, "zi" | "hu") {
            let zero = b.lit_real(0.0);
            let one = b.lit_real(1.0);
            b.call("interval", &[zero, one])
        } else {
            lower_bound(&mut b, if name == "nu" { 1.0 } else { 0.0 })
        };
        parameter(
            &mut b,
            name,
            &recipe.auxiliary_priors[name],
            Some(support),
            &mut parameters,
            recipe.family,
        )?;
    }
    if parameters.is_empty() {
        return Err(Error(
            "a formula model needs at least one sampled parameter".into(),
        ));
    }
    let prior = crate::prior_model::build(&mut b, &parameters)?;
    let parameters: Vec<_> = parameters
        .into_iter()
        .map(|parameter| parameter.name)
        .collect();
    b.bind("Pi_prior", prior);

    for (name, predictor) in recipe.predictors() {
        if name == "mu" && recipe.nonlinear.is_some() {
            continue;
        }
        predictor_values(
            &mut b,
            &recipe.predictor_suffix(name),
            predictor,
            if name == "mu" { thresholds } else { 0 },
            recipe.family,
            recipe.is_nonlinear_parameter(name),
        );
    }
    if let Some(nonlinear) = &recipe.nonlinear {
        let eta = nonlinear.emit(&mut b);
        b.bind("eta_mu", eta);
        let eta = b.self_ref("eta_mu");
        let mu = if thresholds > 0 {
            let cuts = b.self_ref("cuts");
            b.bind("b_Intercept", cuts);
            eta
        } else {
            recipe.link.inverse(&mut b, eta)
        };
        b.bind("mu", mu);
    }
    let mu = if recipe.family == Family::Categorical {
        b.lit_real(0.0)
    } else {
        b.self_ref("mu")
    };
    let trials = recipe.formula.trials.as_ref().map(|trials| match trials {
        Trials::Constant(n) => b.lit_int(*n),
        Trials::Column(_) => b.self_ref("trials_data"),
    });
    let scale = recipe.formula.se.as_ref().map(|se| {
        let known = b.self_ref("se_data");
        if !se.sigma {
            return known;
        }
        let sigma = b.self_ref("sigma");
        let residual = broadcast(&mut b, "mul", &[sigma, sigma]);
        let known = broadcast(&mut b, "mul", &[known, known]);
        let variance = broadcast(&mut b, "add", &[known, residual]);
        broadcast(&mut b, "sqrt", &[variance])
    });
    let auxiliary: Vec<_> = observation_auxiliaries(recipe)
        .into_iter()
        .filter(|name| !(*name == "sigma" && scale.is_some()))
        .map(|name| (name, b.self_ref(name)))
        .collect();
    let cuts = threshold_inputs(&mut b, recipe);
    let observation = crate::family::Observation {
        mu,
        trials,
        scale,
        auxiliary: &auxiliary,
        cuts: &cuts,
    };
    let law = if let Some(truncation) = &recipe.formula.truncation {
        let sigma = scale.unwrap_or_else(|| b.self_ref("sigma"));
        crate::response::truncated_normal(&mut b, mu, sigma, truncation)
    } else {
        recipe.family.observation(&mut b, observation, false)
    };
    let y = b.call("draw", &[law]);
    b.bind("y", y);
    let y = b.self_ref("y");
    let response = b.call_fields("record", &[("y", y)]);
    let kernel = parameter_function(&mut b, "kernelof", response, &parameters);
    b.bind("K_model", kernel);
    if let Some(expected) = crate::expectation::emit(&mut b, recipe, observation) {
        let expected = parameter_function(&mut b, "functionof", expected, &parameters);
        b.bind_doc(
            "expected_response",
            expected,
            &["Conditional response means or category probabilities in training order."],
        );
    }
    let pointwise = pointwise_log_likelihood(&mut b, recipe, scale);
    let likelihood = if recipe.formula.censoring.is_some() {
        crate::response::observed_likelihood(&mut b, pointwise, &parameters)
    } else {
        let y = b.self_ref("y_data");
        let observation = b.call_fields("record", &[("y", y)]);
        let kernel = b.self_ref("K_model");
        b.call("likelihoodof", &[kernel, observation])
    };
    b.bind("L", likelihood);
    let log_lik = parameter_function(&mut b, "functionof", pointwise, &parameters);
    b.bind_doc(
        "log_lik",
        log_lik,
        &["Unweighted log likelihood for each observed row."],
    );
    let prior = b.self_ref("Pi_prior");
    let posterior = if recipe.formula.weights.is_some() {
        let point = Ref {
            ns: RefNs::Local,
            name: b.sym("_point_"),
        };
        let value = b.m.alloc(Node::Ref(point));
        let named = parameters
            .iter()
            .map(|name| {
                let field = b.str_lit(name);
                let value = b.call("get", &[value, field]);
                NamedArg {
                    kind: NamedKind::Kwarg,
                    name: b.sym(name),
                    value,
                }
            })
            .collect::<Vec<_>>();
        let log_lik = b.self_ref("log_lik");
        let pointwise = b.m.alloc(Node::Call(Call {
            head: CallHead::User(log_lik),
            args: Box::default(),
            named: named.into(),
            inputs: None,
        }));
        let weights = b.self_ref("weights_data");
        let weighted = broadcast(&mut b, "mul", &[weights, pointwise]);
        let score = b.call("sum", &[weighted]);
        let head = b.sym("functionof");
        let name = b.sym("point");
        let weight = b.m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: Box::new([score]),
            named: Box::default(),
            inputs: Some(Inputs::Spec(Box::new([(name, point)]))),
        }));
        b.call("logweighted", &[weight, prior])
    } else {
        let likelihood = b.self_ref("L");
        b.call("bayesupdate", &[likelihood, prior])
    };
    b.bind("Pi_post", posterior);
    Ok(module)
}

pub(crate) fn predictor_name(dpar: &str, name: &str) -> String {
    if dpar == "mu" {
        name.into()
    } else {
        format!("{name}_{dpar}")
    }
}

fn predictor_data(b: &mut Builder<'_>, dpar: &str, recipe: &mut crate::PredictorRecipe) {
    for i in 1..=recipe.monotonic.len() {
        external_vector(b, &predictor_name(dpar, &format!("mo_{i}")), "posintegers");
    }
    for (i, smooth) in recipe.smooths.iter_mut().enumerate() {
        smooth.inputs = (0..smooth.basis.transform[0].len())
            .map(|j| {
                let name = b.alloc_name(&predictor_name(
                    dpar,
                    &format!("smooth_{}_{}", i + 1, j + 1),
                ));
                external_vector(b, &name, "reals");
                name
            })
            .collect();
        if let Some(by) = &mut smooth.by {
            by.index = b.alloc_name(&predictor_name(dpar, &format!("smooth_{}_by", i + 1)));
            external_vector(b, &by.index, "posintegers");
        }
    }
    if !recipe.formula.offsets.is_empty() {
        external_vector(b, &predictor_name(dpar, "offset"), "reals");
    }
    for (i, predictor) in recipe.design.predictors.iter().enumerate() {
        let set = if predictor.factor.is_some() {
            "posintegers"
        } else {
            "reals"
        };
        external_vector(b, &predictor_name(dpar, &format!("x_{}", i + 1)), set);
    }
    let fixed_inputs: Vec<_> = (0..recipe.design.predictors.len())
        .map(|i| predictor_name(dpar, &format!("x_{}", i + 1)))
        .collect();
    let mut input_names: std::collections::HashMap<_, _> = recipe
        .design
        .predictors
        .iter()
        .zip(&fixed_inputs)
        .map(|(predictor, name)| (predictor.column.clone(), name.clone()))
        .collect();
    for (group_index, group) in recipe.groups.iter_mut().enumerate() {
        group.index = b.alloc_name(&predictor_name(dpar, &format!("index_{}", group.column)));
        external_vector(b, &group.index, "posintegers");
        group.inputs = group
            .design
            .predictors
            .iter()
            .enumerate()
            .map(|(i, predictor)| {
                if let Some(name) = input_names.get(&predictor.column) {
                    return name.clone();
                }
                let name = b.alloc_name(&format!("group_{}_x_{}", group_index + 1, i + 1));
                let set = if predictor.factor.is_some() {
                    "posintegers"
                } else {
                    "reals"
                };
                external_vector(b, &name, set);
                input_names.insert(predictor.column.clone(), name.clone());
                name
            })
            .collect();
    }
}

fn predictor_parameters(
    b: &mut Builder<'_>,
    dpar: &str,
    recipe: &mut crate::PredictorRecipe,
    family: Family,
    thresholds: usize,
    parameters: &mut Vec<ParameterPrior>,
) -> Result<()> {
    if thresholds > 0 {
        let names = scope(b, family);
        let bindings: Vec<_> = names.iter().map(String::as_str).collect();
        let PriorNode::Measure(prior) =
            priors::emit(b, &recipe.priors.intercept, None, None, &bindings)?
        else {
            return Err(Error(
                "cumulative thresholds require a shared continuous prior".into(),
            ));
        };
        let count = b.lit_int(thresholds as i64);
        let prior = b.call("iid", &[prior, count]);
        let domain = b.call("ordered", &[count]);
        let reference = b.call("Lebesgue", &[domain]);
        let prior = b.call("tilt", &[prior, reference]);
        declare_parameter(b, "cuts", prior, parameters);
    } else if recipe.formula.intercept {
        parameter(
            b,
            &predictor_name(dpar, "temp_Intercept"),
            &recipe.priors.intercept,
            None,
            parameters,
            family,
        )?;
    }
    for (name, priors) in [("b", &recipe.priors.b), ("bs", &recipe.priors.bs)] {
        if priors.is_empty() {
            continue;
        }
        if priors.iter().all(|p| *p == priors[0]) {
            let scope = scope(b, family);
            let bindings: Vec<_> = scope.iter().map(String::as_str).collect();
            let prior = priors::emit(b, &priors[0], None, None, &bindings)?;
            let count = b.lit_int(priors.len() as i64);
            match prior {
                PriorNode::Measure(prior) => {
                    let iid = b.call("iid", &[prior, count]);
                    declare_parameter(b, &predictor_name(dpar, name), iid, parameters);
                }
                PriorNode::Constant(value) => {
                    let values = b.call("fill", &[value, count]);
                    b.bind(&predictor_name(dpar, name), values);
                }
            }
        } else {
            let mut values = Vec::new();
            for (i, prior) in priors.iter().enumerate() {
                let name = predictor_name(dpar, &format!("{name}_{}", i + 1));
                parameter(b, &name, prior, None, parameters, family)?;
                values.push(b.self_ref(&name));
            }
            let coefficients = b.array(&values);
            b.bind(&predictor_name(dpar, name), coefficients);
        }
    }
    let mut scale_priors = recipe.priors.sds.iter();
    for (smooth, term) in recipe.smooths.iter_mut().zip(&recipe.formula.smooths) {
        let rank = b.lit_int(smooth.inputs.len() as i64 - 1);
        for (level, sds, z) in smooth.curves_mut() {
            let name = term.coefficient(level);
            *sds = b.alloc_name(&predictor_name(dpar, &format!("sds_{name}")));
            *z = b.alloc_name(&predictor_name(dpar, &format!("zs_{name}")));
            let support = lower_bound(b, 0.0);
            parameter(
                b,
                sds,
                scale_priors.next().unwrap(),
                Some(support),
                parameters,
                family,
            )?;
            let zero = b.lit_real(0.0);
            let one = b.lit_real(1.0);
            let normal = b.call("Normal", &[zero, one]);
            let law = b.call("iid", &[normal, rank]);
            declare_parameter(b, z, law, parameters);
        }
    }
    for (i, coding) in recipe.monotonic.iter().enumerate() {
        let index = i + 1;
        let name = predictor_name(dpar, &format!("bmo_{index}"));
        parameter(b, &name, &recipe.priors.bmo[i], None, parameters, family)?;
        let name = predictor_name(dpar, &format!("simo_{index}"));
        let steps = coding.steps();
        let law = crate::prior_expr::emit_simplex(b, &recipe.priors.simo[i], steps)?;
        if steps == 1 {
            // A two-level predictor has only one possible simplex.
            let one = b.lit_real(1.0);
            let singleton = b.array(&[one]);
            b.bind(&name, singleton);
        } else {
            declare_parameter(b, &name, law, parameters);
        }
    }
    for (i, group) in recipe.groups.iter_mut().enumerate() {
        group.sd = b.alloc_name(&predictor_name(dpar, &format!("sd_{}", group.column)));
        let dimensions = group.coefficients.len();
        let dimension = b.lit_int(dimensions as i64);
        let support = lower_bound(b, 0.0);
        let sd_priors = &recipe.priors.sd[i];
        if dimensions == 1 {
            parameter(
                b,
                &group.sd,
                &sd_priors[0],
                Some(support),
                parameters,
                family,
            )?;
        } else if sd_priors.iter().all(|prior| *prior == sd_priors[0]) {
            let names = scope(b, family);
            let bindings: Vec<_> = names.iter().map(String::as_str).collect();
            match priors::emit(b, &sd_priors[0], Some(support), None, &bindings)? {
                PriorNode::Measure(law) => {
                    let law = b.call("iid", &[law, dimension]);
                    declare_parameter(b, &group.sd, law, parameters);
                }
                PriorNode::Constant(value) => {
                    let value = b.call("fill", &[value, dimension]);
                    b.bind(&group.sd, value);
                }
            }
        } else {
            let mut values = Vec::new();
            for (j, prior) in sd_priors.iter().enumerate() {
                let name = b.alloc_name(&format!("{}_{}", group.sd, j + 1));
                parameter(b, &name, prior, Some(support), parameters, family)?;
                values.push(b.self_ref(&name));
            }
            let value = b.array(&values);
            b.bind(&group.sd, value);
        }
        group.z = b.alloc_name(&predictor_name(dpar, &format!("z_{}", group.column)));
        let iid = group_innovations(b, dimensions, group.levels.len());
        declare_parameter(b, &group.z, iid, parameters);
        group.r = b.alloc_name(&predictor_name(dpar, &format!("r_{}", group.column)));
        let sd = b.self_ref(&group.sd);
        let z = b.self_ref(&group.z);
        let effect = if dimensions == 1 {
            broadcast(b, "mul", &[sd, z])
        } else {
            let mut scale = b.call("diagmat", &[sd]);
            if let Some(prior) = &recipe.priors.cor[i] {
                let name = b.alloc_name(&predictor_name(dpar, &format!("L_{}", group.column)));
                let names = scope(b, family);
                let bindings: Vec<_> = names.iter().map(String::as_str).collect();
                let law = priors::emit_correlation(b, prior, dimensions, &bindings)?;
                declare_parameter(b, &name, law, parameters);
                let cholesky = b.self_ref(&name);
                scale = b.call("mul", &[scale, cholesky]);
                group.correlation = Some(name);
                let name = b.alloc_name(&predictor_name(dpar, &format!("Cor_{}", group.column)));
                let transposed = b.call("transpose", &[cholesky]);
                let correlation = b.call("mul", &[cholesky, transposed]);
                b.bind(&name, correlation);
                group.correlation_matrix = Some(name);
            }
            let effect = b.call("mul", &[scale, z]);
            b.call("transpose", &[effect])
        };
        b.bind(&group.r, effect);
    }
    Ok(())
}

fn group_innovations(b: &mut Builder<'_>, coefficients: usize, levels: usize) -> NodeId {
    let zero = b.lit_real(0.0);
    let one = b.lit_real(1.0);
    let standard = b.call("Normal", &[zero, one]);
    let count = b.lit_int(levels as i64);
    let size = if coefficients == 1 {
        count
    } else {
        let dimension = b.lit_int(coefficients as i64);
        b.array(&[dimension, count])
    };
    b.call("iid", &[standard, size])
}

pub(crate) fn extend_group_draws(
    recipe: &Recipe,
    mut template: Module,
    shapes: &crate::NewGroupShapes,
) -> Result<Module> {
    let mut b = Builder::new(&mut template);
    for (name, shape) in shapes {
        let group = recipe
            .predictors()
            .flat_map(|(_, predictor)| &predictor.groups)
            .find(|group| group.z == *name)
            .ok_or_else(|| Error(format!("unknown group parameter `{name}`")))?;
        let coefficients = group.coefficients.len();
        let new_count = match shape.as_slice() {
            [count] if coefficients == 1 => *count,
            [rows, count] if coefficients > 1 && *rows == coefficients => *count,
            _ => return Err(Error(format!("invalid innovation shape for `{name}`"))),
        };
        let levels = group
            .levels
            .len()
            .checked_add(new_count)
            .filter(|count| u32::try_from(*count).is_ok())
            .ok_or_else(|| Error(format!("too many levels in group `{}`", group.column)))?;
        let symbol = b.sym(name);
        let binding =
            b.m.binding_by_name(symbol)
                .ok_or_else(|| Error(format!("missing group parameter `{name}`")))?;
        let iid = group_innovations(&mut b, coefficients, levels);
        let draw = b.call("draw", &[iid]);
        b.m.set_binding_rhs(binding, draw);
    }
    Ok(template)
}

fn predictor_values(
    b: &mut Builder<'_>,
    dpar: &str,
    recipe: &crate::PredictorRecipe,
    thresholds: usize,
    family: Family,
    nonlinear_parameter: bool,
) {
    let fixed_inputs: Vec<_> = (0..recipe.design.predictors.len())
        .map(|i| predictor_name(dpar, &format!("x_{}", i + 1)))
        .collect();
    let n = b.self_ref("N");
    let zero = b.lit_real(0.0);
    let mut eta = b.call("fill", &[zero, n]);
    let mut intercept = recipe.formula.intercept.then(|| {
        b.self_ref(&if thresholds > 0 {
            "cuts".into()
        } else {
            predictor_name(dpar, "temp_Intercept")
        })
    });
    if !recipe.design.columns.is_empty() {
        let mut columns = design_columns(b, &recipe.design, &fixed_inputs);
        if recipe.center {
            for (column, &mean) in columns.iter_mut().zip(&recipe.means) {
                let mean = b.lit_real(mean);
                *column = broadcast(b, "sub", &[*column, mean]);
            }
        }
        let columns = b.array(&columns);
        let design = b.call("colstack", &[columns]);
        b.bind(&predictor_name(dpar, "X"), design);
        let x = b.self_ref(&predictor_name(dpar, "X"));
        let coefficients = b.self_ref(&predictor_name(dpar, "b"));
        eta = b.call("mul", &[x, coefficients]);
        if recipe.center {
            let means: Vec<_> = recipe.means.iter().map(|&v| b.lit_real(v)).collect();
            let means = b.array(&means);
            let products = broadcast(b, "mul", &[means, coefficients]);
            let dot = b.call("sum", &[products]);
            intercept = intercept.map(|value| {
                if thresholds > 0 {
                    broadcast(b, "add", &[value, dot])
                } else {
                    b.call("sub", &[value, dot])
                }
            });
        }
    }
    if recipe.formula.intercept && thresholds == 0 {
        let value = b.self_ref(&predictor_name(dpar, "temp_Intercept"));
        eta = broadcast(b, "add", &[eta, value]);
    }
    let mut smooth_offset = 0;
    for smooth in &recipe.smooths {
        let contribution = smooth_effect(b, dpar, smooth, smooth_offset);
        eta = broadcast(b, "add", &[eta, contribution]);
        smooth_offset += smooth.curve_count();
    }
    for (i, coding) in recipe.monotonic.iter().enumerate() {
        let index = i + 1;
        let simplex = b.self_ref(&predictor_name(dpar, &format!("simo_{index}")));
        let prefix = b.call("cumsum", &[simplex]);
        let zero = b.lit_real(0.0);
        let baseline = b.array(&[zero]);
        let levels = b.call("cat", &[baseline, prefix]);
        let codes = b.self_ref(&predictor_name(dpar, &format!("mo_{index}")));
        let selected = b.call("get", &[levels, codes]);
        let slope = b.self_ref(&predictor_name(dpar, &format!("bmo_{index}")));
        // brms defines the slope per step, so the total change is steps times the slope.
        let steps = b.lit_real(coding.steps() as f64);
        let total = b.call("mul", &[slope, steps]);
        let contribution = broadcast(b, "mul", &[selected, total]);
        eta = broadcast(b, "add", &[eta, contribution]);
    }
    for (i, group) in recipe.groups.iter().enumerate() {
        let mut columns = design_columns(b, &group.design, &group.inputs);
        if recipe.formula.groups[i].intercept {
            let one = b.lit_real(1.0);
            columns.insert(0, b.call("fill", &[one, n]));
        }
        for (j, column) in columns.into_iter().enumerate() {
            let name = b.alloc_name(&format!("Z_{}_{}", i + 1, j + 1));
            b.bind(&name, column);
            let values = b.self_ref(&name);
            let effect = b.self_ref(&group.r);
            let index = b.self_ref(&group.index);
            let selected = if group.coefficients.len() == 1 {
                b.call("get", &[effect, index])
            } else {
                let column = b.lit_int(j as i64 + 1);
                b.call("get", &[effect, index, column])
            };
            let contribution = broadcast(b, "mul", &[selected, values]);
            eta = broadcast(b, "add", &[eta, contribution]);
        }
    }
    if !recipe.formula.offsets.is_empty() {
        let offset = b.self_ref(&predictor_name(dpar, "offset"));
        eta = broadcast(b, "add", &[eta, offset]);
    }
    b.bind(&format!("eta_{dpar}"), eta);
    if !nonlinear_parameter {
        let eta = b.self_ref(&format!("eta_{dpar}"));
        let value = if thresholds > 0 || family == Family::Categorical {
            eta
        } else {
            recipe.link.inverse(b, eta)
        };
        b.bind(dpar, value);
    }
    if let Some(intercept) = intercept {
        b.bind(&predictor_name(dpar, "b_Intercept"), intercept);
    }
}

fn smooth_effect(b: &mut Builder<'_>, dpar: &str, smooth: &crate::Smooth, offset: usize) -> NodeId {
    let slopes = b.self_ref(&predictor_name(dpar, "bs"));
    let weights: Vec<_> = smooth
        .curves()
        .enumerate()
        .map(|(i, (_, sds, z))| {
            let index = b.lit_int((offset + i + 1) as i64);
            let fixed = b.call("get", &[slopes, index]);
            let scale = b.self_ref(sds);
            let z = b.self_ref(z);
            (fixed, broadcast(b, "mul", &[scale, z]))
        })
        .collect();
    if let Some(by) = &smooth.by {
        let index = b.self_ref(&by.index);
        // Each row gathers the weights for its own level from the shared basis.
        let mut terms = Vec::new();
        for (j, input) in smooth.inputs.iter().enumerate() {
            let mut coefficients = Vec::with_capacity(by.levels.len());
            if by.first_curve() == 1 {
                coefficients.push(b.lit_real(0.0));
            }
            for &(fixed, random) in &weights {
                coefficients.push(if j == 0 {
                    fixed
                } else {
                    let column = b.lit_int(j as i64);
                    b.call("get", &[random, column])
                });
            }
            let coefficients = b.array(&coefficients);
            let selected = b.call("get", &[coefficients, index]);
            let values = b.self_ref(input);
            terms.push(broadcast(b, "mul", &[selected, values]));
        }
        return terms
            .into_iter()
            .reduce(|left, right| broadcast(b, "add", &[left, right]))
            .unwrap();
    }
    let (fixed, random) = weights[0];
    let linear = b.self_ref(&smooth.inputs[0]);
    let linear = broadcast(b, "mul", &[linear, fixed]);
    let columns: Vec<_> = smooth.inputs[1..]
        .iter()
        .map(|name| b.self_ref(name))
        .collect();
    let columns = b.array(&columns);
    let basis = b.call("colstack", &[columns]);
    let curved = b.call("mul", &[basis, random]);
    broadcast(b, "add", &[linear, curved])
}

pub(crate) fn parameter_function(
    b: &mut Builder<'_>,
    head: &str,
    body: NodeId,
    parameters: &[String],
) -> NodeId {
    let inputs = parameters
        .iter()
        .map(|name| {
            let name = b.sym(name);
            (
                name,
                Ref {
                    ns: RefNs::SelfMod,
                    name,
                },
            )
        })
        .collect::<Vec<_>>();
    let head = b.sym(head);
    b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: vec![body].into(),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }))
}

fn pointwise_log_likelihood(b: &mut Builder<'_>, recipe: &Recipe, scale: Option<NodeId>) -> NodeId {
    let mut inputs = Vec::new();
    let mut argument = |b: &mut Builder<'_>, name: &str| {
        let value = Ref {
            ns: RefNs::Local,
            name: b.sym(&format!("_{name}_")),
        };
        inputs.push((b.sym(name), value));
        b.m.alloc(Node::Ref(value))
    };
    let mu = if recipe.family == Family::Categorical {
        b.lit_real(0.0)
    } else {
        argument(b, "mu")
    };
    let value = argument(b, "value");
    let trials = match recipe.formula.trials {
        Some(Trials::Constant(n)) => Some(b.lit_int(n)),
        Some(Trials::Column(_)) => Some(argument(b, "trials")),
        None => None,
    };
    let row_scale = scale.map(|_| argument(b, "scale"));
    let auxiliary: Vec<_> = observation_auxiliaries(recipe)
        .into_iter()
        .filter(|name| !(*name == "sigma" && scale.is_some()))
        .map(|name| (name, argument(b, name)))
        .collect();
    let cuts: Vec<_> = (1..=recipe.thresholds())
        .map(|i| argument(b, &format!("cut{i}")))
        .collect();
    let score_inputs: Vec<_> = crate::response::score_columns(&recipe.formula)
        .map(|(name, _)| (name, argument(b, name)))
        .collect();
    let law = recipe.family.observation(
        b,
        crate::family::Observation {
            mu,
            trials,
            scale: row_scale,
            auxiliary: &auxiliary,
            cuts: &cuts,
        },
        true,
    );
    let score = crate::response::score(b, &recipe.formula, law, value, &score_inputs);
    let head = b.sym("functionof");
    let row_score = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([score]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    let mut args = vec![row_score];
    if recipe.family != Family::Categorical {
        args.push(b.self_ref("mu"));
    }
    args.push(b.self_ref("y_data"));
    if matches!(recipe.formula.trials, Some(Trials::Column(_))) {
        args.push(b.self_ref("trials_data"));
    }
    if let Some(scale) = scale {
        args.push(scale);
    }
    args.extend(
        observation_auxiliaries(recipe)
            .into_iter()
            .filter(|name| !(*name == "sigma" && scale.is_some()))
            .map(|name| b.self_ref(name)),
    );
    args.extend(threshold_inputs(b, recipe));
    args.extend(score_inputs.iter().map(|(name, _)| b.self_ref(name)));
    b.call("broadcast", &args)
}

fn observation_auxiliaries(recipe: &Recipe) -> Vec<&str> {
    if recipe.family == Family::Categorical {
        recipe
            .predictors()
            .map(|(name, _)| name)
            .filter(|name| *name != "mu")
            .collect()
    } else {
        recipe.family.auxiliaries().to_vec()
    }
}

fn threshold_inputs(b: &mut Builder<'_>, recipe: &Recipe) -> Vec<NodeId> {
    (1..=recipe.thresholds())
        .map(|i| {
            let cuts = b.self_ref("cuts");
            let index = b.lit_int(i as i64);
            b.call("get", &[cuts, index])
        })
        .collect()
}

pub(crate) fn external_vector(b: &mut Builder<'_>, name: &str, set: &str) {
    let set = b.call_head(set);
    let n = b.self_ref("N");
    let domain = b.call("cartpow", &[set, n]);
    let external = b.call("external", &[domain]);
    b.bind(name, external);
}

fn design_columns(b: &mut Builder<'_>, design: &Design, inputs: &[String]) -> Vec<NodeId> {
    let mut components = std::collections::HashMap::new();
    design
        .columns
        .iter()
        .map(|column| {
            let mut values = column.components.iter().map(|component| {
                *components
                    .entry(component.clone())
                    .or_insert_with(|| design_component(b, design, inputs, component))
            });
            let first = values.next().expect("a design column has a component");
            let values: Vec<_> = values.collect();
            values
                .into_iter()
                .fold(first, |left, right| broadcast(b, "mul", &[left, right]))
        })
        .collect()
}

fn design_component(
    b: &mut Builder<'_>,
    design: &Design,
    inputs: &[String],
    component: &DesignComponent,
) -> NodeId {
    let (predictor, weights, name) = match *component {
        DesignComponent::Numeric { predictor } => {
            return b.self_ref(&inputs[predictor]);
        }
        DesignComponent::Contrast { predictor, column } => {
            let factor = design.predictors[predictor].factor.as_ref().unwrap();
            (
                predictor,
                factor
                    .matrix
                    .iter()
                    .map(|row| row[column])
                    .collect::<Vec<_>>(),
                format!("contrast_{}_{}", predictor + 1, column + 1),
            )
        }
        DesignComponent::Indicator { predictor, level } => {
            let factor = design.predictors[predictor].factor.as_ref().unwrap();
            (
                predictor,
                (0..factor.levels.len())
                    .map(|i| if i == level { 1.0 } else { 0.0 })
                    .collect(),
                format!("indicator_{}_{}", predictor + 1, level + 1),
            )
        }
    };
    let weights: Vec<_> = weights.into_iter().map(|value| b.lit_real(value)).collect();
    let weights = b.array(&weights);
    let indices = b.self_ref(&inputs[predictor]);
    let values = b.call("get", &[weights, indices]);
    let name = b.alloc_name(&name);
    b.bind(&name, values);
    b.self_ref(&name)
}

pub(crate) fn broadcast(b: &mut Builder<'_>, name: &str, args: &[NodeId]) -> NodeId {
    let head = b.call_head(name);
    let args: Vec<_> = std::iter::once(head).chain(args.iter().copied()).collect();
    b.call("broadcast", &args)
}

fn scope(b: &Builder<'_>, family: Family) -> Vec<String> {
    b.m.public_bindings()
        .map(|(_, binding)| b.m.resolve(binding.name).to_owned())
        .chain(["temp_Intercept", "cuts", "b", "bs"].map(str::to_owned))
        .chain(family.auxiliaries().iter().flat_map(|name| {
            [
                (*name).to_owned(),
                predictor_name(name, "temp_Intercept"),
                predictor_name(name, "b"),
                predictor_name(name, "bs"),
            ]
        }))
        .collect()
}

fn parameter(
    b: &mut Builder<'_>,
    name: &str,
    prior: &ResolvedPrior,
    support: Option<NodeId>,
    parameters: &mut Vec<ParameterPrior>,
    family: Family,
) -> Result<()> {
    let names = scope(b, family);
    let bindings: Vec<_> = names.iter().map(String::as_str).collect();
    match priors::emit(b, prior, support, None, &bindings)? {
        PriorNode::Measure(law) => declare_parameter(b, name, law, parameters),
        PriorNode::Constant(value) => b.bind(name, value),
    }
    Ok(())
}

fn lower_bound(b: &mut Builder<'_>, value: f64) -> NodeId {
    let lower = b.lit_real(value);
    let inf = b.call_head("inf");
    b.call("interval", &[lower, inf])
}

fn declare_parameter(
    b: &mut Builder<'_>,
    name: &str,
    law: NodeId,
    parameters: &mut Vec<ParameterPrior>,
) {
    // Classify all priors after their parameter references exist. Only proper
    // measures become draws; the others retain an explicit measure factor.
    let support = crate::prior_model::parameter_support(b, law);
    let value = b.call("elementof", &[support]);
    b.bind(name, value);
    parameters.push(ParameterPrior {
        name: name.into(),
        measure: law,
    });
}

pub(crate) fn inline(mut module: Module, data: &DataDocument) -> Result<Module> {
    for (name, value) in &data.constants {
        let binding = module
            .public_bindings()
            .find(|(_, v)| module.resolve(v.name) == name)
            .map(|(id, _)| id)
            .unwrap();
        let mut b = Builder::new(&mut module);
        let node = match value {
            DataValue::Integer(v) => b.lit_int(*v),
            DataValue::Integers(v) => {
                let values: Vec<_> = v.iter().map(|&v| b.lit_int(v)).collect();
                b.array(&values)
            }
            DataValue::Real32(v) => {
                let values: Vec<_> = v.iter().map(|&v| b.lit_real(f64::from(v))).collect();
                b.array(&values)
            }
            DataValue::Real64(v) => {
                let values: Vec<_> = v.iter().map(|&v| b.lit_real(v)).collect();
                b.array(&values)
            }
        };
        module.set_binding_rhs(binding, node);
    }
    Ok(module)
}

pub(crate) fn with_data_source(
    mut module: Module,
    data: &DataDocument,
    source: &str,
) -> Result<Module> {
    let mut b = Builder::new(&mut module);
    let name = b.alloc_name("formula_data");
    let mut fields = Vec::new();
    for (column, value) in &data.constants {
        let (id, rhs) =
            b.m.public_bindings()
                .find(|(_, v)| b.m.resolve(v.name) == column)
                .map(|(id, v)| (id, v.rhs))
                .ok_or_else(|| Error(format!("missing prepared data binding `{column}`")))?;
        if let DataValue::Integer(value) = value {
            if column != "N" {
                return Err(Error(format!("unexpected scalar data binding `{column}`")));
            }
            let value = b.lit_int(*value);
            b.m.set_binding_rhs(id, value);
            continue;
        }
        let Node::Call(call) = b.m.node(rhs) else {
            return Err(Error(format!("data binding `{column}` is not external")));
        };
        if !matches!(call.head, CallHead::Builtin(head) if b.m.resolve(head) == "external")
            || call.args.len() != 1
        {
            return Err(Error(format!("data binding `{column}` is not external")));
        }
        fields.push((column.as_str(), call.args[0]));
        let data = b.self_ref(&name);
        let field = b.str_lit(column);
        let value = b.call("get", &[data, field]);
        b.m.set_binding_rhs(id, value);
    }
    let domain = b.call_fields("cartprod", &fields);
    let source = b.str_lit(source);
    let data = b.call("load_data", &[source, domain]);
    b.bind(&name, data);
    Ok(module)
}

pub(crate) fn info(
    spec: &Spec,
    recipe: &Recipe,
    data: &DataDocument,
    module: &Module,
) -> Result<ModelInfo> {
    let mut parameters = Vec::new();
    let mut group_offset = 0;
    for (dpar, predictor) in recipe.predictors() {
        let thresholds = if dpar == "mu" { recipe.thresholds() } else { 0 };
        parameters.extend(predictor_info(
            dpar,
            &recipe.predictor_suffix(dpar),
            predictor,
            module,
            thresholds,
            group_offset,
        )?);
        group_offset += predictor.groups.len();
    }
    for &name in recipe.family.auxiliaries() {
        if !recipe.predictors.contains_key(name) {
            parameters.push(Parameter {
                name: name.into(),
                binding: name.into(),
                index: None,
                indices: None,
                role: parameter_role(module, name),
                derived: !sampled(module, name),
            });
        }
    }
    let prior_factors = metadata::prior_factors(module, &parameters);
    let prior_sampling = if prior_factors
        .values()
        .any(|prior| !matches!(prior.mass.as_str(), "normalized" | "deferred"))
    {
        crate::PriorSampling::Unavailable
    } else {
        crate::PriorSampling::NotChecked
    };
    let (factor_levels, group_levels) = recipe
        .predictors
        .get("mu")
        .map_or_else(Default::default, metadata::levels);
    let predictor_metadata = |name: &str, formula: &str| {
        let predictor = &recipe.predictors[name];
        let (factor_levels, group_levels) = metadata::levels(predictor);
        crate::PredictorInfo {
            formula: formula.into(),
            link: predictor.link.name().into(),
            design_columns: predictor
                .design
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect(),
            factor_levels,
            group_levels,
        }
    };
    Ok(ModelInfo {
        formula: spec.formula.clone(),
        family: spec.family.clone(),
        link: recipe.link.name().into(),
        group_collation: "C".into(),
        default_prior_rule: if recipe.nonlinear.is_some() {
            "Nonlinear predictors use explicit population priors and uncentred intercepts. Family auxiliary and group priors use the existing formula defaults.".into()
        } else {
            format!(
                "brms {}({}): numeric identity/log/inverse/sqrt responses use the linked median and MAD, rounded to one decimal (MAD constant 1.4826), with the mean offset subtracted from the intercept location; other responses use location 0 and scale 2.5. Scale is at least 2.5; direct sigma uses the identity-link response scale. Lognormal uses log responses for all these statistics. von_mises(tan_half) instead uses student_t(1, 0, 1) for the intercept. Predicted auxiliaries use brms's link-specific intercept defaults and the mean predictor's response scale for group priors.",
                spec.family,
                recipe.link.name()
            )
        },
        design_columns: recipe
            .predictors
            .get("mu")
            .map_or_else(Vec::new, |predictor| {
                predictor
                    .design
                    .columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect()
            }),
        factor_levels,
        group_levels,
        response_levels: recipe.response_levels.clone(),
        dpars: recipe
            .predictors
            .keys()
            .filter(|name| name.as_str() != "mu" && !spec.nlpars.contains_key(*name))
            .map(|name| {
                let formula = spec
                    .dpars
                    .get(name)
                    .map_or(&spec.formula, |dpar| &dpar.formula);
                (name.clone(), predictor_metadata(name, formula))
            })
            .collect(),
        nlpars: spec
            .nlpars
            .iter()
            .map(|(name, formula)| (name.clone(), predictor_metadata(name, formula)))
            .collect(),
        parameters,
        prior_factors,
        prior_sampling,
        has_improper_prior: module.public_bindings().any(|(_, binding)| {
            module.resolve(binding.name) == "Pi_prior"
                && !matches!(
                    module.type_of(binding.rhs),
                    Some(flatppl_core::Type::Measure {
                        mass: flatppl_core::Mass::Normalized,
                        ..
                    })
                )
        }),
        training_rows: data.rows.clone(),
        score_offset: (1..=recipe.thresholds()).map(|i| (i as f64).ln()).sum(),
    })
}

fn predictor_info(
    name: &str,
    dpar: &str,
    recipe: &crate::PredictorRecipe,
    module: &Module,
    thresholds: usize,
    group_offset: usize,
) -> Result<Vec<Parameter>> {
    let prefix = if name == "mu" {
        String::new()
    } else {
        format!("{name}_")
    };
    let mut parameters = Vec::new();
    let mut add = |name: String, binding: &str, indices: &[usize], role| {
        parameters.push(Parameter {
            name,
            binding: binding.into(),
            index: (indices.len() == 1).then(|| indices[0]),
            indices: (indices.len() > 1).then(|| indices.to_vec()),
            role,
            derived: role != ParameterRole::Sampled,
        })
    };
    if thresholds > 0 {
        for i in 1..=thresholds {
            add(
                format!("Intercept[{i}]"),
                "cuts",
                &[i],
                ParameterRole::Sampled,
            );
            add(
                format!("b_Intercept[{i}]"),
                &predictor_name(dpar, "b_Intercept"),
                &[i],
                ParameterRole::Derived,
            );
        }
    } else if recipe.formula.intercept {
        add(
            predictor_name(name, "Intercept"),
            &predictor_name(dpar, "temp_Intercept"),
            &[],
            parameter_role(module, &predictor_name(dpar, "temp_Intercept")),
        );
        add(
            format!("b_{prefix}Intercept"),
            &predictor_name(dpar, "b_Intercept"),
            &[],
            ParameterRole::Derived,
        );
    }
    let mut names = std::collections::HashSet::new();
    if recipe.formula.intercept {
        names.insert("Intercept".into());
    }
    for (i, column) in recipe.design.columns.iter().enumerate() {
        let name = coefficient_name(&column.name);
        if !names.insert(name.clone()) {
            return Err(Error(format!(
                "design columns have duplicate brms name `{name}`"
            )));
        }
        let component = predictor_name(dpar, &format!("b_{}", i + 1));
        if sampled(module, &component) {
            add(
                format!("b_{prefix}{name}"),
                &component,
                &[],
                ParameterRole::Sampled,
            );
        } else {
            add(
                format!("b_{prefix}{name}"),
                &predictor_name(dpar, "b"),
                &[i + 1],
                parameter_role(module, &predictor_name(dpar, "b")),
            );
        }
    }
    let smooth_curves =
        recipe
            .smooths
            .iter()
            .zip(&recipe.formula.smooths)
            .flat_map(|(smooth, term)| {
                smooth.curves().map(move |(level, sds, z)| {
                    (term.coefficient(level), smooth.inputs.len(), sds, z)
                })
            });
    for (i, (name, inputs, sds, z)) in smooth_curves.enumerate() {
        if !names.insert(name.clone()) {
            return Err(Error(format!(
                "design columns have duplicate brms name `{name}`"
            )));
        }
        let component = predictor_name(dpar, &format!("bs_{}", i + 1));
        let binding = predictor_name(dpar, "bs");
        if sampled(module, &component) {
            add(
                format!("b_{prefix}{name}"),
                &component,
                &[],
                ParameterRole::Sampled,
            );
        } else {
            add(
                format!("b_{prefix}{name}"),
                &binding,
                &[i + 1],
                parameter_role(module, &binding),
            );
        }
        add(
            format!("sds_{prefix}{name}"),
            sds,
            &[],
            parameter_role(module, sds),
        );
        for j in 1..inputs {
            add(
                format!("zs_{prefix}{name}[{j}]"),
                z,
                &[j],
                ParameterRole::Sampled,
            );
        }
    }
    for (i, (coding, column)) in recipe
        .monotonic
        .iter()
        .zip(&recipe.formula.monotonic)
        .enumerate()
    {
        let name = crate::monotonic::coefficient(column);
        if !names.insert(name.clone()) {
            return Err(Error(format!(
                "design columns have duplicate brms name `{name}`"
            )));
        }
        let index = i + 1;
        let slope = predictor_name(dpar, &format!("bmo_{index}"));
        add(
            format!("b_{prefix}{name}"),
            &slope,
            &[],
            parameter_role(module, &slope),
        );
        let simplex = predictor_name(dpar, &format!("simo_{index}"));
        for j in 1..=coding.steps() as usize {
            add(
                format!("simo_{prefix}{name}1[{j}]"),
                &simplex,
                &[j],
                parameter_role(module, &simplex),
            );
        }
    }
    for (group_index, group) in recipe.groups.iter().enumerate() {
        let k = group.coefficients.len();
        for (j, coefficient) in group.coefficients.iter().enumerate() {
            let name = format!("sd_{}__{prefix}{coefficient}", group.column);
            if let Some(component) = sampled_component(module, &group.sd, j) {
                add(name, component, &[], ParameterRole::Sampled);
            } else {
                let indices = [j + 1];
                add(
                    name,
                    &group.sd,
                    if k == 1 { &[] } else { &indices },
                    parameter_role(module, &group.sd),
                );
            }
        }
        for (i, level) in group.levels.iter().enumerate() {
            for (j, coefficient) in group.coefficients.iter().enumerate() {
                let vector_index = [i + 1];
                let z_index = [j + 1, i + 1];
                let r_index = [i + 1, j + 1];
                add(
                    format!("z_{}[{},{}]", group_offset + group_index + 1, j + 1, i + 1),
                    &group.z,
                    if k == 1 { &vector_index } else { &z_index },
                    ParameterRole::Sampled,
                );
                add(
                    format!("r_{}[{level},{prefix}{coefficient}]", group.column),
                    &group.r,
                    if k == 1 { &vector_index } else { &r_index },
                    ParameterRole::Derived,
                );
            }
        }
        if let Some(cholesky) = &group.correlation {
            for i in 0..k {
                for j in 0..=i {
                    add(
                        format!("L_{}[{},{}]", group_offset + group_index + 1, i + 1, j + 1),
                        cholesky,
                        &[i + 1, j + 1],
                        ParameterRole::Sampled,
                    );
                }
            }
        }
        if let Some(correlation) = &group.correlation_matrix {
            for i in 0..k {
                for j in i + 1..k {
                    add(
                        format!(
                            "cor_{}__{prefix}{}__{prefix}{}",
                            group.column, group.coefficients[i], group.coefficients[j]
                        ),
                        correlation,
                        &[i + 1, j + 1],
                        ParameterRole::Derived,
                    );
                }
            }
        }
    }
    Ok(parameters)
}

fn sampled(module: &Module, name: &str) -> bool {
    module.public_bindings().find(|(_, binding)| module.resolve(binding.name) == name).is_some_and(|(_, binding)| matches!(module.node(binding.rhs), Node::Call(call) if matches!(call.head, CallHead::Builtin(s) if matches!(module.resolve(s), "draw" | "elementof"))))
}

fn parameter_role(module: &Module, name: &str) -> ParameterRole {
    if sampled(module, name) {
        ParameterRole::Sampled
    } else {
        ParameterRole::Fixed
    }
}

fn sampled_component<'a>(module: &'a Module, name: &str, index: usize) -> Option<&'a str> {
    let (_, binding) = module
        .public_bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)?;
    let Node::Call(call) = module.node(binding.rhs) else {
        return None;
    };
    if !matches!(call.head, CallHead::Builtin(head) if module.resolve(head) == "vector") {
        return None;
    }
    let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = module.node(*call.args.get(index)?)
    else {
        return None;
    };
    let name = module.resolve(*name);
    sampled(module, name).then_some(name)
}

pub(crate) fn coefficient_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        let replacement = match ch {
            ' ' | '(' | ')' | '[' | ']' | ',' | '"' | '\'' | '?' => "",
            '+' => "P",
            '-' => "M",
            '*' => "MU",
            '/' => "D",
            '^' => "E",
            '=' => "EQ",
            '$' => "USD",
            _ => {
                out.push(ch);
                continue;
            }
        };
        out.push_str(replacement);
    }
    out
}
