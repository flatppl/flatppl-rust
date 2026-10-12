use crate::{
    DataDocument, DataValue, Design, DesignComponent, Error, Family, ModelInfo, Parameter, Recipe,
    ResolvedPrior, Result, Spec, Trials,
    priors::{self, PriorNode},
};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs};

pub(crate) fn model(spec: &Spec, recipe: &mut Recipe) -> Result<Module> {
    let mut module = Module::new();
    let mut b = Builder::new(&mut module);
    b.stamp_compat();
    let name = b.str_lit("regression");
    let version = b.str_lit("0.1");
    let reg = b.call("standard_module", &[name, version]);
    b.bind("reg", reg);
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
    let predictor_names: Vec<_> = recipe
        .predictors()
        .map(|(name, _)| name.to_owned())
        .collect();
    for name in &predictor_names {
        predictor_data(&mut b, name, recipe.predictors.get_mut(name).unwrap());
    }
    let mut parameters = Vec::new();
    let thresholds = recipe.thresholds();
    for name in &predictor_names {
        predictor_parameters(
            &mut b,
            name,
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
        let support = if name == "zi" {
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
    let fields: Vec<_> = parameters
        .iter()
        .map(|name| (name.as_str(), b.self_ref(name)))
        .collect();
    let record = b.call_fields("record", &fields);
    let prior = b.call("lawof", &[record]);
    b.bind("Pi_prior", prior);

    for (name, predictor) in recipe.predictors() {
        predictor_values(
            &mut b,
            name,
            predictor,
            if name == "mu" { thresholds } else { 0 },
        );
    }
    let mu = b.self_ref("mu");
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
    let auxiliary: Vec<_> = recipe
        .predictors
        .keys()
        .filter(|&name| name != "mu")
        .map(|name| (name.as_str(), b.self_ref(name)))
        .collect();
    let law = recipe.family.observation(
        &mut b,
        crate::family::Observation {
            mu,
            trials,
            scale,
            auxiliary: &auxiliary,
        },
        thresholds,
        false,
    );
    let y = b.call("draw", &[law]);
    b.bind("y", y);
    let y = b.self_ref("y");
    let response = b.call_fields("record", &[("y", y)]);
    let kernel = parameter_function(&mut b, "kernelof", response, &parameters);
    b.bind("K_model", kernel);
    let y = b.self_ref("y_data");
    let observation = b.call_fields("record", &[("y", y)]);
    let kernel = b.self_ref("K_model");
    let likelihood = b.call("likelihoodof", &[kernel, observation]);
    b.bind("L", likelihood);
    let pointwise = pointwise_log_likelihood(&mut b, recipe, scale);
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
    let formula = b.str_lit(&spec.formula);
    let family = b.str_lit(&spec.family);
    let link = b.str_lit(recipe.link.name());
    let improper = b.lit_bool(false);
    let score_offset = b.lit_real((1..=thresholds).map(|i| (i as f64).ln()).sum());
    let mut fields = vec![
        ("formula", formula),
        ("family", family),
        ("link", link),
        ("has_improper_prior", improper),
        ("score_offset", score_offset),
    ];
    if !spec.dpars.is_empty() {
        let dpars: Vec<_> = recipe
            .predictors
            .iter()
            .filter(|(name, _)| name.as_str() != "mu")
            .map(|(name, predictor)| {
                let formula = b.str_lit(&spec.dpars[name].formula);
                let link = b.str_lit(predictor.link.name());
                (
                    name.as_str(),
                    b.call_fields("record", &[("formula", formula), ("link", link)]),
                )
            })
            .collect();
        fields.push(("dpars", b.call_fields("record", &dpars)));
    }
    let metadata = b.call_fields("record", &fields);
    b.bind("formula_model", metadata);
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
    parameters: &mut Vec<String>,
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
        draw(b, "cuts", prior, parameters);
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
    if !recipe.design.columns.is_empty() {
        if recipe.priors.b.iter().all(|p| *p == recipe.priors.b[0]) {
            let scope = scope(b, family);
            let bindings: Vec<_> = scope.iter().map(String::as_str).collect();
            let prior = priors::emit(b, &recipe.priors.b[0], None, None, &bindings)?;
            let count = b.lit_int(recipe.design.columns.len() as i64);
            match prior {
                PriorNode::Measure(prior) => {
                    let iid = b.call("iid", &[prior, count]);
                    draw(b, &predictor_name(dpar, "b"), iid, parameters);
                }
                PriorNode::Constant(value) => {
                    let values = b.call("fill", &[value, count]);
                    b.bind(&predictor_name(dpar, "b"), values);
                }
            }
        } else {
            let mut values = Vec::new();
            for (i, prior) in recipe.priors.b.iter().enumerate() {
                let name = predictor_name(dpar, &format!("b_{}", i + 1));
                parameter(b, &name, prior, None, parameters, family)?;
                values.push(b.self_ref(&name));
            }
            let coefficients = b.array(&values);
            b.bind(&predictor_name(dpar, "b"), coefficients);
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
                    draw(b, &group.sd, law, parameters);
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
        let zero = b.lit_real(0.0);
        let one = b.lit_real(1.0);
        let standard = b.call("Normal", &[zero, one]);
        let count = b.lit_int(group.levels.len() as i64);
        let size = if dimensions == 1 {
            count
        } else {
            b.array(&[dimension, count])
        };
        let iid = b.call("iid", &[standard, size]);
        draw(b, &group.z, iid, parameters);
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
                draw(b, &name, law, parameters);
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

fn predictor_values(
    b: &mut Builder<'_>,
    dpar: &str,
    recipe: &crate::PredictorRecipe,
    thresholds: usize,
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
    let eta = b.self_ref(&format!("eta_{dpar}"));
    let value = if thresholds > 0 {
        eta
    } else {
        recipe.link.inverse(b, eta)
    };
    b.bind(dpar, value);
    if let Some(intercept) = intercept {
        b.bind(&predictor_name(dpar, "b_Intercept"), intercept);
    }
}

fn parameter_function(
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
    let mu = argument(b, "mu");
    let value = argument(b, "value");
    let trials = match recipe.formula.trials {
        Some(Trials::Constant(n)) => Some(b.lit_int(n)),
        Some(Trials::Column(_)) => Some(argument(b, "trials")),
        None => None,
    };
    let row_scale = scale.map(|_| argument(b, "scale"));
    let auxiliary: Vec<_> = recipe
        .predictors
        .keys()
        .filter(|&name| name != "mu")
        .map(|name| (name.as_str(), argument(b, name)))
        .collect();
    let law = recipe.family.observation(
        b,
        crate::family::Observation {
            mu,
            trials,
            scale: row_scale,
            auxiliary: &auxiliary,
        },
        recipe.thresholds(),
        true,
    );
    let score = b.call("logdensityof", &[law, value]);
    let head = b.sym("functionof");
    let row_score = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([score]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    let mut args = vec![row_score, b.self_ref("mu"), b.self_ref("y_data")];
    if matches!(recipe.formula.trials, Some(Trials::Column(_))) {
        args.push(b.self_ref("trials_data"));
    }
    if let Some(scale) = scale {
        args.push(scale);
    }
    args.extend(
        recipe
            .predictors
            .keys()
            .filter(|&name| name != "mu")
            .map(|name| b.self_ref(name)),
    );
    b.call("broadcast", &args)
}

fn external_vector(b: &mut Builder<'_>, name: &str, set: &str) {
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
        .chain(["temp_Intercept", "cuts", "b"].map(str::to_owned))
        .chain(family.auxiliaries().iter().flat_map(|name| {
            [
                (*name).to_owned(),
                predictor_name(name, "temp_Intercept"),
                predictor_name(name, "b"),
            ]
        }))
        .collect()
}

fn parameter(
    b: &mut Builder<'_>,
    name: &str,
    prior: &ResolvedPrior,
    support: Option<NodeId>,
    parameters: &mut Vec<String>,
    family: Family,
) -> Result<()> {
    let names = scope(b, family);
    let bindings: Vec<_> = names.iter().map(String::as_str).collect();
    match priors::emit(b, prior, support, None, &bindings)? {
        PriorNode::Measure(law) => draw(b, name, law, parameters),
        PriorNode::Constant(value) => b.bind(name, value),
    }
    Ok(())
}

fn lower_bound(b: &mut Builder<'_>, value: f64) -> NodeId {
    let lower = b.lit_real(value);
    let inf = b.call_head("inf");
    b.call("interval", &[lower, inf])
}

fn draw(b: &mut Builder<'_>, name: &str, law: NodeId, parameters: &mut Vec<String>) {
    let value = b.call("draw", &[law]);
    b.bind(name, value);
    parameters.push(name.into());
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
                derived: !sampled(module, name),
            });
        }
    }
    Ok(ModelInfo {
        formula: spec.formula.clone(),
        family: spec.family.clone(),
        link: recipe.link.name().into(),
        group_collation: "C".into(),
        default_prior_rule: format!(
            "brms {}({}): numeric identity/log/inverse/sqrt responses use the linked median and MAD, rounded to one decimal (MAD constant 1.4826), with the mean offset subtracted from the intercept location; other responses use location 0 and scale 2.5. Scale is at least 2.5; direct sigma uses the identity-link response scale. Lognormal uses log responses for all these statistics. von_mises(tan_half) instead uses student_t(1, 0, 1) for the intercept. Predicted auxiliaries use brms's link-specific intercept defaults and the mean predictor's response scale for group priors.",
            spec.family,
            recipe.link.name()
        ),
        design_columns: recipe.predictors["mu"]
            .design
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        dpars: recipe
            .predictors
            .iter()
            .filter(|(name, _)| name.as_str() != "mu")
            .map(|(name, predictor)| {
                (
                    name.clone(),
                    crate::PredictorInfo {
                        formula: spec.dpars[name].formula.clone(),
                        link: predictor.link.name().into(),
                        design_columns: predictor
                            .design
                            .columns
                            .iter()
                            .map(|column| column.name.clone())
                            .collect(),
                    },
                )
            })
            .collect(),
        parameters,
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
    dpar: &str,
    recipe: &crate::PredictorRecipe,
    module: &Module,
    thresholds: usize,
    group_offset: usize,
) -> Result<Vec<Parameter>> {
    let prefix = if dpar == "mu" {
        String::new()
    } else {
        format!("{dpar}_")
    };
    let mut parameters = Vec::new();
    let mut add = |name: String, binding: &str, indices: &[usize], derived| {
        parameters.push(Parameter {
            name,
            binding: binding.into(),
            index: (indices.len() == 1).then(|| indices[0]),
            indices: (indices.len() > 1).then(|| indices.to_vec()),
            derived,
        })
    };
    if thresholds > 0 {
        for i in 1..=thresholds {
            add(format!("Intercept[{i}]"), "cuts", &[i], false);
            add(
                format!("b_Intercept[{i}]"),
                &predictor_name(dpar, "b_Intercept"),
                &[i],
                true,
            );
        }
    } else if recipe.formula.intercept {
        add(
            predictor_name(dpar, "Intercept"),
            &predictor_name(dpar, "temp_Intercept"),
            &[],
            !sampled(module, &predictor_name(dpar, "temp_Intercept")),
        );
        add(
            format!("b_{prefix}Intercept"),
            &predictor_name(dpar, "b_Intercept"),
            &[],
            true,
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
            add(format!("b_{prefix}{name}"), &component, &[], false);
        } else {
            add(
                format!("b_{prefix}{name}"),
                &predictor_name(dpar, "b"),
                &[i + 1],
                !sampled(module, &predictor_name(dpar, "b")),
            );
        }
    }
    for (group_index, group) in recipe.groups.iter().enumerate() {
        let k = group.coefficients.len();
        for (j, coefficient) in group.coefficients.iter().enumerate() {
            let name = format!("sd_{}__{prefix}{coefficient}", group.column);
            if let Some(component) = sampled_component(module, &group.sd, j) {
                add(name, component, &[], false);
            } else {
                let indices = [j + 1];
                add(
                    name,
                    &group.sd,
                    if k == 1 { &[] } else { &indices },
                    !sampled(module, &group.sd),
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
                    false,
                );
                add(
                    format!("r_{}[{level},{prefix}{coefficient}]", group.column),
                    &group.r,
                    if k == 1 { &vector_index } else { &r_index },
                    true,
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
                        false,
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
                        true,
                    );
                }
            }
        }
    }
    Ok(parameters)
}

fn sampled(module: &Module, name: &str) -> bool {
    module.public_bindings().find(|(_, binding)| module.resolve(binding.name) == name).is_some_and(|(_, binding)| matches!(module.node(binding.rhs), Node::Call(call) if matches!(call.head, CallHead::Builtin(s) if module.resolve(s) == "draw")))
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

pub(crate) fn mark_improper(module: &mut Module, improper: bool) {
    let (id, binding) = module
        .public_bindings()
        .find(|(_, binding)| module.resolve(binding.name) == "formula_model")
        .unwrap();
    let Node::Call(mut call) = module.node(binding.rhs).clone() else {
        unreachable!()
    };
    let value = module.alloc(Node::Lit(flatppl_core::Scalar::Bool(improper)));
    for field in &mut call.named {
        if module.resolve(field.name) == "has_improper_prior" {
            field.value = value;
        }
    }
    let metadata = module.alloc(Node::Call(call));
    module.set_binding_rhs(id, metadata);
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
