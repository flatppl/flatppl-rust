use crate::{
    DataDocument, DataValue, DesignComponent, Error, ModelInfo, NormalPrior, Parameter, Recipe,
    Result, Spec,
};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs};

const INTERCEPT_RULE: &str =
    "brms gaussian(identity): centred intercept location = round(median(y), 1).";
const SCALE_RULE: &str =
    "brms gaussian(identity): scale = max(2.5, round(mad(y), 1)). R MAD constant = 1.4826.";

pub(crate) fn model(spec: &Spec, recipe: &mut Recipe) -> Result<Module> {
    let mut module = Module::new();
    let mut b = Builder::new(&mut module);
    b.stamp_compat();
    let improper = !recipe.design.columns.is_empty() && recipe.priors.b.is_none();
    if improper {
        let name = b.str_lit("regression");
        let version = b.str_lit("0.1");
        let reg = b.call("standard_module", &[name, version]);
        b.bind("reg", reg);
    }
    let positive = b.call_head("posintegers");
    let size = b.call("external", &[positive]);
    b.bind("N", size);
    external_vector(&mut b, "y_data", "reals");
    for (i, predictor) in recipe.design.predictors.iter().enumerate() {
        let set = if predictor.factor.is_some() {
            "posintegers"
        } else {
            "reals"
        };
        external_vector(&mut b, &format!("x_{}", i + 1), set);
    }
    for group in &mut recipe.groups {
        group.index = b.alloc_name(&format!("index_{}", group.column));
        external_vector(&mut b, &group.index, "posintegers");
    }
    let mut parameters = Vec::new();
    if recipe.formula.intercept {
        let prior = prior(
            &mut b,
            recipe.priors.intercept,
            recipe.priors.intercept_location,
            recipe.priors.scale,
            false,
        );
        let docs: &[&str] = if recipe.priors.intercept.is_none() {
            &[INTERCEPT_RULE, SCALE_RULE]
        } else {
            &[]
        };
        draw(&mut b, "temp_Intercept", prior, &mut parameters, docs);
    }
    if !recipe.design.columns.is_empty() {
        let prior = if let Some(normal) = recipe.priors.b {
            prior(&mut b, Some(normal), 0.0, 1.0, false)
        } else {
            b.module_user_call("reg", "Flat", &[])
        };
        let count = b.lit_int(recipe.design.columns.len() as i64);
        let iid = b.call("iid", &[prior, count]);
        draw(&mut b, "b", iid, &mut parameters, &[]);
    }
    for (i, group) in recipe.groups.iter_mut().enumerate() {
        group.sd = b.alloc_name(&format!("sd_{}", group.column));
        let law = prior(&mut b, recipe.priors.sd[i], 0.0, recipe.priors.scale, true);
        let docs: &[&str] = if recipe.priors.sd[i].is_none() {
            &[SCALE_RULE]
        } else {
            &[]
        };
        draw(&mut b, &group.sd, law, &mut parameters, docs);
        group.z = b.alloc_name(&format!("z_{}", group.column));
        let zero = b.lit_real(0.0);
        let one = b.lit_real(1.0);
        let standard = b.call("Normal", &[zero, one]);
        let count = b.lit_int(group.levels.len() as i64);
        let iid = b.call("iid", &[standard, count]);
        draw(&mut b, &group.z, iid, &mut parameters, &[]);
        group.r = b.alloc_name(&format!("r_{}", group.column));
        let sd = b.self_ref(&group.sd);
        let z = b.self_ref(&group.z);
        let effect = broadcast(&mut b, "mul", &[sd, z]);
        b.bind(&group.r, effect);
    }
    let law = prior(&mut b, recipe.priors.sigma, 0.0, recipe.priors.scale, true);
    let docs: &[&str] = if recipe.priors.sigma.is_none() {
        &[SCALE_RULE]
    } else {
        &[]
    };
    draw(&mut b, "sigma", law, &mut parameters, docs);
    let fields: Vec<_> = parameters
        .iter()
        .map(|name| (name.as_str(), b.self_ref(name)))
        .collect();
    let record = b.call_fields("record", &fields);
    let prior = b.call("lawof", &[record]);
    b.bind("Pi_prior", prior);

    let n = b.self_ref("N");
    let zero = b.lit_real(0.0);
    let mut eta = b.call("fill", &[zero, n]);
    let mut intercept = recipe
        .formula
        .intercept
        .then(|| b.self_ref("temp_Intercept"));
    if !recipe.design.columns.is_empty() {
        let mut columns = Vec::new();
        let mut components = std::collections::HashMap::new();
        for (column, &mean) in recipe.design.columns.iter().zip(&recipe.means) {
            let mut values = column.components.iter().map(|component| {
                *components
                    .entry(component.clone())
                    .or_insert_with(|| design_component(&mut b, recipe, component))
            });
            let first = values.next().expect("a design column has a component");
            let values: Vec<_> = values.collect();
            let column = values.into_iter().fold(first, |left, right| {
                broadcast(&mut b, "mul", &[left, right])
            });
            columns.push(if recipe.center {
                let mean = b.lit_real(mean);
                broadcast(&mut b, "sub", &[column, mean])
            } else {
                column
            });
        }
        let columns = b.array(&columns);
        let design = b.call("colstack", &[columns]);
        b.bind("X", design);
        let x = b.self_ref("X");
        let coefficients = b.self_ref("b");
        eta = b.call("mul", &[x, coefficients]);
        if recipe.center {
            let means: Vec<_> = recipe.means.iter().map(|&v| b.lit_real(v)).collect();
            let means = b.array(&means);
            let products = broadcast(&mut b, "mul", &[means, coefficients]);
            let dot = b.call("sum", &[products]);
            intercept = intercept.map(|value| b.call("sub", &[value, dot]));
        }
    }
    if recipe.formula.intercept {
        let value = b.self_ref("temp_Intercept");
        eta = broadcast(&mut b, "add", &[eta, value]);
    }
    for group in &recipe.groups {
        let effect = b.self_ref(&group.r);
        let index = b.self_ref(&group.index);
        let selected = b.call("get", &[effect, index]);
        eta = broadcast(&mut b, "add", &[eta, selected]);
    }
    b.bind("eta_mu", eta);
    let eta = b.self_ref("eta_mu");
    b.bind("mu", eta);
    if let Some(intercept) = intercept {
        b.bind("b_Intercept", intercept);
    }
    let mu = b.self_ref("mu");
    let sigma = b.self_ref("sigma");
    let law = broadcast(&mut b, "Normal", &[mu, sigma]);
    let y = b.call("draw", &[law]);
    b.bind("y", y);
    let y = b.self_ref("y");
    let response = b.call_fields("record", &[("y", y)]);
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
    let head = b.sym("kernelof");
    let kernel = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: vec![response].into(),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    b.bind("K_model", kernel);
    let y = b.self_ref("y_data");
    let observation = b.call_fields("record", &[("y", y)]);
    let kernel = b.self_ref("K_model");
    let likelihood = b.call("likelihoodof", &[kernel, observation]);
    b.bind("L", likelihood);
    let likelihood = b.self_ref("L");
    let prior = b.self_ref("Pi_prior");
    let posterior = b.call("bayesupdate", &[likelihood, prior]);
    b.bind("Pi_post", posterior);
    let formula = b.str_lit(&spec.formula);
    let family = b.str_lit(&spec.family);
    let improper = b.lit_bool(improper);
    let metadata = b.call_fields(
        "record",
        &[
            ("formula", formula),
            ("family", family),
            ("has_improper_prior", improper),
        ],
    );
    b.bind("formula_model", metadata);
    Ok(module)
}

fn external_vector(b: &mut Builder<'_>, name: &str, set: &str) {
    let set = b.call_head(set);
    let n = b.self_ref("N");
    let domain = b.call("cartpow", &[set, n]);
    let external = b.call("external", &[domain]);
    b.bind(name, external);
}

fn design_component(b: &mut Builder<'_>, recipe: &Recipe, component: &DesignComponent) -> NodeId {
    let (predictor, weights, name) = match *component {
        DesignComponent::Numeric { predictor } => {
            return b.self_ref(&format!("x_{}", predictor + 1));
        }
        DesignComponent::Contrast { predictor, column } => {
            let factor = recipe.design.predictors[predictor].factor.as_ref().unwrap();
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
            let factor = recipe.design.predictors[predictor].factor.as_ref().unwrap();
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
    let indices = b.self_ref(&format!("x_{}", predictor + 1));
    let values = b.call("get", &[weights, indices]);
    let name = b.alloc_name(&name);
    b.bind(&name, values);
    b.self_ref(&name)
}

fn broadcast(b: &mut Builder<'_>, name: &str, args: &[NodeId]) -> NodeId {
    let head = b.call_head(name);
    let args: Vec<_> = std::iter::once(head).chain(args.iter().copied()).collect();
    b.call("broadcast", &args)
}

fn prior(
    b: &mut Builder<'_>,
    normal: Option<NormalPrior>,
    location: f64,
    scale: f64,
    positive: bool,
) -> NodeId {
    let measure = if let Some(normal) = normal {
        let location = b.lit_real(normal.mean);
        let scale = b.lit_real(normal.sd);
        b.call("Normal", &[location, scale])
    } else {
        let three = b.lit_real(3.0);
        let student = b.call("StudentT", &[three]);
        // Default scale priors are centred at zero. Truncate their unit law
        // before the positive scaling so normalization sees a leaf distribution.
        let student = if positive {
            positive_part(b, student)
        } else {
            student
        };
        let location = b.lit_real(location);
        let scale = b.lit_real(scale);
        b.call("locscale", &[student, location, scale])
    };
    if positive && normal.is_some() {
        positive_part(b, measure)
    } else {
        measure
    }
}

fn positive_part(b: &mut Builder<'_>, measure: NodeId) -> NodeId {
    let zero = b.lit_real(0.0);
    let inf = b.call_head("inf");
    let domain = b.call("interval", &[zero, inf]);
    let truncated = b.call("truncate", &[measure, domain]);
    b.call("normalize", &[truncated])
}

fn draw(b: &mut Builder<'_>, name: &str, law: NodeId, parameters: &mut Vec<String>, docs: &[&str]) {
    let value = b.call("draw", &[law]);
    if docs.is_empty() {
        b.bind(name, value);
    } else {
        b.bind_doc(name, value, docs);
    }
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

pub(crate) fn info(spec: &Spec, recipe: &Recipe, data: &DataDocument) -> Result<ModelInfo> {
    let mut parameters = Vec::new();
    let mut add = |name: String, binding: &str, index: Option<usize>, derived| {
        parameters.push(Parameter {
            name,
            binding: binding.into(),
            index,
            derived,
        })
    };
    if recipe.formula.intercept {
        add("Intercept".into(), "temp_Intercept", None, false);
        add("b_Intercept".into(), "b_Intercept", None, true);
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
        add(format!("b_{name}"), "b", Some(i + 1), false);
    }
    for (group_index, group) in recipe.groups.iter().enumerate() {
        add(
            format!("sd_{}__Intercept", group.column),
            &group.sd,
            None,
            false,
        );
        for (i, level) in group.levels.iter().enumerate() {
            add(
                format!("z_{}[1,{}]", group_index + 1, i + 1),
                &group.z,
                Some(i + 1),
                false,
            );
            let label = match level {
                crate::GroupLevel::Integer(value) => value.to_string(),
                crate::GroupLevel::Real(value) => value.to_string(),
                crate::GroupLevel::Text(value) => value.clone(),
            };
            add(
                format!("r_{}[{label},Intercept]", group.column),
                &group.r,
                Some(i + 1),
                true,
            );
        }
    }
    add("sigma".into(), "sigma", None, false);
    Ok(ModelInfo {
        formula: spec.formula.clone(),
        family: spec.family.clone(),
        group_collation: "C".into(),
        default_prior_rule: format!("{INTERCEPT_RULE} {SCALE_RULE}"),
        design_columns: recipe
            .design
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        parameters,
        has_improper_prior: !recipe.design.columns.is_empty() && recipe.priors.b.is_none(),
        training_rows: data.rows.clone(),
        score_offset: 0.0,
    })
}

fn coefficient_name(name: &str) -> String {
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
