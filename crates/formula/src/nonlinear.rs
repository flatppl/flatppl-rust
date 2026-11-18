use crate::{Arithmetic, ColumnExpression, Error, Family, Formula, Result, Spec};
use flatppl_convert::Builder;
use flatppl_core::NodeId;
use serde::{Deserialize, Serialize};

/// The response expression and inputs retained by the fitted recipe.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NonlinearPredictor {
    pub expression: ColumnExpression,
    pub parameters: Vec<String>,
    pub columns: Vec<String>,
}

pub(crate) fn parse(spec: &Spec, family: Family) -> Result<(Formula, Option<NonlinearPredictor>)> {
    if spec.nlpars.is_empty() {
        return Ok((crate::parse_formula(&spec.formula)?, None));
    }
    if family == Family::Categorical {
        return Err(Error(format!(
            "nonlinear predictors do not support family `{}`",
            spec.family
        )));
    }
    let (lhs, rhs) = spec
        .formula
        .split_once('~')
        .ok_or_else(|| Error("nonlinear formula must contain `response ~ expression`".into()))?;
    // Ordinal threshold intercepts remain part of the response model.
    let intercept = u8::from(family == Family::Cumulative);
    let formula = crate::parse_formula(&format!("{lhs} ~ {intercept}"))?;
    let expression = crate::formula::parse_numeric_expression(rhs)?;
    let mut names = Vec::new();
    expression.columns(&mut names);
    if names.contains(&formula.response.as_str()) {
        return Err(Error(
            "the response cannot appear in the nonlinear predictor".into(),
        ));
    }
    for name in spec.nlpars.keys() {
        flatppl_convert::check_binding_name(name, "nonlinear parameter").map_err(Error)?;
        if name == "mu" || family.auxiliaries().contains(&name.as_str()) {
            return Err(Error(format!("invalid nonlinear parameter name `{name}`")));
        }
        if !names.contains(&name.as_str()) {
            return Err(Error(format!("nonlinear parameter `{name}` is unused")));
        }
    }
    let columns = names
        .into_iter()
        .filter(|name| !spec.nlpars.contains_key(*name))
        .map(str::to_owned)
        .collect();
    let parameters = spec.nlpars.keys().cloned().collect();
    Ok((
        formula,
        Some(NonlinearPredictor {
            expression,
            parameters,
            columns,
        }),
    ))
}

pub(crate) fn data_name(i: usize) -> String {
    format!("nl_data_{}", i + 1)
}

// Keep parameter suffixes distinct from generated role names such as
// `Intercept_a` and from ordinary family parameters.
pub(crate) fn parameter_suffix(name: &str) -> String {
    format!("nl_{name}")
}

impl NonlinearPredictor {
    pub(crate) fn emit(&self, b: &mut Builder<'_>) -> NodeId {
        emit_expression(b, &self.expression, &|name| {
            self.columns
                .iter()
                .position(|column| column == name)
                .map_or_else(|| format!("eta_{}", parameter_suffix(name)), data_name)
        })
    }
}

fn emit_expression(
    b: &mut Builder<'_>,
    expression: &ColumnExpression,
    binding: &impl Fn(&str) -> String,
) -> NodeId {
    use ColumnExpression as Expr;
    match expression {
        Expr::Column(name) => b.self_ref(&binding(name)),
        Expr::Intercept => b.lit_real(1.0),
        Expr::Number(value) => b.lit_real(*value),
        Expr::Identity(x) | Expr::Positive(x) | Expr::Parentheses(x) => {
            emit_expression(b, x, binding)
        }
        Expr::Log(x) | Expr::Exp(x) | Expr::Negate(x) => {
            let op = match expression {
                Expr::Log(_) => "log",
                Expr::Exp(_) => "exp",
                _ => "neg",
            };
            let x = emit_expression(b, x, binding);
            crate::emit::broadcast(b, op, &[x])
        }
        Expr::Binary(op, x, y) => {
            let op = match op {
                Arithmetic::Add => "add",
                Arithmetic::Subtract => "sub",
                Arithmetic::Multiply => "mul",
                Arithmetic::Divide => "divide",
                Arithmetic::Power => "pow",
            };
            let x = emit_expression(b, x, binding);
            let y = emit_expression(b, y, binding);
            crate::emit::broadcast(b, op, &[x, y])
        }
    }
}
