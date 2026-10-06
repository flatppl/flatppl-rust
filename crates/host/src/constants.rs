//! Fixed host values become ordinary FlatPPL literal and constructor nodes.

use flatppl_core::{
    Call, CallHead, Dim, Module, NamedArg, NamedKind, Node, NodeId, Scalar, ScalarType, Symbol,
    Type, ValueSet,
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Constant {
    Bool(bool),
    Integer(i64),
    Real(f64),
    String(String),
    /// Flat data in axis order from slowest to fastest (C order).
    Array {
        shape: Vec<u32>,
        data: Vec<Constant>,
    },
    Record(Vec<(String, Constant)>),
    Table(Vec<(String, Constant)>),
    Tuple(Vec<Constant>),
}

impl Constant {
    pub(crate) fn validated(&self, module: &Module, declaration: NodeId) -> Result<Self, String> {
        let ty = module
            .type_of(declaration)
            .ok_or("unresolved external type")?;
        let set = module
            .valueset_of(declaration)
            .ok_or("unresolved external domain")?;
        let value = self.normalized(module, ty)?;
        if !value.in_set(module, set)? {
            return Err(format!("value is outside {}", module.display_valueset(set)));
        }
        Ok(value)
    }

    pub(crate) fn normalized(&self, module: &Module, ty: &Type) -> Result<Self, String> {
        match (self, ty) {
            (_, Type::Any) => Ok(self.clone()),
            (Self::Bool(_), Type::Scalar(ScalarType::Boolean))
            | (Self::Integer(_), Type::Scalar(ScalarType::Integer))
            | (Self::Real(_), Type::Scalar(ScalarType::Real)) => Ok(self.clone()),
            (Self::Bool(value), Type::Scalar(ScalarType::Integer)) => {
                Ok(Self::Integer(i64::from(*value)))
            }
            (Self::Bool(value), Type::Scalar(ScalarType::Real)) => {
                Ok(Self::Real(f64::from(*value)))
            }
            (Self::Integer(value), Type::Scalar(ScalarType::Real))
                if (*value as f64) as i128 == i128::from(*value) =>
            {
                Ok(Self::Real(*value as f64))
            }
            (
                Self::Array { shape, data },
                Type::Array {
                    shape: expected,
                    elem,
                },
            ) if shape.len() == expected.len()
                && shape
                    .iter()
                    .zip(expected.iter())
                    .all(|(&n, &d)| d == Dim::Static(n) || d == Dim::Dynamic) =>
            {
                Ok(Self::Array {
                    shape: shape.clone(),
                    data: data
                        .iter()
                        .map(|v| v.normalized(module, elem))
                        .collect::<Result<_, _>>()?,
                })
            }
            (Self::Record(fields), Type::Record(expected)) => {
                Ok(Self::Record(normalize_fields(module, fields, expected)?))
            }
            (Self::Table(fields) | Self::Record(fields), Type::Table { columns, nrows }) => {
                let expected: Vec<_> = columns
                    .iter()
                    .map(|(name, ty)| {
                        (
                            *name,
                            match ty {
                                Type::Record(fields) => Type::Table {
                                    columns: fields.clone(),
                                    nrows: *nrows,
                                },
                                _ => Type::Array {
                                    shape: vec![*nrows].into(),
                                    elem: Box::new(ty.clone()),
                                },
                            },
                        )
                    })
                    .collect();
                Ok(Self::Table(normalize_fields(module, fields, &expected)?))
            }
            (Self::Tuple(values), Type::Tuple(types)) if values.len() == types.len() => {
                Ok(Self::Tuple(
                    values
                        .iter()
                        .zip(types.iter())
                        .map(|(v, t)| v.normalized(module, t))
                        .collect::<Result<_, _>>()?,
                ))
            }
            _ => Err(format!(
                "value does not match declared type {ty:?}; dimensions must be fixed"
            )),
        }
    }

    fn in_set(&self, module: &Module, set: &ValueSet) -> Result<bool, String> {
        use ValueSet::*;
        let number = match self {
            Self::Bool(v) => Some(f64::from(*v)),
            Self::Integer(v) => Some(*v as f64),
            Self::Real(v) => Some(*v),
            _ => None,
        };
        if let Some(value) = number {
            let contains = match set {
                Booleans => value == 0.0 || value == 1.0,
                Integers => value.is_finite() && value.fract() == 0.0,
                PosIntegers => value > 0.0 && value.is_finite() && value.fract() == 0.0,
                NonNegIntegers => value >= 0.0 && value.is_finite() && value.fract() == 0.0,
                Reals => !value.is_nan(),
                PosReals => value > 0.0,
                NonNegReals => value >= 0.0,
                UnitInterval => (0.0..=1.0).contains(&value),
                Interval(lo, hi) => *lo <= value && value <= *hi,
                _ => {
                    return match set {
                        Anything => Ok(true),
                        Deferred | Unknown => {
                            Err("external domain cannot be resolved after substitution".into())
                        }
                        _ => Ok(false),
                    };
                }
            };
            return Ok(contains);
        }
        Ok(match (self, set) {
            (_, Anything) => true,
            (_, Deferred | Unknown) => {
                return Err("external domain cannot be resolved after substitution".into());
            }
            (Self::Array { shape, data }, _) => array_in_set(module, shape, data, set)?,
            (Self::Record(fields), RecordSet(sets)) => {
                fields.len() == sets.len()
                    && fields
                        .iter()
                        .zip(sets.iter())
                        .map(|((name, value), (key, set))| {
                            Ok(name == module.resolve(*key) && value.in_set(module, set)?)
                        })
                        .collect::<Result<Vec<_>, String>>()?
                        .into_iter()
                        .all(|v| v)
            }
            (Self::Table(columns), CartPow(row, n)) => {
                if let RecordSet(fields) = row.as_ref() {
                    columns.len() == fields.len()
                        && columns
                            .iter()
                            .zip(fields.iter())
                            .map(|((name, value), (key, set))| {
                                Ok(name == module.resolve(*key)
                                    && value.in_set(module, &CartPow(Box::new(set.clone()), *n))?)
                            })
                            .collect::<Result<Vec<_>, String>>()?
                            .into_iter()
                            .all(|v| v)
                } else {
                    false
                }
            }
            _ => false,
        })
    }

    pub(crate) fn alloc(&self, module: &mut Module) -> Result<NodeId, String> {
        let scalar = match self {
            Self::Bool(value) => Some(Scalar::Bool(*value)),
            Self::Integer(value) => Some(Scalar::Int(*value)),
            Self::Real(value) => Some(Scalar::Real(*value)),
            Self::String(value) => Some(Scalar::Str(value.clone().into())),
            _ => None,
        };
        if let Some(scalar) = scalar {
            return Ok(module.alloc(Node::Lit(scalar)));
        }
        let (head, args, named) = match self {
            Self::Array { shape, data } => {
                let size = shape
                    .iter()
                    .try_fold(1_usize, |n, &d| n.checked_mul(d as usize));
                if shape.is_empty() || shape.contains(&0) || size != Some(data.len()) {
                    return Err("constant array shape must match its nonempty flat data".into());
                }
                let values = data
                    .iter()
                    .map(|x| x.alloc(module))
                    .collect::<Result<Vec<_>, _>>()?;
                let values = call(module, "vector", values, vec![]);
                if shape.len() == 1 {
                    return Ok(values);
                }
                let sizes = shape
                    .iter()
                    .map(|&d| module.alloc(Node::Lit(Scalar::Int(d.into()))))
                    .collect();
                let sizes = call(module, "vector", sizes, vec![]);
                let order = (1..=shape.len())
                    .map(|d| module.alloc(Node::Lit(Scalar::Int(d as i64))))
                    .collect();
                let order = call(module, "vector", order, vec![]);
                ("array", vec![values, sizes, order], vec![])
            }
            Self::Record(fields) | Self::Table(fields) => {
                let mut named = Vec::new();
                for (name, value) in fields {
                    let name = module.intern(name);
                    if named.iter().any(|field: &NamedArg| field.name == name) {
                        return Err("constant record fields must be unique".into());
                    }
                    named.push(NamedArg {
                        name,
                        value: value.alloc(module)?,
                        kind: NamedKind::Field,
                    });
                }
                (
                    if matches!(self, Self::Table(_)) {
                        "table"
                    } else {
                        "record"
                    },
                    vec![],
                    named,
                )
            }
            Self::Tuple(items) => (
                "tuple",
                items
                    .iter()
                    .map(|x| x.alloc(module))
                    .collect::<Result<_, _>>()?,
                vec![],
            ),
            _ => unreachable!("scalars handled above"),
        };
        Ok(call(module, head, args, named))
    }
}

fn normalize_fields(
    module: &Module,
    fields: &[(String, Constant)],
    expected: &[(Symbol, Type)],
) -> Result<Vec<(String, Constant)>, String> {
    if fields.len() != expected.len() {
        return Err("field names must match the external declaration".into());
    }
    expected
        .iter()
        .map(|(name, ty)| {
            let name = module.resolve(*name);
            let (_, value) = fields
                .iter()
                .find(|(key, _)| key == name)
                .ok_or_else(|| format!("missing field `{name}`"))?;
            Ok((name.into(), value.normalized(module, ty)?))
        })
        .collect()
}

fn array_in_set(
    module: &Module,
    shape: &[u32],
    data: &[Constant],
    set: &ValueSet,
) -> Result<bool, String> {
    use ValueSet::*;
    if shape.is_empty() {
        return data[0].in_set(module, set);
    }
    match set {
        CartPow(elem, Dim::Static(n)) if *n == shape[0] => {
            if *n == 0 {
                return Ok(data.is_empty());
            }
            let width = data.len() / *n as usize;
            for row in data.chunks(width) {
                if !array_in_set(module, &shape[1..], row, elem)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        StdSimplex(Dim::Static(n)) if shape == [*n] => {
            let mut sum = 0.0;
            for value in data {
                let Constant::Real(value) = value else {
                    return Ok(false);
                };
                if !value.is_finite() || *value < 0.0 {
                    return Ok(false);
                }
                sum += value;
            }
            Ok(sum == 1.0)
        }
        CartProd(parts) if shape.len() == 1 => {
            let mut offset = 0;
            for part in parts {
                let width = product_width(part).ok_or("unresolved Cartesian product domain")?;
                let Some(values) = data.get(offset..offset + width) else {
                    return Ok(false);
                };
                let dims = match part {
                    CartPow(_, _) | CartProd(_) | StdSimplex(_) => vec![width as u32],
                    _ => vec![],
                };
                if !array_in_set(module, &dims, values, part)? {
                    return Ok(false);
                }
                offset += width;
            }
            Ok(offset == data.len())
        }
        Deferred | Unknown | CartPow(_, Dim::Dynamic) | StdSimplex(Dim::Dynamic) => {
            Err("external domain cannot be resolved after substitution".into())
        }
        _ => Ok(false),
    }
}

fn product_width(set: &ValueSet) -> Option<usize> {
    match set {
        ValueSet::CartPow(_, Dim::Static(n)) | ValueSet::StdSimplex(Dim::Static(n)) => {
            Some(*n as usize)
        }
        ValueSet::CartProd(parts) => parts
            .iter()
            .try_fold(0_usize, |n, set| n.checked_add(product_width(set)?)),
        ValueSet::Reals
        | ValueSet::PosReals
        | ValueSet::NonNegReals
        | ValueSet::UnitInterval
        | ValueSet::Integers
        | ValueSet::PosIntegers
        | ValueSet::NonNegIntegers
        | ValueSet::Booleans
        | ValueSet::Interval(_, _) => Some(1),
        _ => None,
    }
}

pub(crate) fn call(
    module: &mut Module,
    head: &str,
    args: Vec<NodeId>,
    named: Vec<NamedArg>,
) -> NodeId {
    let head = CallHead::Builtin(module.intern(head));
    module.alloc(Node::Call(Call {
        head,
        args: args.into(),
        named: named.into(),
        inputs: None,
    }))
}
