//! Fixed host values become ordinary FlatPPL literal and constructor nodes, and
//! literal bindings read back as host values.

use flatppl_core::{
    Call, CallHead, Dim, Module, NamedArg, NamedKind, Node, NodeId, RefNs, Scalar, ScalarType,
    Symbol, Type, ValueSet,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A fixed value. JSON writes infinite reals as `"inf"` and `"-inf"`, and an
/// array as one typed flat list, `{"shape", "dtype", "data"}`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Constant {
    Bool(bool),
    Integer(i64),
    Real(#[serde(with = "real")] f64),
    String(String),
    /// Flat data in axis order from slowest to fastest (C order).
    #[serde(
        serialize_with = "packed::serialize",
        deserialize_with = "packed::deserialize"
    )]
    Array {
        shape: Vec<u32>,
        data: Vec<Constant>,
    },
    Record(Vec<(String, Constant)>),
    Table(Vec<(String, Constant)>),
    Tuple(Vec<Constant>),
    /// A `fixed(value)` marker read from source. It binds no external.
    Fixed(Box<Constant>),
    /// A reference to another binding read from source. It binds no external.
    Reference(String),
}

mod real {
    use super::*;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Json {
        Number(f64),
        Text(String),
    }

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        match *value {
            f64::INFINITY => serializer.serialize_str("inf"),
            f64::NEG_INFINITY => serializer.serialize_str("-inf"),
            value => serializer.serialize_f64(value),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        match Json::deserialize(deserializer)? {
            Json::Number(value) => Ok(value),
            Json::Text(text) if text == "inf" => Ok(f64::INFINITY),
            Json::Text(text) if text == "-inf" => Ok(f64::NEG_INFINITY),
            Json::Text(text) => Err(serde::de::Error::custom(format!(
                "expected a number, \"inf\" or \"-inf\", got {text:?}"
            ))),
        }
    }
}

mod packed {
    use super::*;

    #[derive(Deserialize, Serialize)]
    struct Real(#[serde(with = "real")] f64);

    #[derive(Deserialize, Serialize)]
    #[serde(tag = "dtype", content = "data", rename_all = "snake_case")]
    enum Data {
        Bool(Vec<bool>),
        Integer(Vec<i64>),
        Real(Vec<Real>),
    }

    #[derive(Deserialize, Serialize)]
    struct Array {
        shape: Vec<u32>,
        #[serde(flatten)]
        data: Data,
    }

    pub fn serialize<S: Serializer>(
        shape: &[u32],
        data: &[Constant],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let fail = || serde::ser::Error::custom("array elements must be scalars of one kind");
        let packed = match data.first() {
            Some(Constant::Bool(_)) | None => data
                .iter()
                .map(|x| {
                    if let Constant::Bool(v) = x {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect::<Option<_>>()
                .map(Data::Bool),
            Some(Constant::Integer(_)) => data
                .iter()
                .map(|x| {
                    if let Constant::Integer(v) = x {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .collect::<Option<_>>()
                .map(Data::Integer),
            Some(Constant::Real(_)) => data
                .iter()
                .map(|x| {
                    if let Constant::Real(v) = x {
                        Some(Real(*v))
                    } else {
                        None
                    }
                })
                .collect::<Option<_>>()
                .map(Data::Real),
            Some(_) => None,
        }
        .ok_or_else(fail)?;
        Array {
            shape: shape.to_vec(),
            data: packed,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<(Vec<u32>, Vec<Constant>), D::Error> {
        let Array { shape, data } = Array::deserialize(deserializer)?;
        let data = match data {
            Data::Bool(values) => values.into_iter().map(Constant::Bool).collect(),
            Data::Integer(values) => values.into_iter().map(Constant::Integer).collect(),
            Data::Real(values) => values
                .into_iter()
                .map(|Real(v)| Constant::Real(v))
                .collect(),
        };
        Ok((shape, data))
    }
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
            Self::Fixed(_) | Self::Reference(_) => {
                return Err("`fixed` markers and references are not values".into());
            }
            _ => unreachable!("scalars handled above"),
        };
        Ok(call(module, head, args, named))
    }

    /// Read a binding's right-hand side back as a value, structurally.
    ///
    /// Literals, `neg` of a number, `inf`, `pi`, and the `vector`, `array`,
    /// `record`, `table`, `tuple` and `fixed` constructors read back. A
    /// reference to another binding reads as [`Constant::Reference`]. Nothing
    /// is evaluated, so any other expression is refused.
    pub(crate) fn read(module: &Module, node: NodeId) -> Result<Self, String> {
        let name = |s: Symbol| module.resolve(s);
        match module.node(node) {
            Node::Lit(Scalar::Bool(v)) => Ok(Self::Bool(*v)),
            Node::Lit(Scalar::Int(v)) => Ok(Self::Integer(*v)),
            Node::Lit(Scalar::Real(v)) => Ok(Self::Real(*v)),
            Node::Lit(Scalar::Str(v)) => Ok(Self::String(v.to_string())),
            Node::Const(s) if name(*s) == "inf" => Ok(Self::Real(f64::INFINITY)),
            Node::Const(s) if name(*s) == "pi" => Ok(Self::Real(std::f64::consts::PI)),
            Node::Ref(r) if r.ns == RefNs::SelfMod => Ok(Self::Reference(name(r.name).into())),
            Node::Call(c) => {
                let CallHead::Builtin(head) = c.head else {
                    return Err("a call is not a literal".into());
                };
                let args = || {
                    c.args
                        .iter()
                        .map(|&a| Self::read(module, a))
                        .collect::<Result<Vec<_>, _>>()
                };
                let fields = || {
                    c.named
                        .iter()
                        .map(|f| Ok((name(f.name).to_string(), Self::read(module, f.value)?)))
                        .collect::<Result<Vec<_>, String>>()
                };
                match (name(head), c.args.len(), c.named.is_empty()) {
                    ("neg", 1, true) => match Self::read(module, c.args[0])? {
                        Self::Integer(v) => Ok(Self::Integer(-v)),
                        Self::Real(v) => Ok(Self::Real(-v)),
                        _ => Err("`neg` of a non-number is not a literal".into()),
                    },
                    ("fixed", 1, true) => Ok(Self::Fixed(Box::new(Self::read(module, c.args[0])?))),
                    ("tuple", _, true) => Ok(Self::Tuple(args()?)),
                    ("record", 0, _) => Ok(Self::Record(fields()?)),
                    ("table", 0, _) => Ok(Self::Table(fields()?)),
                    ("vector", n, true) => Ok(Self::Array {
                        shape: vec![u32::try_from(n).map_err(|_| "vector is too long")?],
                        data: scalars(args()?)?,
                    }),
                    ("array", 3, true) => read_array(module, &c.args),
                    (other, _, _) => Err(format!("`{other}(...)` is not a literal")),
                }
            }
            _ => Err("the expression is not a literal".into()),
        }
    }
}

fn scalars(values: Vec<Constant>) -> Result<Vec<Constant>, String> {
    let kind = std::mem::discriminant;
    match values.first() {
        Some(first @ (Constant::Bool(_) | Constant::Integer(_) | Constant::Real(_)))
            if values.iter().all(|v| kind(v) == kind(first)) =>
        {
            Ok(values)
        }
        None => Ok(values),
        _ => Err("array elements must be literal numbers or booleans of one kind".into()),
    }
}

fn read_array(module: &Module, args: &[NodeId]) -> Result<Constant, String> {
    let integers = |node| match Constant::read(module, node)? {
        Constant::Array { data, .. } => data
            .into_iter()
            .map(|v| match v {
                Constant::Integer(v) => u32::try_from(v).map_err(|_| "negative array extent"),
                _ => Err("array extents must be integers"),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(String::from),
        _ => Err("array extents must be a vector".to_string()),
    };
    let Constant::Array { data, .. } = Constant::read(module, args[0])? else {
        return Err("array data must be a vector".into());
    };
    let shape = integers(args[1])?;
    let order = integers(args[2])?;
    if !order.iter().copied().eq(1..=shape.len() as u32) {
        return Err("only arrays in the default axis order read back".into());
    }
    if shape.iter().map(|&d| d as usize).product::<usize>() != data.len() {
        return Err("array data does not match its shape".into());
    }
    Ok(Constant::Array { shape, data })
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
