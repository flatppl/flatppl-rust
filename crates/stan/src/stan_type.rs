//! Stan declarations from inferred FlatPPL types and value sets.

use crate::refuse::{Refusal, Result};
use flatppl_core::{Dim, NodeId, ScalarType, Type, ValueSet};

/// A declared Stan type, printed as `base<bounds>[dims] name` pieces.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StanType {
    /// Outer `array[...]` dimensions.
    pub array: Vec<String>,
    /// `real`, `int`, `vector`, `simplex`, `matrix`, …
    pub base: &'static str,
    /// The base's own dimensions (`vector[n]`, `matrix[r, c]`).
    pub dims: Vec<String>,
    pub bounds: Bounds,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Bounds {
    pub lower: Option<String>,
    pub upper: Option<String>,
}

impl StanType {
    pub(crate) fn render(&self, name: &str) -> String {
        let mut out = String::new();
        if !self.array.is_empty() {
            out.push_str(&format!("array[{}] ", self.array.join(", ")));
        }
        out.push_str(self.base);
        out.push_str(&self.bounds.render());
        if !self.dims.is_empty() {
            out.push_str(&format!("[{}]", self.dims.join(", ")));
        }
        out.push(' ');
        out.push_str(name);
        out
    }

    pub(crate) fn is_int(&self) -> bool {
        self.base == "int"
    }

    /// Rank of the value as Stan indexes it.
    pub(crate) fn rank(&self) -> usize {
        self.array.len() + self.dims.len()
    }
}

impl Bounds {
    fn render(&self) -> String {
        match (&self.lower, &self.upper) {
            (None, None) => String::new(),
            (Some(l), None) => format!("<lower={l}>"),
            (None, Some(u)) => format!("<upper={u}>"),
            (Some(l), Some(u)) => format!("<lower={l}, upper={u}>"),
        }
    }
}

/// Renders a static extent; the caller may substitute a size name.
pub(crate) type SizeName<'a> = &'a mut dyn FnMut(u32) -> String;

fn extent(dim: &Dim, size: SizeName<'_>, node: NodeId) -> Result<String> {
    match dim {
        Dim::Static(n) => Ok(size(*n)),
        Dim::Dynamic => Err(Refusal::not_yet(Some(node), "a dynamic extent")),
    }
}

/// The scalar value set of an array value set (`cartpow(S, n)` → `S`).
fn element_set(vs: &ValueSet) -> &ValueSet {
    match vs {
        ValueSet::CartPow(inner, _) => element_set(inner),
        other => other,
    }
}

pub(crate) fn number(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn bounds_of(vs: &ValueSet) -> Bounds {
    let b = |l: Option<&str>, u: Option<&str>| Bounds {
        lower: l.map(str::to_string),
        upper: u.map(str::to_string),
    };
    match element_set(vs) {
        ValueSet::PosReals | ValueSet::NonNegReals | ValueSet::NonNegIntegers => b(Some("0"), None),
        ValueSet::PosIntegers => b(Some("1"), None),
        ValueSet::UnitInterval | ValueSet::Booleans => b(Some("0"), Some("1")),
        ValueSet::Interval(lo, hi) => Bounds {
            lower: lo.is_finite().then(|| number(*lo)),
            upper: hi.is_finite().then(|| number(*hi)),
        },
        _ => Bounds::default(),
    }
}

/// The declaration of a value of inferred type `ty` and value set `vs`.
/// `bounded` adds `<lower, upper>` (data and parameters); locals omit them.
pub(crate) fn declare(
    ty: &Type,
    vs: Option<&ValueSet>,
    bounded: bool,
    size: SizeName<'_>,
    node: NodeId,
) -> Result<StanType> {
    let unknown = ValueSet::Unknown;
    let vs = vs.unwrap_or(&unknown);
    let bounds = if bounded {
        bounds_of(vs)
    } else {
        Bounds::default()
    };
    let scalar = |s: &ScalarType| -> Result<&'static str> {
        match s {
            ScalarType::Real => Ok("real"),
            ScalarType::Integer | ScalarType::Boolean => Ok("int"),
            ScalarType::Complex => Err(Refusal::not_yet(Some(node), "a complex value")),
        }
    };
    match ty {
        Type::Scalar(s) => Ok(StanType {
            array: vec![],
            base: scalar(s)?,
            dims: vec![],
            bounds,
        }),
        Type::TVector { len, elem } => match elem.as_ref() {
            Type::Scalar(ScalarType::Real) => Ok(StanType {
                array: vec![],
                base: "row_vector",
                dims: vec![extent(len, size, node)?],
                bounds,
            }),
            _ => Err(Refusal::not_yet(Some(node), "a non-real transposed vector")),
        },
        Type::Array { shape, elem } => {
            let dims = shape
                .iter()
                .map(|d| extent(d, size, node))
                .collect::<Result<Vec<_>>>()?;
            match elem.as_ref() {
                Type::Scalar(ScalarType::Real) => {
                    // A constrained vector type carries its own bounds.
                    let special = if dims.len() == 1 && bounded {
                        match vs {
                            ValueSet::StdSimplex(_) => Some("simplex"),
                            ValueSet::Ordered(_) => Some("ordered"),
                            ValueSet::ZeroSum(_) => Some("sum_to_zero_vector"),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some(base) = special {
                        return Ok(StanType {
                            array: vec![],
                            base,
                            dims,
                            bounds: Bounds::default(),
                        });
                    }
                    match dims.len() {
                        1 => Ok(StanType {
                            array: vec![],
                            base: "vector",
                            dims,
                            bounds,
                        }),
                        2 => Ok(StanType {
                            array: vec![],
                            base: "matrix",
                            dims,
                            bounds,
                        }),
                        _ => Ok(StanType {
                            array: dims,
                            base: "real",
                            dims: vec![],
                            bounds,
                        }),
                    }
                }
                Type::Scalar(s) => Ok(StanType {
                    array: dims,
                    base: scalar(s)?,
                    dims: vec![],
                    bounds,
                }),
                inner @ (Type::Array { .. } | Type::TVector { .. }) => {
                    let inner_vs = match vs {
                        ValueSet::CartPow(inner, _) => Some(inner.as_ref()),
                        _ => None,
                    };
                    let mut t = declare(inner, inner_vs, bounded, size, node)?;
                    if !t.array.is_empty() {
                        return Err(Refusal::not_yet(Some(node), "an array of integer arrays"));
                    }
                    let mut array = dims;
                    array.extend(t.array);
                    t.array = array;
                    Ok(t)
                }
                _ => Err(Refusal::not_yet(
                    Some(node),
                    "an array of this element type",
                )),
            }
        }
        Type::Record(_) | Type::Table { .. } | Type::Tuple(_) => Err(Refusal::not_yet(
            Some(node),
            "a record value outside a data source",
        )),
        _ => Err(Refusal::not_yet(Some(node), "a value of this type")),
    }
}
