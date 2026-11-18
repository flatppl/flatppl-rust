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

/// Stands for a `Dim::Dynamic` extent while a declaration names its axes:
/// the axis must then take a symbolic size (a `filter` count).
pub(crate) const DYNAMIC: u32 = u32::MAX;

/// `ty` with every dynamic extent marked [`DYNAMIC`].
pub(crate) fn mark_dynamic(ty: &Type) -> Type {
    let d = |x: &Dim| match x {
        Dim::Dynamic => Dim::Static(DYNAMIC),
        s => *s,
    };
    match ty {
        Type::Array { shape, elem } => Type::Array {
            shape: shape.iter().map(d).collect(),
            elem: Box::new(mark_dynamic(elem)),
        },
        Type::TVector { len, elem } => Type::TVector {
            len: d(len),
            elem: Box::new(mark_dynamic(elem)),
        },
        t => t.clone(),
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

pub(crate) fn bounds_of(vs: &ValueSet) -> Bounds {
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
                    // `array[n, m] int` indexes as `x[i][j]` and `x[i, j]` alike.
                    let mut t = declare(inner, inner_vs, bounded, size, node)?;
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

#[cfg(test)]
mod tests {
    use super::declare;
    use flatppl_core::{Dim, Idx, NodeId, ScalarType, Type, ValueSet};

    /// Every `sets.ron` row this builder covers, filled from the row's own
    /// template, must equal the declaration `declare` writes for that set.
    #[test]
    fn declarations_match_sets_ron() {
        let real = || Type::Scalar(ScalarType::Real);
        let int = || Type::Scalar(ScalarType::Integer);
        let vector = |elem: Type| Type::Array {
            shape: vec![Dim::Static(4)].into(),
            elem: Box::new(elem),
        };
        let matrix = Type::Array {
            shape: vec![Dim::Static(3), Dim::Static(4)].into(),
            elem: Box::new(real()),
        };
        let inf = f64::INFINITY;
        // (row key, type, value set, {lo}, {hi}, {bounds})
        let cases: Vec<(&str, Type, ValueSet, &str, &str, &str)> = vec![
            ("reals", real(), ValueSet::Reals, "", "", ""),
            (
                "interval(-inf, inf)",
                real(),
                ValueSet::Interval(-inf, inf),
                "",
                "",
                "",
            ),
            ("posreals", real(), ValueSet::PosReals, "", "", ""),
            ("nonnegreals", real(), ValueSet::NonNegReals, "", "", ""),
            ("unitinterval", real(), ValueSet::UnitInterval, "", "", ""),
            (
                "interval(a, b)",
                real(),
                ValueSet::Interval(-1.5, 2.5),
                "-1.5",
                "2.5",
                "",
            ),
            (
                "interval(a, inf)",
                real(),
                ValueSet::Interval(0.5, inf),
                "0.5",
                "",
                "",
            ),
            (
                "interval(-inf, b)",
                real(),
                ValueSet::Interval(-inf, 3.0),
                "",
                "3",
                "",
            ),
            (
                "ordered(n)",
                vector(real()),
                ValueSet::Ordered(Dim::Static(4)),
                "",
                "",
                "",
            ),
            (
                "zerosum(n)",
                vector(real()),
                ValueSet::ZeroSum(Dim::Static(4)),
                "",
                "",
                "",
            ),
            (
                "stdsimplex(n)",
                vector(real()),
                ValueSet::StdSimplex(Dim::Static(4)),
                "",
                "",
                "",
            ),
            (
                "cartpow(S, n)",
                vector(real()),
                ValueSet::CartPow(Box::new(ValueSet::PosReals), Dim::Static(4)),
                "",
                "",
                "<lower=0>",
            ),
            (
                "cartpow(S, n), S integer",
                vector(int()),
                ValueSet::CartPow(Box::new(ValueSet::NonNegIntegers), Dim::Static(4)),
                "",
                "",
                "<lower=0>",
            ),
            (
                "cartpow(S, [m, n])",
                matrix,
                ValueSet::CartPow(
                    Box::new(ValueSet::CartPow(
                        Box::new(ValueSet::UnitInterval),
                        Dim::Static(4),
                    )),
                    Dim::Static(3),
                ),
                "",
                "",
                "<lower=0, upper=1>",
            ),
            ("integers", int(), ValueSet::Integers, "", "", ""),
            ("posintegers", int(), ValueSet::PosIntegers, "", "", ""),
            (
                "nonnegintegers",
                int(),
                ValueSet::NonNegIntegers,
                "",
                "",
                "",
            ),
            (
                "booleans",
                Type::Scalar(ScalarType::Boolean),
                ValueSet::Booleans,
                "",
                "",
                "",
            ),
        ];
        // Rows the emitter builds elsewhere or refuses: records flatten to
        // one declaration per field, complex values and the `none` rows refuse.
        let elsewhere = [
            "cartprod(S1, ..., Sk)",
            "cartprod(a = S1, ...)",
            "complexes",
            "anything",
            "rngstates",
        ];
        for row in crate::tables::sets() {
            if elsewhere.contains(&row.set.as_str()) {
                continue;
            }
            let (_, ty, vs, lo, hi, bounds) =
                cases.iter().find(|c| c.0 == row.set).unwrap_or_else(|| {
                    panic!("sets.ron row `{}` is not bound to stan_type.rs", row.set)
                });
            let expected = row
                .decl
                .replace("{name}", "x")
                .replace("{lo}", lo)
                .replace("{hi}", hi)
                .replace("{bounds}", bounds)
                .replace("{m}", "m")
                .replace("{n}", "n");
            let size = &mut |e: u32| {
                if e == 3 {
                    "m".to_string()
                } else {
                    "n".to_string()
                }
            };
            let got = declare(ty, Some(vs), true, size, NodeId::from_usize(0)).expect("declares");
            assert_eq!(
                format!("{};", got.render("x")),
                expected,
                "sets.ron row `{}`",
                row.set
            );
        }
    }
}
