//! Canonical coordinates for continuous domains, expressed with existing FlatPPL operations.

use std::sync::Arc;

use flatppl_core::Type;

use crate::{Domain, DomainField, LoadedModule};

/// Three query modules accompanying an unchanged constrained query.
pub struct TransformedQueries {
    /// The original output evaluated in free coordinates, optionally with a density correction.
    pub unconstrained: Arc<LoadedModule>,
    /// Map free coordinates to the declared domain.
    pub constrain: Arc<LoadedModule>,
    /// Map interior domain values to free coordinates, without clipping.
    pub unconstrain: Arc<LoadedModule>,
}

pub(crate) struct Chart {
    pub domain: String,
    pub free: String,
    pub constrain: String,
    pub unconstrain: String,
    pub log_det: String,
    size: u32,
    free_size: u32,
    scalar: bool,
}

fn vector_range(point: &str, start: u32, end: u32) -> String {
    vector_select(point, start..=end)
}

fn vector_select(point: &str, indices: impl Iterator<Item = u32>) -> String {
    let indices = indices
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{point}[[{indices}]]")
}

fn sum(parts: impl Iterator<Item = String>) -> String {
    let parts: Vec<_> = parts.filter(|part| part != "0.0").collect();
    if parts.is_empty() {
        "0.0".into()
    } else {
        parts.join(" + ")
    }
}

// Inspection metadata flattens nested Cartesian powers. Do not rebuild those
// as matrices: their broadcast semantics differ even when the ABI agrees.
pub(crate) fn supported_container(ty: &Type) -> bool {
    match ty {
        Type::Scalar(_) => true,
        Type::Array { elem, .. } => matches!(elem.as_ref(), Type::Scalar(_)),
        Type::Record(fields) => fields.iter().all(|(_, ty)| supported_container(ty)),
        _ => false,
    }
}

fn record_fields(domain: &Domain) -> Option<Vec<DomainField>> {
    match domain {
        Domain::Record { fields } => Some(fields.clone()),
        Domain::CartProd { of } => Some(
            of.iter()
                .map(record_fields)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect(),
        ),
        _ => None,
    }
}

impl Domain {
    pub(crate) fn chart(&self, point: &str) -> Result<Chart, String> {
        let scalar =
            |domain: String, constrain: String, unconstrain: String, log_det: String| Chart {
                domain,
                free: "reals".into(),
                constrain,
                unconstrain,
                log_det,
                size: 1,
                free_size: 1,
                scalar: true,
            };
        let positive_log_det =
            || format!("logdensityof(pushfwd(log, Lebesgue(posreals)), {point})");
        match self {
            Self::Reals => Ok(scalar(
                "reals".into(),
                point.into(),
                point.into(),
                "0.0".into(),
            )),
            Self::PosReals | Self::NonNegReals => Ok(scalar(
                if matches!(self, Self::PosReals) {
                    "posreals"
                } else {
                    "nonnegreals"
                }
                .into(),
                format!("exp({point})"),
                format!("log({point})"),
                positive_log_det(),
            )),
            Self::UnitInterval => Self::Interval {
                lo: crate::Bound(0.0),
                hi: crate::Bound(1.0),
            }
            .chart(point),
            Self::Interval { lo, hi } => {
                let (lo, hi) = (lo.0, hi.0);
                if lo.is_nan() || hi.is_nan() || lo >= hi {
                    return Err("a coordinate transform needs a nonempty interval".into());
                }
                let domain = format!("interval({lo}, {hi})");
                let (forward, inverse, log_det) = match (lo.is_finite(), hi.is_finite()) {
                    (false, false) => (point.into(), point.into(), "0.0".into()),
                    (true, false) => (
                        format!("{lo} + exp({point})"),
                        format!("log({point} - ({lo}))"),
                        positive_log_det(),
                    ),
                    (false, true) => (
                        format!("{hi} - exp({point})"),
                        format!("log(({hi}) - {point})"),
                        positive_log_det(),
                    ),
                    (true, true) => {
                        let width = hi - lo;
                        if !width.is_finite() {
                            return Err("interval width overflows the coordinate transform".into());
                        }
                        (
                            format!("{lo} + ({width}) * invlogit({point})"),
                            format!("logit(({point} - ({lo})) / ({width}))"),
                            format!(
                                "log({width}) + logdensityof(pushfwd(logit, Lebesgue(unitinterval)), {point})"
                            ),
                        )
                    }
                };
                Ok(scalar(domain, forward, inverse, log_det))
            }
            Self::CartPow { of, shape } => {
                let shape = shape
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                    .ok_or("bind domain sizes before building coordinate transforms")?;
                let cell = of.chart("_")?;
                if !cell.scalar {
                    return Err(
                        "Cartesian powers of non-scalar domains need a structured chart".into(),
                    );
                }
                let size = shape
                    .iter()
                    .try_fold(1u32, |a, &b| a.checked_mul(b))
                    .filter(|&n| n > 0)
                    .ok_or("coordinate array size is zero or overflows")?;
                let dims = shape
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let lift = |expression: &str| {
                    if expression == "_" {
                        point.into()
                    } else {
                        format!("broadcast(fn({expression}), {point})")
                    }
                };
                Ok(Chart {
                    domain: format!("cartpow({}, [{dims}])", cell.domain),
                    free: format!("cartpow(reals, [{dims}])"),
                    constrain: lift(&cell.constrain),
                    unconstrain: lift(&cell.unconstrain),
                    log_det: if cell.log_det == "0.0" {
                        "0.0".into()
                    } else {
                        format!("sum({})", lift(&cell.log_det))
                    },
                    size,
                    free_size: size,
                    scalar: false,
                })
            }
            Self::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|field| {
                        Ok((
                            &field.name,
                            field.set.chart(&format!("{point}.{}", field.name))?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let join = |value: fn(&Chart) -> &String| {
                    fields
                        .iter()
                        .map(|(name, chart)| format!("{name} = {}", value(chart)))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                Ok(Chart {
                    domain: format!("cartprod({})", join(|c| &c.domain)),
                    free: format!("cartprod({})", join(|c| &c.free)),
                    constrain: format!("record({})", join(|c| &c.constrain)),
                    unconstrain: format!("record({})", join(|c| &c.unconstrain)),
                    log_det: sum(fields.iter().map(|(_, c)| c.log_det.clone())),
                    // Record products concatenate fields, never tensor slices.
                    size: 0,
                    free_size: 0,
                    scalar: false,
                })
            }
            Self::Ordered { n: Some(n) } if *n > 0 => {
                let (forward, inverse, log_det) = if *n == 1 {
                    (point.into(), point.into(), "0.0".into())
                } else {
                    let tail = vector_range(point, 2, *n);
                    let head = vector_range(point, 1, n - 1);
                    (
                        format!("cumsum(cat([{point}[1]], exp.({tail})))"),
                        format!("cat([{point}[1]], log.({tail} .- {head}))"),
                        format!("sum({tail})"),
                    )
                };
                Ok(Chart {
                    domain: format!("ordered({n})"),
                    free: format!("cartpow(reals, {n})"),
                    constrain: forward,
                    unconstrain: inverse,
                    log_det,
                    size: *n,
                    free_size: *n,
                    scalar: false,
                })
            }
            Self::ZeroSum { n: Some(n) } if *n > 1 => Ok(Chart {
                domain: format!("zerosum({n})"),
                free: format!("cartpow(reals, {})", n - 1),
                constrain: format!("cat({point}, [-sum({point})])"),
                unconstrain: vector_range(point, 1, n - 1),
                log_det: "0.0".into(),
                size: *n,
                free_size: n - 1,
                scalar: false,
            }),
            Self::StdSimplex { n: Some(n) } if *n > 1 => {
                let offsets = (1..*n)
                    .rev()
                    .map(|i| (i as f64).ln().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let z = format!("({point} .- [{offsets}])");
                let breaks = format!("invlogit.({z})");
                let log_sig =
                    |x: &str| format!("((-abs.({x}) .+ {x}) ./ 2.0 .- log1p.(exp.(-abs.({x}))))");
                // Prefix sums in log space avoid products that round to zero
                // before their derivatives, and use the qualified cumsum AD path.
                let remaining = format!("exp.(cumsum({}))", log_sig(&format!("(-{z})")));
                let before = if *n == 2 {
                    "[1.0]".into()
                } else {
                    format!(
                        "cat([1.0], {})",
                        vector_range(&format!("({remaining})"), 1, n - 2)
                    )
                };
                let head = vector_range(point, 1, n - 1);
                // Sum positive tails rather than subtract prefixes from one:
                // a valid interior point can have a prefix that rounds to one.
                let tail = vector_select(point, (2..=*n).rev());
                let remaining_tail = vector_select(&format!("(cumsum({tail}))"), (1..*n).rev());
                // Centered stick-breaking. The reference is coordinate Lebesgue,
                // so there is no surface-volume factor sqrt(n).
                let counts = (1..*n)
                    .rev()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(Chart {
                    domain: format!("stdsimplex({n})"),
                    free: format!("cartpow(reals, {})", n - 1),
                    constrain: format!("cat({breaks} .* {before}, [({remaining})[{}]])", n - 1),
                    unconstrain: format!("log.({head}) .- log.({remaining_tail}) .+ [{offsets}]"),
                    log_det: format!(
                        "sum({} .+ [{counts}] .* {})",
                        log_sig(&z),
                        log_sig(&format!("(-{z})"))
                    ),
                    size: *n,
                    free_size: n - 1,
                    scalar: false,
                })
            }
            Self::CartProd { of } => {
                if let Some(fields) = record_fields(self) {
                    return Self::Record { fields }.chart(point);
                }
                let mut start = 1u32;
                let mut free_start = 1u32;
                let mut parts = Vec::new();
                for domain in of {
                    if matches!(domain, Self::CartPow { shape, .. } if shape.len() != 1) {
                        return Err("coordinate products currently require scalar or flat-vector components".into());
                    }
                    let base = domain.chart(point)?;
                    let end = start
                        .checked_add(base.size)
                        .ok_or("coordinate size overflows")?;
                    let free_end = free_start
                        .checked_add(base.free_size)
                        .ok_or("coordinate size overflows")?;
                    let access = |start: u32, size: u32| {
                        if base.scalar {
                            format!("{point}[{start}]")
                        } else {
                            vector_range(point, start, start + size - 1)
                        }
                    };
                    let forward = domain.chart(&access(free_start, base.free_size))?;
                    let inverse = domain.chart(&access(start, base.size))?;
                    start = end;
                    free_start = free_end;
                    parts.push((base, forward, inverse));
                }
                let join = |value: fn(&(Chart, Chart, Chart)) -> String| {
                    parts.iter().map(value).collect::<Vec<_>>().join(", ")
                };
                let array =
                    |s: &str, scalar: bool| if scalar { format!("[{s}]") } else { s.into() };
                Ok(Chart {
                    domain: format!("cartprod({})", join(|p| p.0.domain.clone())),
                    free: format!("cartprod({})", join(|p| p.0.free.clone())),
                    constrain: format!(
                        "cat({})",
                        parts
                            .iter()
                            .map(|p| array(&p.1.constrain, p.0.scalar))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    unconstrain: format!(
                        "cat({})",
                        parts
                            .iter()
                            .map(|p| array(&p.2.unconstrain, p.0.scalar))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    log_det: sum(parts.iter().map(|p| p.1.log_det.clone())),
                    size: start - 1,
                    free_size: free_start - 1,
                    scalar: false,
                })
            }
            _ => Err(format!("no continuous coordinate transform for {self:?}")),
        }
    }
}
