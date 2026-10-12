use crate::{Error, Result, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::NodeId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    #[default]
    Gaussian,
    Student,
    Lognormal,
    Bernoulli,
    Binomial,
    Poisson,
    Negbinomial,
    Geometric,
    #[serde(rename = "Gamma")]
    Gamma,
    Exponential,
    Weibull,
    Beta,
    #[serde(rename = "von_mises")]
    VonMises,
    Cumulative,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Link {
    #[default]
    Identity,
    Log,
    Inverse,
    Sqrt,
    Softplus,
    Squareplus,
    Logit,
    Probit,
    ProbitApprox,
    Cloglog,
    Cauchit,
    Softit,
    TanHalf,
}

impl Family {
    pub(crate) fn resolve(name: &str, link: Option<&str>) -> Result<(Self, Link)> {
        let family = match name {
            "gaussian" => Self::Gaussian,
            "student" => Self::Student,
            "lognormal" => Self::Lognormal,
            "bernoulli" => Self::Bernoulli,
            "binomial" => Self::Binomial,
            "poisson" => Self::Poisson,
            "negbinomial" => Self::Negbinomial,
            "geometric" => Self::Geometric,
            "Gamma" | "gamma" => Self::Gamma,
            "exponential" => Self::Exponential,
            "weibull" => Self::Weibull,
            "beta" => Self::Beta,
            "von_mises" => Self::VonMises,
            "cumulative" => Self::Cumulative,
            _ => return Err(Error(format!("unsupported response family `{name}`"))),
        };
        let default = match family {
            Self::Gaussian | Self::Student | Self::Lognormal => Link::Identity,
            Self::Bernoulli | Self::Binomial | Self::Beta | Self::Cumulative => Link::Logit,
            Self::VonMises => Link::TanHalf,
            _ => Link::Log,
        };
        let link = link
            .map(|name| {
                serde_json::from_value(serde_json::Value::String(name.into()))
                    .map_err(|_| Error(format!("unsupported link `{name}`")))
            })
            .transpose()?
            .unwrap_or(default);
        use Link::*;
        let supported = match family {
            Self::Gaussian | Self::Student => !matches!(link, Sqrt | TanHalf),
            Self::Lognormal => matches!(link, Identity | Inverse),
            Self::Bernoulli | Self::Binomial | Self::Beta => {
                matches!(
                    link,
                    Logit | Probit | ProbitApprox | Cloglog | Cauchit | Softit | Identity | Log
                )
            }
            Self::Poisson | Self::Negbinomial | Self::Geometric => {
                matches!(link, Log | Identity | Sqrt | Softplus | Squareplus)
            }
            Self::Gamma | Self::Exponential | Self::Weibull => {
                matches!(link, Log | Identity | Inverse | Softplus | Squareplus)
            }
            Self::VonMises => matches!(link, TanHalf | Identity),
            Self::Cumulative => link == Logit,
        };
        if !supported {
            return Err(Error(format!(
                "link `{}` is not supported by family `{name}`",
                link.name()
            )));
        }
        Ok((family, link))
    }

    pub(crate) fn discrete(self) -> bool {
        matches!(
            self,
            Self::Bernoulli
                | Self::Binomial
                | Self::Poisson
                | Self::Negbinomial
                | Self::Geometric
                | Self::Cumulative
        )
    }

    pub(crate) fn auxiliaries(self) -> &'static [&'static str] {
        match self {
            Self::Gaussian | Self::Lognormal => &["sigma"],
            Self::Student => &["sigma", "nu"],
            Self::Gamma | Self::Negbinomial | Self::Weibull => &["shape"],
            Self::Beta => &["phi"],
            Self::VonMises => &["kappa"],
            _ => &[],
        }
    }

    pub(crate) fn prior_scale_link(self, link: Link) -> Link {
        if self == Self::Lognormal {
            Link::Log
        } else {
            link
        }
    }

    pub(crate) fn default_auxiliary(self, name: &str, sigma_scale: f64) -> String {
        match name {
            "sigma" => format!("student_t(3, 0, {sigma_scale})"),
            "nu" => "gamma(2, 0.1)".into(),
            "shape" if self == Self::Negbinomial => "InverseGamma(0.4, 0.3)".into(),
            "shape" | "phi" => "Gamma(0.01, 0.01)".into(),
            "kappa" => "Gamma(2, 0.01)".into(),
            _ => unreachable!("auxiliary belongs to the family"),
        }
    }

    pub(crate) fn observation(
        self,
        b: &mut Builder<'_>,
        mu: NodeId,
        trials: Option<NodeId>,
        thresholds: usize,
        scalar: bool,
    ) -> NodeId {
        let apply = |b: &mut Builder<'_>, name, args: &[NodeId]| {
            if scalar {
                b.call(name, args)
            } else {
                broadcast(b, name, args)
            }
        };
        match self {
            Self::Gaussian | Self::Lognormal => {
                let sigma = b.self_ref("sigma");
                let law = if self == Self::Lognormal {
                    "LogNormal"
                } else {
                    "Normal"
                };
                apply(b, law, &[mu, sigma])
            }
            Self::Student => {
                let nu = b.self_ref("nu");
                let standard = b.call("StudentT", &[nu]);
                let sigma = b.self_ref("sigma");
                apply(b, "locscale", &[standard, mu, sigma])
            }
            Self::Bernoulli => apply(b, "Bernoulli", &[mu]),
            Self::Binomial => {
                let n = trials.expect("binomial trials were validated");
                apply(b, "Binomial", &[n, mu])
            }
            Self::Poisson => apply(b, "Poisson", &[mu]),
            Self::Negbinomial | Self::Geometric | Self::Gamma => {
                let shape = if self == Self::Geometric {
                    b.lit_real(1.0)
                } else {
                    b.self_ref("shape")
                };
                let rate = apply(b, "divide", &[shape, mu]);
                let law = if self == Self::Gamma {
                    "Gamma"
                } else {
                    "NegativeBinomial"
                };
                apply(b, law, &[shape, rate])
            }
            Self::Exponential => {
                let one = b.lit_real(1.0);
                let rate = apply(b, "divide", &[one, mu]);
                apply(b, "Exponential", &[rate])
            }
            Self::Weibull => {
                let shape = b.self_ref("shape");
                let one = b.lit_real(1.0);
                let inverse = b.call("divide", &[one, shape]);
                let argument = b.call("add", &[one, inverse]);
                let gamma = b.call("gamma", &[argument]);
                let scale = apply(b, "divide", &[mu, gamma]);
                apply(b, "Weibull", &[shape, scale])
            }
            Self::Beta => {
                let phi = b.self_ref("phi");
                let alpha = apply(b, "mul", &[mu, phi]);
                let one = b.lit_real(1.0);
                let complement = apply(b, "sub", &[one, mu]);
                let beta = apply(b, "mul", &[complement, phi]);
                apply(b, "Beta", &[alpha, beta])
            }
            Self::VonMises => {
                let kappa = b.self_ref("kappa");
                apply(b, "VonMises", &[mu, kappa])
            }
            Self::Cumulative => crate::ordinal::observation(b, mu, thresholds),
        }
    }
}

impl Link {
    pub fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Log => "log",
            Self::Inverse => "inverse",
            Self::Sqrt => "sqrt",
            Self::Softplus => "softplus",
            Self::Squareplus => "squareplus",
            Self::Logit => "logit",
            Self::Probit => "probit",
            Self::ProbitApprox => "probit_approx",
            Self::Cloglog => "cloglog",
            Self::Cauchit => "cauchit",
            Self::Softit => "softit",
            Self::TanHalf => "tan_half",
        }
    }

    pub(crate) fn uses_response_scale(self) -> bool {
        matches!(
            self,
            Self::Identity | Self::Log | Self::Inverse | Self::Sqrt
        )
    }

    pub(crate) fn inverse(self, b: &mut Builder<'_>, eta: NodeId) -> NodeId {
        match self {
            Self::Identity => eta,
            Self::Log => broadcast(b, "exp", &[eta]),
            Self::Logit => broadcast(b, "invlogit", &[eta]),
            Self::Probit => broadcast(b, "invprobit", &[eta]),
            Self::Inverse => {
                let one = b.lit_real(1.0);
                broadcast(b, "divide", &[one, eta])
            }
            Self::Sqrt => broadcast(b, "mul", &[eta, eta]),
            Self::Softplus => softplus(b, eta),
            Self::Squareplus => {
                let square = broadcast(b, "mul", &[eta, eta]);
                let four = b.lit_real(4.0);
                let sum = broadcast(b, "add", &[square, four]);
                let root = broadcast(b, "sqrt", &[sum]);
                let sum = broadcast(b, "add", &[eta, root]);
                let two = b.lit_real(2.0);
                broadcast(b, "divide", &[sum, two])
            }
            Self::ProbitApprox => {
                let square = broadcast(b, "mul", &[eta, eta]);
                let a = b.lit_real(0.07056);
                let cubic = broadcast(b, "mul", &[square, a]);
                let c = b.lit_real(1.5976);
                let inner = broadcast(b, "add", &[cubic, c]);
                let scaled = broadcast(b, "mul", &[eta, inner]);
                broadcast(b, "invlogit", &[scaled])
            }
            Self::Cloglog => {
                let exp = broadcast(b, "exp", &[eta]);
                let neg = broadcast(b, "neg", &[exp]);
                let exp = broadcast(b, "expm1", &[neg]);
                broadcast(b, "neg", &[exp])
            }
            Self::Cauchit => {
                let atan = broadcast(b, "atan", &[eta]);
                let pi = b.call_head("pi");
                let ratio = broadcast(b, "divide", &[atan, pi]);
                let half = b.lit_real(0.5);
                broadcast(b, "add", &[ratio, half])
            }
            Self::Softit => {
                let value = softplus(b, eta);
                let one = b.lit_real(1.0);
                let denominator = broadcast(b, "add", &[one, value]);
                broadcast(b, "divide", &[value, denominator])
            }
            Self::TanHalf => {
                let atan = broadcast(b, "atan", &[eta]);
                let two = b.lit_real(2.0);
                broadcast(b, "mul", &[two, atan])
            }
        }
    }
}

fn softplus(b: &mut Builder<'_>, eta: NodeId) -> NodeId {
    let magnitude = broadcast(b, "abs", &[eta]);
    // This form avoids overflow and gives derivative 1/2 at zero regardless
    // of the backend's choice of derivative for abs at its cusp.
    let half = b.lit_real(0.5);
    let eta_half = broadcast(b, "mul", &[eta, half]);
    let magnitude_half = broadcast(b, "mul", &[magnitude, half]);
    let positive = broadcast(b, "add", &[eta_half, magnitude_half]);
    let negative = broadcast(b, "neg", &[magnitude]);
    let exp = broadcast(b, "exp", &[negative]);
    let remainder = broadcast(b, "log1p", &[exp]);
    broadcast(b, "add", &[positive, remainder])
}

pub(crate) fn default_scale(link: Link, response: Option<&[f64]>) -> (f64, f64) {
    let Some(response) = response.filter(|_| link.uses_response_scale()) else {
        return (0.0, 2.5);
    };
    let values: Vec<_> = response
        .iter()
        .map(|&y| match link {
            Link::Log => (if y == 0.0 { 0.1 } else { y }).ln(),
            Link::Inverse => (if y == 0.0 { 0.1 } else { y }).recip(),
            Link::Sqrt => y.sqrt(),
            _ => y,
        })
        .collect();
    let location = median(values.clone());
    let mad = 1.4826 * median(values.iter().map(|y| (y - location).abs()).collect());
    let location = round_decimal(location);
    let scale = round_decimal(mad);
    (
        if location.is_finite() { location } else { 0.0 },
        if scale.is_finite() {
            scale.max(2.5)
        } else {
            2.5
        },
    )
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    }
}

// R rounds the input binary value, rather than a possibly rounded x*10.
fn round_decimal(value: f64) -> f64 {
    let x = value.abs();
    let index = (x * 10.0).floor();
    let lower = index / 10.0;
    let upper = (index + 1.0) / 10.0;
    let distance = x - lower;
    let rounded = if distance > upper - x || (distance == upper - x && index % 2.0 == 1.0) {
        upper
    } else {
        lower
    };
    rounded.copysign(value)
}
