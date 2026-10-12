use crate::{Error, Result, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};
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
    #[serde(rename = "zero_inflated_binomial")]
    ZeroInflatedBinomial,
    Poisson,
    #[serde(rename = "zero_inflated_poisson")]
    ZeroInflatedPoisson,
    Negbinomial,
    #[serde(rename = "zero_inflated_negbinomial")]
    ZeroInflatedNegbinomial,
    Geometric,
    #[serde(rename = "Gamma")]
    Gamma,
    Exponential,
    Weibull,
    Beta,
    #[serde(rename = "zero_inflated_beta")]
    ZeroInflatedBeta,
    #[serde(rename = "von_mises")]
    VonMises,
    Cumulative,
    Categorical,
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
    Logm1,
}

#[derive(Clone, Copy)]
pub(crate) struct Observation<'a> {
    pub mu: NodeId,
    pub trials: Option<NodeId>,
    pub scale: Option<NodeId>,
    pub auxiliary: &'a [(&'a str, NodeId)],
}

impl Family {
    pub(crate) fn resolve(name: &str, link: Option<&str>) -> Result<(Self, Link)> {
        let family = match name {
            "gaussian" => Self::Gaussian,
            "student" => Self::Student,
            "lognormal" => Self::Lognormal,
            "bernoulli" => Self::Bernoulli,
            "binomial" => Self::Binomial,
            "zero_inflated_binomial" => Self::ZeroInflatedBinomial,
            "poisson" => Self::Poisson,
            "zero_inflated_poisson" => Self::ZeroInflatedPoisson,
            "negbinomial" => Self::Negbinomial,
            "zero_inflated_negbinomial" => Self::ZeroInflatedNegbinomial,
            "geometric" => Self::Geometric,
            "Gamma" | "gamma" => Self::Gamma,
            "exponential" => Self::Exponential,
            "weibull" => Self::Weibull,
            "beta" => Self::Beta,
            "zero_inflated_beta" => Self::ZeroInflatedBeta,
            "von_mises" => Self::VonMises,
            "cumulative" => Self::Cumulative,
            "categorical" => Self::Categorical,
            _ => return Err(Error(format!("unsupported response family `{name}`"))),
        };
        let default = match family {
            Self::Gaussian | Self::Student | Self::Lognormal => Link::Identity,
            Self::Bernoulli
            | Self::Binomial
            | Self::ZeroInflatedBinomial
            | Self::Beta
            | Self::ZeroInflatedBeta
            | Self::Cumulative
            | Self::Categorical => Link::Logit,
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
            Self::Gaussian | Self::Student => !matches!(link, Sqrt | TanHalf | Logm1),
            Self::Lognormal => matches!(link, Identity | Inverse),
            Self::Bernoulli
            | Self::Binomial
            | Self::ZeroInflatedBinomial
            | Self::Beta
            | Self::ZeroInflatedBeta => {
                matches!(
                    link,
                    Logit | Probit | ProbitApprox | Cloglog | Cauchit | Softit | Identity | Log
                )
            }
            Self::Poisson
            | Self::ZeroInflatedPoisson
            | Self::Negbinomial
            | Self::ZeroInflatedNegbinomial
            | Self::Geometric => {
                matches!(link, Log | Identity | Sqrt | Softplus | Squareplus)
            }
            Self::Gamma | Self::Exponential | Self::Weibull => {
                matches!(link, Log | Identity | Inverse | Softplus | Squareplus)
            }
            Self::VonMises => matches!(link, TanHalf | Identity),
            Self::Cumulative | Self::Categorical => link == Logit,
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
                | Self::ZeroInflatedBinomial
                | Self::Poisson
                | Self::ZeroInflatedPoisson
                | Self::Negbinomial
                | Self::ZeroInflatedNegbinomial
                | Self::Geometric
                | Self::Cumulative
                | Self::Categorical
        )
    }

    pub(crate) fn uses_trials(self) -> bool {
        matches!(self, Self::Binomial | Self::ZeroInflatedBinomial)
    }

    pub(crate) fn auxiliaries(self) -> &'static [&'static str] {
        match self {
            Self::Gaussian | Self::Lognormal => &["sigma"],
            Self::Student => &["sigma", "nu"],
            Self::Gamma | Self::Negbinomial | Self::Weibull => &["shape"],
            Self::Beta => &["phi"],
            Self::ZeroInflatedBeta => &["phi", "zi"],
            Self::VonMises => &["kappa"],
            Self::ZeroInflatedPoisson | Self::ZeroInflatedBinomial => &["zi"],
            Self::ZeroInflatedNegbinomial => &["shape", "zi"],
            _ => &[],
        }
    }

    pub(crate) fn auxiliary_link(self, name: &str, link: Option<&str>) -> Result<Link> {
        if !self.auxiliaries().contains(&name) {
            return Err(Error(format!(
                "family has no distributional parameter `{name}`"
            )));
        }
        let default = match name {
            "nu" => Link::Logm1,
            "zi" => Link::Logit,
            _ => Link::Log,
        };
        let link = link
            .map(|name| {
                serde_json::from_value(serde_json::Value::String(name.into()))
                    .map_err(|_| Error(format!("unsupported link `{name}`")))
            })
            .transpose()?
            .unwrap_or(default);
        let supported = link == Link::Identity
            || link == default
            || (default == Link::Log && matches!(link, Link::Softplus | Link::Squareplus));
        if !supported {
            return Err(Error(format!(
                "link `{}` is not supported by parameter `{name}`",
                link.name()
            )));
        }
        Ok(link)
    }

    pub(crate) fn default_predictor_prior(self, name: &str, link: Link, scale: f64) -> String {
        if name == "sigma" || link == Link::Identity {
            return self.default_auxiliary(name, scale);
        }
        match name {
            "nu" => "normal(2.7, 0.8)".into(),
            "kappa" => "normal(5.0, 0.8)".into(),
            "zi" => "logistic(0, 1)".into(),
            _ => "student_t(3, 0, 2.5)".into(),
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
            "shape" if matches!(self, Self::Negbinomial | Self::ZeroInflatedNegbinomial) => {
                "InverseGamma(0.4, 0.3)".into()
            }
            "shape" | "phi" => "Gamma(0.01, 0.01)".into(),
            "kappa" => "Gamma(2, 0.01)".into(),
            "zi" => "Beta(1, 1)".into(),
            _ => unreachable!("auxiliary belongs to the family"),
        }
    }

    pub(crate) fn observation(
        self,
        b: &mut Builder<'_>,
        observation: Observation<'_>,
        thresholds: usize,
        scalar: bool,
    ) -> NodeId {
        let Observation {
            mu,
            trials,
            scale,
            auxiliary,
        } = observation;
        if (!auxiliary.is_empty()
            || matches!(
                self,
                Self::Student
                    | Self::ZeroInflatedPoisson
                    | Self::ZeroInflatedNegbinomial
                    | Self::ZeroInflatedBinomial
                    | Self::ZeroInflatedBeta
            ))
            && !scalar
        {
            let local = Ref {
                ns: RefNs::Local,
                name: b.sym("_mu_"),
            };
            let mean = if self == Self::Categorical {
                mu
            } else {
                b.m.alloc(Node::Ref(local))
            };
            let mut inputs = Vec::new();
            let mut args = Vec::new();
            if self != Self::Categorical {
                inputs.push((b.sym("mu"), local));
                args.push(mu);
            }
            let trials = trials.map(|value| {
                let local = Ref {
                    ns: RefNs::Local,
                    name: b.sym("_trials_"),
                };
                inputs.push((b.sym("trials"), local));
                args.push(value);
                b.m.alloc(Node::Ref(local))
            });
            let mut argument = |b: &mut Builder<'_>, name: &str, value| {
                let local = Ref {
                    ns: RefNs::Local,
                    name: b.sym(&format!("_{name}_")),
                };
                inputs.push((b.sym(name), local));
                args.push(value);
                b.m.alloc(Node::Ref(local))
            };
            let scale = scale.map(|value| argument(b, "scale", value));
            let auxiliary: Vec<_> = auxiliary
                .iter()
                .map(|&(name, value)| (name, argument(b, name, value)))
                .collect();
            let law = self.observation(
                b,
                Observation {
                    mu: mean,
                    trials,
                    scale,
                    auxiliary: &auxiliary,
                },
                thresholds,
                true,
            );
            let head = b.sym("functionof");
            let kernel = b.m.alloc(Node::Call(Call {
                head: CallHead::Builtin(head),
                args: vec![law].into(),
                named: Box::default(),
                inputs: Some(Inputs::Spec(inputs.into())),
            }));
            let name = b.alloc_name("response_law");
            b.bind(&name, kernel);
            let kernel = b.self_ref(&name);
            args.insert(0, kernel);
            return b.call("broadcast", &args);
        }
        let apply = |b: &mut Builder<'_>, name, args: &[NodeId]| {
            if scalar {
                b.call(name, args)
            } else {
                broadcast(b, name, args)
            }
        };
        let auxiliary = |b: &mut Builder<'_>, name: &str| {
            auxiliary
                .iter()
                .find(|(key, _)| *key == name)
                .map_or_else(|| b.self_ref(name), |(_, value)| *value)
        };
        match self {
            Self::Gaussian | Self::Lognormal => {
                let sigma = scale.unwrap_or_else(|| auxiliary(b, "sigma"));
                let law = if self == Self::Lognormal {
                    "LogNormal"
                } else {
                    "Normal"
                };
                apply(b, law, &[mu, sigma])
            }
            Self::Student => {
                let nu = auxiliary(b, "nu");
                let standard = b.call("StudentT", &[nu]);
                let sigma = scale.unwrap_or_else(|| auxiliary(b, "sigma"));
                crate::prior_expr::positive_locscale(b, standard, mu, sigma)
            }
            Self::Bernoulli => apply(b, "Bernoulli", &[mu]),
            Self::Binomial => {
                let n = trials.expect("binomial trials were validated");
                apply(b, "Binomial", &[n, mu])
            }
            Self::Poisson => apply(b, "Poisson", &[mu]),
            Self::ZeroInflatedPoisson
            | Self::ZeroInflatedNegbinomial
            | Self::ZeroInflatedBinomial
            | Self::ZeroInflatedBeta => {
                let base = match self {
                    Self::ZeroInflatedPoisson => Self::Poisson,
                    Self::ZeroInflatedNegbinomial => Self::Negbinomial,
                    Self::ZeroInflatedBeta => Self::Beta,
                    _ => Self::Binomial,
                };
                let base = base.observation(b, observation, thresholds, true);
                let zi = auxiliary(b, "zi");
                let zero = if self.discrete() {
                    b.lit_int(0)
                } else {
                    b.lit_real(0.0)
                };
                let atom = b.call("Dirac", &[zero]);
                let atom = b.call("weighted", &[zi, atom]);
                let one = b.lit_real(1.0);
                let complement = b.call("sub", &[one, zi]);
                let component = b.call("weighted", &[complement, base]);
                let law = b.call("superpose", &[atom, component]);
                b.call("normalize", &[law])
            }
            Self::Negbinomial | Self::Geometric | Self::Gamma => {
                let shape = if self == Self::Geometric {
                    b.lit_real(1.0)
                } else {
                    auxiliary(b, "shape")
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
                let shape = auxiliary(b, "shape");
                let one = b.lit_real(1.0);
                let inverse = b.call("divide", &[one, shape]);
                let argument = b.call("add", &[one, inverse]);
                let gamma = b.call("gamma", &[argument]);
                let scale = apply(b, "divide", &[mu, gamma]);
                apply(b, "Weibull", &[shape, scale])
            }
            Self::Beta => {
                let phi = auxiliary(b, "phi");
                let alpha = apply(b, "mul", &[mu, phi]);
                let one = b.lit_real(1.0);
                let complement = apply(b, "sub", &[one, mu]);
                let beta = apply(b, "mul", &[complement, phi]);
                apply(b, "Beta", &[alpha, beta])
            }
            Self::VonMises => {
                let kappa = auxiliary(b, "kappa");
                apply(b, "VonMises", &[mu, kappa])
            }
            Self::Cumulative => crate::ordinal::observation(b, mu, thresholds),
            Self::Categorical => {
                let mut logits = vec![b.lit_real(0.0)];
                logits.extend(observation.auxiliary.iter().map(|(_, value)| *value));
                let logits = b.array(&logits);
                let probabilities = b.call("softmax", &[logits]);
                b.call("Categorical", &[probabilities])
            }
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
            Self::Logm1 => "logm1",
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
            Self::Logm1 => {
                let exponential = broadcast(b, "exp", &[eta]);
                let one = b.lit_real(1.0);
                broadcast(b, "add", &[one, exponential])
            }
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
