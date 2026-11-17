//! Phase-1 mapping tables: FlatPPL laws and functions to Stan. A stopgap until
//! the stan-profile-check RON tables land; the lookups below are the only API
//! the emitter uses, so the swap stays local to this file.

/// How a FlatPPL law's canonical arguments become Stan arguments.
pub(crate) enum ArgMap {
    /// The FlatPPL argument at this index.
    Arg(usize),
    /// A literal Stan expression.
    Lit(&'static str),
    /// `p / (1 - p)` of the argument at this index (`Geometric`).
    Odds(usize),
}

pub(crate) struct LawMap {
    pub stan: &'static str,
    pub args: &'static [ArgMap],
    /// Stan accepts vector arguments and a vector variate for this law.
    pub vectorized: bool,
    /// Variate is a vector per draw (`dirichlet`, `multi_normal`, …).
    pub multivariate: bool,
}

use ArgMap::{Arg, Lit, Odds};

macro_rules! law {
    ($stan:literal, [$($a:expr),*], vec = $v:literal, mv = $m:literal) => {
        LawMap { stan: $stan, args: &[$($a),*], vectorized: $v, multivariate: $m }
    };
}

/// The Stan law for a FlatPPL distribution, keyed by name. Argument indices
/// refer to the canonical order of `flatppl_infer::distribution_param_names`.
pub(crate) fn law(name: &str) -> Option<LawMap> {
    Some(match name {
        "Normal" => law!("normal", [Arg(0), Arg(1)], vec = true, mv = false),
        "Cauchy" => law!("cauchy", [Arg(0), Arg(1)], vec = true, mv = false),
        "StudentT" => law!(
            "student_t",
            [Arg(0), Lit("0"), Lit("1")],
            vec = true,
            mv = false
        ),
        "Logistic" => law!("logistic", [Arg(0), Arg(1)], vec = true, mv = false),
        "Laplace" => law!(
            "double_exponential",
            [Arg(0), Arg(1)],
            vec = true,
            mv = false
        ),
        "LogNormal" => law!("lognormal", [Arg(0), Arg(1)], vec = true, mv = false),
        "Exponential" => law!("exponential", [Arg(0)], vec = true, mv = false),
        "Gamma" => law!("gamma", [Arg(0), Arg(1)], vec = true, mv = false),
        "InverseGamma" => law!("inv_gamma", [Arg(0), Arg(1)], vec = true, mv = false),
        "Weibull" => law!("weibull", [Arg(0), Arg(1)], vec = true, mv = false),
        "Beta" => law!("beta", [Arg(0), Arg(1)], vec = true, mv = false),
        "ChiSquared" => law!("chi_square", [Arg(0)], vec = true, mv = false),
        "Rayleigh" => law!("rayleigh", [Arg(0)], vec = true, mv = false),
        // Stan's pareto takes (y_min, alpha): FlatPPL's (shape, scale) swap.
        "Pareto" => law!("pareto", [Arg(1), Arg(0)], vec = true, mv = false),
        "VonMises" => law!("von_mises", [Arg(0), Arg(1)], vec = true, mv = false),
        "Bernoulli" => law!("bernoulli", [Arg(0)], vec = true, mv = false),
        "Binomial" => law!("binomial", [Arg(0), Arg(1)], vec = true, mv = false),
        "Poisson" => law!("poisson", [Arg(0)], vec = true, mv = false),
        "NegativeBinomial" => law!("neg_binomial", [Arg(0), Arg(1)], vec = true, mv = false),
        "NegativeBinomial2" => law!("neg_binomial_2", [Arg(0), Arg(1)], vec = true, mv = false),
        // Failures before the first success: neg_binomial with alpha = 1 and
        // beta = p / (1 - p) has the same support {0, 1, …} and pmf.
        "Geometric" => law!("neg_binomial", [Lit("1"), Odds(0)], vec = true, mv = false),
        "Categorical" => law!("categorical", [Arg(0)], vec = false, mv = false),
        "Dirichlet" => law!("dirichlet", [Arg(0)], vec = false, mv = true),
        "MvNormal" => law!("multi_normal", [Arg(0), Arg(1)], vec = false, mv = true),
        // Stan infers the count from the variate.
        "Multinomial" => law!("multinomial", [Arg(1)], vec = false, mv = true),
        _ => return None,
    })
}

/// A FlatPPL function with a same-arity Stan function. The flag says whether
/// Stan applies it elementwise to a vector argument.
pub(crate) fn function(name: &str) -> Option<(&'static str, bool)> {
    Some(match name {
        "exp" => ("exp", true),
        "log" => ("log", true),
        "log10" => ("log10", true),
        "sqrt" => ("sqrt", true),
        "abs" => ("abs", true),
        "sin" => ("sin", true),
        "cos" => ("cos", true),
        "tan" => ("tan", true),
        "asin" => ("asin", true),
        "acos" => ("acos", true),
        "atan" => ("atan", true),
        "sinh" => ("sinh", true),
        "cosh" => ("cosh", true),
        "tanh" => ("tanh", true),
        "asinh" => ("asinh", true),
        "acosh" => ("acosh", true),
        "atanh" => ("atanh", true),
        "log1p" => ("log1p", true),
        "expm1" => ("expm1", true),
        "floor" => ("floor", true),
        "ceil" => ("ceil", true),
        "gamma" => ("tgamma", true),
        "loggamma" => ("lgamma", true),
        "logit" => ("logit", true),
        "invlogit" => ("inv_logit", true),
        "probit" => ("inv_Phi", true),
        "invprobit" => ("Phi", true),
        "atan2" => ("atan2", false),
        "sum" => ("sum", false),
        "prod" => ("prod", false),
        "mean" => ("mean", false),
        "var" => ("variance", false),
        "std" => ("sd", false),
        "maximum" => ("max", false),
        "minimum" => ("min", false),
        "cumsum" => ("cumulative_sum", false),
        "logsumexp" => ("log_sum_exp", false),
        "softmax" => ("softmax", false),
        "det" => ("determinant", false),
        "inv" => ("inverse", false),
        "trace" => ("trace", false),
        "logabsdet" => ("log_determinant", false),
        "lower_cholesky" => ("cholesky_decompose", false),
        "diagmat" => ("diag_matrix", false),
        "diag" => ("diagonal", false),
        "quadform" => ("quad_form", false),
        "eye" => ("identity_matrix", false),
        _ => return None,
    })
}

/// FlatPPL functions with no Stan counterpart, with what Stan offers instead.
pub(crate) const NO_STAN_FUNCTION: &[(&str, &str)] = &[(
    "polygamma",
    "Stan has `digamma` and `trigamma` for orders 0 and 1 only",
)];

/// Stan keywords and reserved names; a FlatPPL name in this list gets a
/// trailing underscore.
pub(crate) const RESERVED: &[&str] = &[
    "for",
    "in",
    "while",
    "repeat",
    "until",
    "if",
    "then",
    "else",
    "true",
    "false",
    "target",
    "functions",
    "model",
    "data",
    "parameters",
    "quantities",
    "transformed",
    "generated",
    "int",
    "real",
    "complex",
    "vector",
    "row_vector",
    "matrix",
    "array",
    "tuple",
    "simplex",
    "ordered",
    "positive_ordered",
    "unit_vector",
    "sum_to_zero_vector",
    "sum_to_zero_matrix",
    "cholesky_factor_corr",
    "cholesky_factor_cov",
    "corr_matrix",
    "cov_matrix",
    "column_stochastic_matrix",
    "row_stochastic_matrix",
    "void",
    "return",
    "break",
    "continue",
    "print",
    "reject",
    "fatal_error",
    "lower",
    "upper",
    "offset",
    "multiplier",
    "profile",
    "struct",
    "typedef",
    "export",
    "auto",
    "extern",
    "var",
    "static",
    "lp__",
    "jacobian",
];
