//! Call-arity rules: §07 argument counts for builtins whose parameter list the
//! catalogue declares, and declared-parameter counts for user-defined callables.
//!
//! Before these rules a mis-arity call typed silently — every rule arm that
//! indexes fixed argument positions ignored extras and defaulted a missing one,
//! so `exp(1.0, 2.0)` was `real` and `log()` was `real`.

use flatppl_infer::{Diagnostic, Severity, infer};

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    flatppl_flatpir::write(&m)
}

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d: &Diagnostic| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// The six leaks that motivated the rule. Each typed a concrete scalar (or
/// `%deferred`) before; each is `%failed` now, with the callee and its declared
/// count named.
#[test]
fn measured_leaks_are_now_static_errors() {
    for (src, want) in [
        ("x = add(1.0)", "`add` takes 2 arguments (spec §07), got 1"),
        (
            "x = add(1.0, 2.0, 3.0)",
            "`add` takes 2 arguments (spec §07), got 3",
        ),
        (
            "x = exp(1.0, 2.0)",
            "`exp` takes 1 argument (spec §07), got 2",
        ),
        (
            "x = sqrt(1.0, 2.0, 3.0)",
            "`sqrt` takes 1 argument (spec §07), got 3",
        ),
        (
            "x = lengthof([1.0], 2.0)",
            "`lengthof` takes 1 argument (spec §07), got 2",
        ),
        ("x = log()", "`log` takes 1 argument (spec §07), got 0"),
    ] {
        assert_eq!(errors(src), vec![want.to_string()], "for {src}");
        assert!(
            ir(src).contains("(%failed"),
            "{src} must type %failed:\n{}",
            ir(src)
        );
    }
}

/// A well-formed call is untouched: no error, and the type the op's own rule
/// produces.
#[test]
fn correct_arity_still_types() {
    assert!(errors("x = exp(1.0)").is_empty());
    assert!(errors("x = add(1.0, 2.0)").is_empty());
    assert!(errors("x = ifelse(true, 1.0, 2.0)").is_empty());
    assert!(errors("a = [1.0, 2.0]\nx = lengthof(a)").is_empty());
}

/// §07 "Linear algebra": "when called as `diag(A)`, `k` defaults to `0`" — both
/// spellings pass, a third argument does not.
#[test]
fn optional_parameter_admits_both_spellings() {
    let m = "M = rowstack([[1.0, 2.0], [3.0, 4.0]])\n";
    assert!(errors(&format!("{m}x = diag(M)")).is_empty());
    assert!(errors(&format!("{m}x = diag(M, 1)")).is_empty());
    assert_eq!(
        errors(&format!("{m}x = diag(M, 1, 2)")),
        vec!["`diag` takes 1 or 2 arguments (spec §07), got 3".to_string()]
    );
}

/// §07: `builtin_sample(rngstate, kernel, kernel_input, n, m, ...)` — "or a
/// scalar `X` if no `n, m, ...` are given", so three is a minimum. `get`'s
/// selector list is variadic the same way.
#[test]
fn variadic_parameter_has_a_minimum_but_no_maximum() {
    // §07 "Random value generation": `rnginit`'s seed is a byte vector.
    let s = "state = rnginit([1, 2, 3])\nk = record(mu = 0.0, sigma = 1.0)\n";
    assert!(errors(&format!("{s}xs, s2 = builtin_sample(state, Normal, k)")).is_empty());
    assert!(
        errors(&format!(
            "{s}xs, s2 = builtin_sample(state, Normal, k, 4, 5)"
        ))
        .is_empty()
    );
    assert_eq!(
        errors(&format!("{s}xs, s2 = builtin_sample(state, Normal)")),
        vec!["`builtin_sample` takes at least 3 arguments (spec §07), got 2".to_string()]
    );

    let v = "v = [1.0, 2.0, 3.0]\n";
    assert!(errors(&format!("{v}x = get(v, 1)")).is_empty());
    assert_eq!(
        errors(&format!("{v}x = get(v)")),
        vec!["`get` takes at least 2 arguments (spec §07), got 1".to_string()]
    );
}

/// §07 "Array and table operations" makes `cat(scalar1, scalar2, ...)`
/// "Equivalent to `vector(scalar1, scalar2, ...)`", and array literals lower to
/// real `vector` calls — `[1.0]` to `(vector 1.0)` and `[]` to `(vector)`. So the
/// shared `x1, x2, ...` notation admits one argument, and reading it as "at least
/// two" for `cat` would reject `cat(1.0)` while accepting the equivalent
/// `vector(1.0)`.
#[test]
fn cat_and_vector_read_the_same_notation_the_same_way() {
    assert!(errors("v = cat(1.0)").is_empty());
    assert!(errors("v = cat(1.0, 2.0)").is_empty());
    assert!(errors("v = vector(1.0)").is_empty());
    assert!(errors("v = vector()").is_empty());
    // The surface spellings that lower to those `vector` calls.
    assert!(errors("v = [1.0]").is_empty());
    assert!(errors("v = []").is_empty());
    // `cat` still needs something to concatenate.
    assert_eq!(
        errors("v = cat()"),
        vec!["`cat` takes at least 1 argument (spec §07), got 0".to_string()]
    );
}

/// Named arguments count toward the total, so the keyword spelling of a §07
/// parameter is not a missing argument.
#[test]
fn named_arguments_count_toward_arity() {
    assert!(errors("n = 3\nx = checked(n, condition = n > 0)").is_empty());
    assert!(errors("n = 3\nx = checked(value = n, condition = n > 0)").is_empty());
    assert_eq!(
        errors("n = 3\nx = checked(n, condition = n > 0, extra = 1.0)"),
        vec!["`checked` takes 2 arguments (spec §07), got 3".to_string()]
    );
}

/// §04 scopes auto-splatting to "built-in or user defined value functions,
/// constructors or transition kernels", and its `fchain` paragraph relies on it
/// ("if `f1` returns a record and `f2` accepts keyword arguments matching the
/// record fields, the two functions compose directly"), so a USER call counts a
/// sole record argument by its fields too. Two corpus models are this shape —
/// `simple-transport2.flatppl`'s `k_model(glob_pars)` and `feature-test1
/// .flatppl`'s `forward_kernel(rand_pars)` — and both were falsely rejected
/// while the user-call path counted a sole record as one argument.
#[test]
fn a_user_call_splats_a_sole_record_argument_too() {
    let f = "lin(a, b, mu) = add(a, add(b, mu))\n\
             pars = record(a = 1.0, b = 2.0, mu = 3.0)\n";
    assert!(
        errors(&format!("{f}y = lin(pars)")).is_empty(),
        "a 3-field record fills three parameters by splatting"
    );
    // The lambda spelling `k_model` uses, whose boundary the sugar emits.
    let g = "pars = record(a = 1.0, b = 2.0, mu = 3.0)\n\
             k = (a, b, mu) -> add(a, add(b, mu))\n";
    assert!(
        errors(&format!("{g}y = k(pars)")).is_empty(),
        "the lambda's `%specinputs` boundary is three parameters, and the record fills them"
    );
    // A record whose field count matches nothing still fails, at the splat count.
    assert_eq!(
        errors(&format!("{f}two = record(a = 1.0, b = 2.0)\ny = lin(two)")),
        vec!["`lin` declares 3 parameters, got 2 arguments".to_string()]
    );
}

/// §04 gives a sole record argument two readings and does not disambiguate them
/// for a positional call, so a callee EITHER reading satisfies is not a mis-arity
/// call. Both readings are live in the corpus: `simple-transport1.flatppl`'s
/// `generator = kernelof(x, pars = pars)` takes a record as its ONE `pars`
/// parameter, while `k_model` splats one of the same shape into three. Preferring
/// the splat unconditionally falsely rejects the former.
#[test]
fn a_sole_record_may_also_be_the_one_ordinary_argument() {
    let src = "pars = record(a = 1.0, b = 2.0, mu = 3.0)\n\
               takes_a_record(p) = get(p, [\"a\"])\n\
               y = takes_a_record(pars)";
    assert!(
        errors(src).is_empty(),
        "a 3-field record is also a valid single argument to a 1-parameter callable"
    );
}

/// Applying a user-defined callable at the wrong arity is a static error naming
/// the callee, instead of typing through `substituted_result`'s
/// bind-what-you-can and reaching the determiniser as a `ResidualUserCall`.
#[test]
fn user_call_arity_is_checked_against_the_declared_parameters() {
    let f = "scale(x) = mul(x, 2.0)\n";
    assert!(errors(&format!("{f}s = scale(1.5)")).is_empty());
    assert_eq!(
        errors(&format!("{f}s = scale(1.5, 3.0)")),
        vec!["`scale` declares 1 parameter, got 2 arguments".to_string()]
    );
    assert_eq!(
        errors(&format!("{f}s = scale()")),
        vec!["`scale` declares 1 parameter, got 0 arguments".to_string()]
    );

    let g = "lin(a, b, x) = add(a, mul(b, x))\n";
    assert!(errors(&format!("{g}y = lin(1.0, 2.0, 3.0)")).is_empty());
    assert!(
        errors(&format!("{g}y = lin(a = 1.0, b = 2.0, x = 3.0)")).is_empty(),
        "keyword application of every parameter must pass"
    );
    assert_eq!(
        errors(&format!("{g}y = lin(1.0, 2.0)")),
        vec!["`lin` declares 3 parameters, got 2 arguments".to_string()]
    );
}

/// §08 "Built-in distributions": "The names and order of the distribution
/// parameters specified below define the names and positional order of the
/// kernel arguments." No §08 entry gives a parameter a default, so a
/// constructor's arity is exactly its declared count — over- and under-supply
/// are both static errors naming the constructor and that count.
#[test]
fn distribution_constructor_arity_is_the_declared_parameter_count() {
    for (src, want) in [
        (
            "m = Normal(0.0, 1.0, 2.0)",
            "`Normal` takes 2 arguments (spec §08), got 3",
        ),
        (
            "m = Normal(0.0)",
            "`Normal` takes 2 arguments (spec §08), got 1",
        ),
        (
            "m = StudentT(3.0, 1.0)",
            "`StudentT` takes 1 argument (spec §08), got 2",
        ),
        (
            "m = GeneralizedNormal(0.0, 1.0)",
            "`GeneralizedNormal` takes 3 arguments (spec §08), got 2",
        ),
        (
            "m = Poisson()",
            "`Poisson` takes 1 argument (spec §08), got 0",
        ),
    ] {
        assert_eq!(errors(src), vec![want.to_string()], "for {src}");
        assert!(
            ir(src).contains("(%failed"),
            "{src} must type %failed:\n{}",
            ir(src)
        );
    }
}

/// Every spelling §04 admits for a built-in constructor passes: positional,
/// keyword, and mixed. §04 "Calling conventions": "All built-in ordinary
/// callables have a defined input order and accept both positional and keyword
/// arguments."
#[test]
fn every_constructor_call_spelling_passes_at_the_declared_count() {
    assert!(errors("m = Normal(0.0, 1.0)").is_empty());
    assert!(errors("m = Normal(mu = 0.0, sigma = 1.0)").is_empty());
    assert!(
        errors("m = Normal(sigma = 1.0, mu = 0.0)").is_empty(),
        "keyword order is not significant (spec §04)"
    );
    assert!(errors("m = Normal(0.0, sigma = 1.0)").is_empty());
    assert!(errors("m = Multinomial(3, [0.2, 0.8])").is_empty());
}

/// §04 "Calling conventions": "`f(record(a = x, b = y, ...))` … are equivalent
/// to `f(a = x, b = y, ...)`", so a sole record argument supplies one argument
/// per field. Counting it as one would reject the splat spelling of every
/// multi-parameter constructor.
#[test]
fn a_sole_record_argument_supplies_one_argument_per_field() {
    assert!(errors("m = Normal(record(mu = 0.0, sigma = 1.0))").is_empty());
    assert!(
        errors("p = record(mu = 0.0, sigma = 1.0)\nm = Normal(p)").is_empty(),
        "the splat is about the argument being a record, not about spelling `record(…)` inline"
    );
    // The field count is what is checked, so a wrong-width record still fails.
    assert_eq!(
        errors("m = Normal(record(mu = 0.0, sigma = 1.0, tau = 2.0))"),
        vec!["`Normal` takes 2 arguments (spec §08), got 3".to_string()]
    );
    // §04: auto-splatting fires only for a SOLE argument, so a record alongside
    // other arguments counts as the one ordinary value it is — three here, not
    // four.
    assert_eq!(
        errors("m = Normal(record(mu = 0.0, sigma = 1.0), 1.0, 2.0)"),
        vec!["`Normal` takes 2 arguments (spec §08), got 3".to_string()]
    );
}

/// §07's "Domains" column for the elementary functions lists `reals` and
/// `complexes`, never `integers`: an integer argument is admitted only through
/// §03's `integers ⊂ reals`, so the real-domain function applies and the result
/// is real. `exp(2)` typed `integer` before this row fix, which is what forced
/// the determiniser's value-identity `real()` wrap.
#[test]
fn elementary_functions_of_an_integer_are_real() {
    for src in [
        "x = exp(2)",
        "x = log(2)",
        "x = sqrt(4)",
        "x = sin(1)",
        "x = loggamma(3)",
        "x = conj(2)",
    ] {
        let out = ir(src);
        assert!(
            out.contains("(%scalar real)") && !out.contains("(%scalar integer)"),
            "{src} must be a real scalar; got:\n{out}"
        );
    }
    // The complex path is unchanged.
    let out = ir("x = exp(complex(1.0, 2.0))");
    assert!(
        out.contains("(%scalar complex)"),
        "exp of a complex stays complex; got:\n{out}"
    );
}

/// §04 "Calling conventions" binds keyword arguments by name — "Arguments are
/// bound to inputs by name, the order of the arguments is not relevant" — and
/// states the rule outright for the splat form: "A call with field or column
/// names that do not match the callable's argument names is a static error."
/// Checking the count alone let `Normal(mu = 0.0, tau = 1.0)` through and
/// determinize to `builtin_logdensityof(Normal, record(mu = 0.0, tau = 1.0), …)`
/// — a nonexistent `tau` and a missing `sigma` handed to the engine.
#[test]
fn a_named_argument_must_be_a_declared_parameter() {
    assert_eq!(
        errors("m = Normal(mu = 0.0, tau = 1.0)"),
        vec!["`Normal` has no parameter `tau` (spec §08 parameters: `mu`, `sigma`)".to_string()]
    );
    // Every unbindable name is reported, not just the first.
    assert_eq!(
        errors("m = Normal(record(aaa = 0.0, bbb = 1.0))"),
        vec![
            "`Normal` has no parameter `aaa` (spec §08 parameters: `mu`, `sigma`)".to_string(),
            "`Normal` has no parameter `bbb` (spec §08 parameters: `mu`, `sigma`)".to_string(),
        ],
        "a splatted record's field names are the names that bind"
    );
    // The §08 names themselves still pass, in any order, and positionally.
    assert!(errors("m = Normal(mu = 0.0, sigma = 1.0)").is_empty());
    assert!(errors("m = Normal(sigma = 1.0, mu = 0.0)").is_empty());
    assert!(errors("m = Normal(0.0, sigma = 1.0)").is_empty());
    assert!(errors("m = Normal(0.0, 1.0)").is_empty());
    // A user callable's parameters are declared by its own boundary.
    assert_eq!(
        errors("f(x) = mul(x, 2.0)\ny = f(zzz = 1.5)"),
        vec!["`f` has no parameter `zzz` (declares: `x`)".to_string()]
    );
    assert!(errors("f(x) = mul(x, 2.0)\ny = f(x = 1.5)").is_empty());
}

/// The name check must not fire on a call whose count is admissible under the
/// ORDINARY reading of a sole record — there the record is one value and its
/// fields are not binding names, so checking them would reject a legal call.
#[test]
fn an_ordinary_record_argument_is_not_name_checked() {
    let src = "pars = record(zzz = 1.0, qqq = 2.0)\n\
               takes_a_record(p) = get(p, [\"zzz\"])\n\
               y = takes_a_record(pars)";
    assert!(
        errors(src).is_empty(),
        "the record binds to `p`; `zzz`/`qqq` are its fields, not argument names"
    );
}
