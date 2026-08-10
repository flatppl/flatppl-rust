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

/// The tail every diagnostic gets when its count or names came from a SPLATTED
/// record or table, pointing at the keyword spelling §04 names. Spelled out here
/// rather than imported from the crate: importing would make the assertions
/// tautological (the message equalling itself), whereas this copy pins the rendered
/// text and fails loudly if it drifts.
const SPLAT_HINT: &str = " — this sole positional record or table splatted into one \
                           argument per field (spec §04), so its field names bind as \
                           argument names; to pass it as one ordinary argument use \
                           the keyword spelling, as in `f(pars = record(...))`";

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
/// record fields, the two functions compose directly"), so a USER call must be
/// able to count a sole record argument by its fields. Two corpus models are this
/// shape — `simple-transport2.flatppl`'s `k_model(glob_pars)` and `feature-test1
/// .flatppl`'s `forward_kernel(rand_pars)` — and both were falsely rejected while
/// the user-call path counted a sole record as one argument.
///
/// design#74 settled that the splat reading is not merely available but the ONLY
/// one for a sole positional record — see
/// `a_sole_positional_record_always_splats_and_is_not_one_argument`.
#[test]
fn a_user_call_can_splat_a_sole_record_argument() {
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
        vec![format!(
            "`lin` declares 3 parameters, got 2 arguments{SPLAT_HINT}"
        )]
    );
}

/// The INVERSE of what this test asserted while §04 was ambiguous. design#74
/// settled the question in favour of always-splat: "A sole positional record or
/// table therefore always splats: whether its field or column names match the
/// callable's argument names decides only whether the call is valid, never whether
/// the splat occurs." So a 3-field record handed positionally to a 1-parameter
/// callable is NOT one argument — it splats to three, and the arity check rejects
/// it. Passing it as one value takes the keyword spelling (§04: "requires the
/// keyword spelling, as in `f(pars = record(...))`"), asserted below.
///
/// Kept rather than deleted, per the review that placed the interim version: the
/// shape is exactly the one the permissive rule existed to admit, so it is the
/// case a regression would resurrect.
#[test]
fn a_sole_positional_record_always_splats_and_is_not_one_argument() {
    let f = "pars = record(a = 1.0, b = 2.0, mu = 3.0)\n\
             takes_a_record(p) = get(p, [\"a\"])\n";
    assert_eq!(
        errors(&format!("{f}y = takes_a_record(pars)")),
        vec![format!(
            "`takes_a_record` declares 1 parameter, got 3 arguments{SPLAT_HINT}"
        )],
        "the 3-field record splats to three arguments, so the 1-parameter callable is over-supplied"
    );
    // The keyword spelling §04 names is how the record passes as ONE value.
    assert!(
        errors(&format!("{f}y = takes_a_record(p = pars)")).is_empty(),
        "bound to a parameter by keyword, the record is an ordinary value and does not splat"
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
        vec![format!(
            "`Normal` takes 2 arguments (spec §08), got 3{SPLAT_HINT}"
        )]
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
    // Every unbindable name is reported, not just the first. A SPLATTING call also
    // names the keyword spelling, since the splat is what made these field names
    // bind and the author may have meant to pass the record as one value.
    assert_eq!(
        errors("m = Normal(record(aaa = 0.0, bbb = 1.0))"),
        vec![
            format!(
                "`Normal` has no parameter `aaa` (spec §08 parameters: `mu`, `sigma`){SPLAT_HINT}"
            ),
            format!(
                "`Normal` has no parameter `bbb` (spec §08 parameters: `mu`, `sigma`){SPLAT_HINT}"
            ),
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

/// The name check must not reach the fields of a record that is NOT splatted.
/// design#74 removed the case this test was originally written around — a sole
/// positional record read as one ordinary value — so it now pins the two
/// non-splatting cases §04 still names: "a record given alongside other arguments,
/// or bound to a parameter by keyword, is an ordinary value and is not splatted."
/// In both, the record binds as a value and its field names are not argument names.
#[test]
fn a_record_that_does_not_splat_is_not_name_checked() {
    let f = "pars = record(zzz = 1.0, qqq = 2.0)\n\
             takes_a_record(p) = get(p, [\"zzz\"])\n";
    // Bound to a parameter by keyword.
    assert!(
        errors(&format!("{f}y = takes_a_record(p = pars)")).is_empty(),
        "the record binds to `p`; `zzz`/`qqq` are its fields, not argument names"
    );
    // Alongside another argument.
    let g = "pars = record(zzz = 1.0, qqq = 2.0)\n\
             pick(p, k) = get(p, [k])\n";
    assert!(
        errors(&format!("{g}y = pick(pars, \"zzz\")")).is_empty(),
        "a record given alongside another argument is an ordinary value"
    );
}

/// The always-splat rule switches the §04 name check ON for SINGLE-parameter
/// constructors, which the earlier either-reading rule left unchecked: a 1-field
/// record satisfied the plain reading on count, so its field name never bound and
/// never got compared. `Poisson(record(zzz = 0.5))` was silently accepted and
/// determinized to a `Poisson` with no `rate`.
///
/// The flatppl-js twin of this case landed alongside (js#136), where the same
/// spelling produced a silent NaN rather than a silent accept.
#[test]
fn a_single_parameter_constructor_name_checks_a_splatted_record() {
    assert_eq!(
        errors("m = Poisson(record(zzz = 0.5))"),
        vec![format!(
            "`Poisson` has no parameter `zzz` (spec §08 parameters: `rate`){SPLAT_HINT}"
        )],
        "the sole record splats, so `zzz` binds as an argument name and fails"
    );
    // The right field name splats cleanly, and the positional form is unaffected.
    assert!(errors("m = Poisson(record(rate = 0.5))").is_empty());
    assert!(errors("m = Poisson(0.5)").is_empty());
    assert!(errors("m = Poisson(rate = 0.5)").is_empty());
    // Not specific to `Poisson` — every single-parameter row is now covered.
    assert_eq!(
        errors("m = Geometric(record(zzz = 0.5))"),
        vec![format!(
            "`Geometric` has no parameter `zzz` (spec §08 parameters: `p`){SPLAT_HINT}"
        )]
    );
}

/// §04 names records AND tables in the same breath ("`f(record(a = x, ...))` and
/// `f(table(a = x, ...))` are equivalent to `f(a = x, ...)`"), and the amendment
/// says "a sole positional record or table". A table splats by COLUMN name, on the
/// same unconditional terms.
#[test]
fn a_sole_positional_table_splats_by_column_name() {
    assert_eq!(
        errors("d = table(zzz = [1.0, 2.0])\nm = Poisson(d)"),
        vec![format!(
            "`Poisson` has no parameter `zzz` (spec §08 parameters: `rate`){SPLAT_HINT}"
        )],
        "the sole table splats, so its column name binds as an argument name"
    );
    // A column named for the parameter splats cleanly.
    assert!(errors("d = table(rate = [1.0, 2.0])\nm = Poisson(d)").is_empty());
}

/// The splat hint must reach the ARITY diagnostics, not only the name check. The
/// arity branch is where §04's always-splat rule is most confusing: the author
/// wrote one argument and the error reports several, none of which they typed. All
/// three reporting paths carry it — the two arity mismatches and the name check.
///
/// A call that does NOT splat gets no hint: an ordinary over-arity call has
/// nothing to explain, and the hint would be noise on the common case.
#[test]
fn the_splat_hint_reaches_every_diagnostic_a_splatting_call_can_produce() {
    // §08 constructor, arity path: 2 fields against 1 parameter.
    assert_eq!(
        errors("m = Poisson(record(zzz = 0.5, qqq = 1.0))"),
        vec![format!(
            "`Poisson` takes 1 argument (spec §08), got 2{SPLAT_HINT}"
        )]
    );
    // §07 value function, arity path — the reach onto §07 the amendment created.
    // `prod` and not `sum`: `sum` is exempt under #78's carve-out (see
    // `a_single_input_callable_whose_domain_admits_tables_is_exempt`), while `prod` is
    // an arrays-only row and still splats.
    assert_eq!(
        errors("d = load_data(\"d.csv\", cartpow(cartprod(x = reals, y = reals), 4))\ns = prod(d)"),
        vec![format!(
            "`prod` takes 1 argument (spec §07), got 2{SPLAT_HINT}"
        )]
    );
    // USER call, arity path — the shape the transport models used.
    assert_eq!(
        errors("pars = record(a = 1.0, b = 2.0, mu = 3.0)\nf(p) = get(p, [\"a\"])\ny = f(pars)"),
        vec![format!(
            "`f` declares 1 parameter, got 3 arguments{SPLAT_HINT}"
        )]
    );
    // Not a splat: no hint on either arity path.
    assert_eq!(
        errors("f(x) = mul(x, 2.0)\ny = f(1.0, 2.0)"),
        vec!["`f` declares 1 parameter, got 2 arguments".to_string()]
    );
    assert_eq!(
        errors("x = exp(1.0, 2.0)"),
        vec!["`exp` takes 1 argument (spec §07), got 2".to_string()]
    );
}

/// §04's single-input carve-out (flatppl-design#78, pending owner review): "A
/// callable with exactly one input whose documented domain admits records or tables
/// is exempt and receives a sole positional record or table whole, so that `sum(t)`
/// and `lengthof(t)` reduce over the table rather than splatting."
///
/// Without it the unconditional splat binds by NAME, so `sum(t)` was valid for no
/// table at any column count — which made §07's **Table reductions** paragraph dead
/// prose and contradicted §03 "Tables": "`lengthof(t)` returns the number of table
/// rows."
///
/// The exempt set was derived by reading every §07 domain, not from #78's two
/// examples: `lengthof` is the only row whose *Domains* cell names tables
/// ("vectors, tables"), and `sum`/`mean`/`var` get theirs from the Table reductions
/// paragraph. Both halves of the condition come from the CALLEE's signature.
#[test]
fn a_single_input_callable_whose_domain_admits_tables_is_exempt() {
    let t = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals, y = reals), 4))\n";
    // The four exempt rows take the two-column table whole.
    for f in ["sum", "mean", "var", "lengthof"] {
        assert!(
            errors(&format!("{t}s = {f}(d)")).is_empty(),
            "`{f}` is exempt and must take the table whole"
        );
        // Same for a record, which §04 names alongside tables.
        assert!(
            errors(&format!("r = record(a = 1.0, b = 2.0)\ns = {f}(r)")).is_empty(),
            "`{f}` is exempt for a record too"
        );
    }
    // `std` is the near miss and is NOT exempt: it is sqrt(var) over "real arrays",
    // but the Table reductions paragraph names three functions and `std` is not one,
    // so no documented domain admits a table. Following the spec as written.
    assert_eq!(
        errors(&format!("{t}s = std(d)")),
        vec![format!(
            "`std` takes 1 argument (spec §07), got 2{SPLAT_HINT}"
        )],
        "`std` has no documented table domain, so it still splats"
    );
    // Arrays-only rows keep splatting.
    for f in ["prod", "sizeof"] {
        assert_eq!(
            errors(&format!("{t}s = {f}(d)")),
            vec![format!(
                "`{f}` takes 1 argument (spec §07), got 2{SPLAT_HINT}"
            )],
            "`{f}` is arrays-only and must still splat"
        );
    }
    // The ARITY half matters independently of the domain half: `Exponential` has one
    // input, but its domain is `reals`, so it splats — and here the field name
    // happens to match, which is what makes the call valid rather than the exemption.
    assert!(errors("m = Exponential(record(rate = 1.0))").is_empty());
    assert_eq!(
        errors("m = Exponential(record(zzz = 1.0))"),
        vec![format!(
            "`Exponential` has no parameter `zzz` (spec §08 parameters: `rate`){SPLAT_HINT}"
        )],
        "one input is not enough to exempt: the domain must admit aggregates"
    );
    // Multi-input splats are untouched, and so are the vector domains of the exempt
    // rows themselves.
    assert!(errors("m = Normal(record(mu = 0.0, sigma = 1.0))").is_empty());
    assert!(errors("v = [1.0, 2.0]\ns = sum(v)").is_empty());
    assert!(errors("v = [1.0, 2.0]\ns = lengthof(v)").is_empty());
}

/// A user callable is never exempt: #78 keys the carve-out on a *documented domain*,
/// and a user boundary declares parameters, not domains. So a sole positional record
/// still splats into a user call — which is what `a_user_call_can_splat_a_sole_record
/// _argument` relies on.
#[test]
fn a_user_callable_is_never_exempt_from_the_splat() {
    let f = "pars = record(a = 1.0, b = 2.0)\n";
    assert_eq!(
        errors(&format!("{f}one(p) = get(p, [\"a\"])\ny = one(pars)")),
        vec![format!(
            "`one` declares 1 parameter, got 2 arguments{SPLAT_HINT}"
        )],
        "a one-parameter user callable has no documented domain, so no exemption"
    );
    // And the keyword spelling is still the way to pass the record whole.
    assert!(errors(&format!("{f}one(p) = get(p, [\"a\"])\ny = one(p = pars)")).is_empty());
}
