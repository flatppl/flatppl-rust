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

/// The nonempty calls that motivated the rule. Each typed a concrete scalar (or
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
    ] {
        assert_eq!(errors(src), vec![want.to_string()], "for {src}");
        assert!(
            ir(src).contains("(%failed"),
            "{src} must type %failed:\n{}",
            ir(src)
        );
    }
}

/// A predefined constant (spec §03: `true`, `false`, `inf`, `pi`, `im`, and the
/// eleven named value-sets) is a known value, never a callable — §04 "Language
/// design": "No callables may have nullary inputs, as this would make them
/// equivalent to known values." Applying one to arguments, the `(pi 0.5)`
/// class, used to reach no type rule and fall through to `%deferred` with no
/// diagnostic — indistinguishable from an honest "no rule yet" gap and
/// invisible to the `is_flatpdl` `Type::Failed` backstop. `true`/`false` are
/// NOT covered here: they parse as a `CallHead::User` callee application (a
/// different code path — `call_rule`'s user-call arm, not `CallHead::Builtin`)
/// — see `predefined_constant_true_false_applied_via_user_call_is_a_static_error`
/// below for that route.
#[test]
fn predefined_constant_applied_to_arguments_is_now_a_static_error() {
    for (src, want) in [
        (
            "x = pi(0.5)",
            "`pi` is a predefined constant (spec §03), not a callable, so it cannot be applied \
             to arguments (spec §04 \"Language design\": no callable has nullary inputs, which \
             is what a known value like `pi` would need to be one)",
        ),
        (
            "x = inf(0.5)",
            "`inf` is a predefined constant (spec §03), not a callable, so it cannot be applied \
             to arguments (spec §04 \"Language design\": no callable has nullary inputs, which \
             is what a known value like `inf` would need to be one)",
        ),
        (
            "x = reals(0.5)",
            "`reals` is a predefined constant (spec §03), not a callable, so it cannot be \
             applied to arguments (spec §04 \"Language design\": no callable has nullary \
             inputs, which is what a known value like `reals` would need to be one)",
        ),
    ] {
        assert_eq!(errors(src), vec![want.to_string()], "for {src}");
        assert!(
            ir(src).contains("(%failed"),
            "{src} must type %failed:\n{}",
            ir(src)
        );
    }
    // A bare reference to the constant, or a well-formed expression using one,
    // is untouched — only its APPLICATION is malformed.
    assert!(errors("x = pi\ny = 3.0 * pi / 4.0").is_empty());
}

/// `true`/`false` are lexed unconditionally to a bool `Lit` (never a `Ref`),
/// so `true(0.5)` is postfix APPLICATION on that literal (spec §11), reaching
/// `CallHead::User` — `user_call_type` — rather than the `CallHead::Builtin`
/// path the test above covers. Same spec §03/§04 violation, same outcome.
#[test]
fn predefined_constant_true_false_applied_via_user_call_is_a_static_error() {
    for (src, name) in [("x = true(0.5)", "true"), ("x = false(0.5)", "false")] {
        let want = format!(
            "`{name}` is a predefined constant (spec §03), not a callable, so it cannot be \
             applied to arguments (spec §04 \"Language design\": no callable has nullary \
             inputs, which is what a known value like `{name}` would need to be one)"
        );
        assert_eq!(errors(src), vec![want], "for {src}");
        assert!(
            ir(src).contains("(%failed"),
            "{src} must type %failed:\n{}",
            ir(src)
        );
    }
    // A bare `true`/`false`, or a genuine user callable, is untouched.
    assert!(errors("x = true\ny = false").is_empty());
    assert!(errors("f(a) = a + 1.0\nx = f(1.0)").is_empty());
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

/// §07 "Linear algebra" gives `rowstack(vs)` and `colstack(vs)` ONE parameter —
/// "a vector of equal-length vectors" — so neither stacks a vararg list of
/// vectors. `rowstack(a, b)` is the natural mistake (NumPy's `row_stack` and
/// Julia's `vcat` both take varargs) and a 2026-07-22 build accepted it; nothing
/// pinned the refusal afterwards.
#[test]
fn rowstack_and_colstack_take_one_vector_of_vectors() {
    let v = "a = [1.0, 2.0]\nb = [3.0, 4.0]\n";
    for head in ["rowstack", "colstack"] {
        // The one declared spelling, positional and keyword.
        assert!(errors(&format!("{v}M = {head}([a, b])")).is_empty());
        assert!(errors(&format!("{v}M = {head}(vs = [a, b])")).is_empty());
        // A vararg list of vectors is not a second spelling.
        assert_eq!(
            errors(&format!("{v}M = {head}(a, b)")),
            vec![format!("`{head}` takes 1 argument (spec §07), got 2")]
        );
        assert_eq!(
            errors(&format!("{v}M = {head}(a, b, a)")),
            vec![format!("`{head}` takes 1 argument (spec §07), got 3")]
        );
    }
}

/// §07 "Array and table operations" makes `cat(scalar1, scalar2, ...)`
/// "Equivalent to `vector(scalar1, scalar2, ...)`", and array literals lower to
/// real `vector` calls — `[1.0]` to `(vector 1.0)`. So the shared
/// `x1, x2, ...` notation admits one argument, and reading it as "at least
/// two" for `cat` would reject `cat(1.0)` while accepting the equivalent
/// `vector(1.0)`.
///
/// The empty `(vector)` IR node remains available for axis lists. A bare `[]`
/// fails on its position; the parser separately rejects `vector()` and `cat()`.
#[test]
fn cat_and_vector_read_the_same_notation_the_same_way() {
    assert!(errors("v = cat(1.0)").is_empty());
    assert!(errors("v = cat(1.0, 2.0)").is_empty());
    assert!(errors("v = vector(1.0)").is_empty());
    assert!(errors("v = [1.0]").is_empty());
    let msgs = errors("v = []");
    assert_eq!(msgs.len(), 1);
    assert!(
        msgs[0].contains("an axis list is legal only as `output_axes`"),
        "a bare empty axis list must fail on position: {}",
        msgs[0]
    );
}

/// Named arguments count toward the total, so the keyword spelling of a §07
/// parameter is not a missing argument.
#[test]
fn named_arguments_count_toward_arity() {
    assert!(!errors("n = 3\nx = checked(n, condition = n > 0)").is_empty());
    assert!(errors("n = 3\nx = checked(value = n, condition = n > 0)").is_empty());
    assert_eq!(
        errors("n = 3\nx = checked(n, condition = n > 0, extra = 1.0)"),
        vec!["ordinary calls use either positional or keyword arguments, not both".to_string()]
    );
}

/// A callable requiring multiple inputs receives a sole record by field name.
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
    assert!(!errors("m = Normal(0.0, sigma = 1.0)").is_empty());
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
    assert!(!errors("m = Normal(0.0, sigma = 1.0)").is_empty());
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

/// Unary table operations keep the whole table and infer its result shape.
#[test]
fn unary_operations_type_their_whole_table_argument() {
    let t = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals, y = reals), 4))\n";
    let pir = |f: &str| ir(&format!("{t}s = {f}(d)"));
    // Table in, same table out.
    for f in ["reverse", "identity"] {
        let out = pir(f);
        assert!(
            out.contains("(%bind s (%meta ((%table (%columns (x (%scalar real)) (y (%scalar real))) (%nrows 4))"),
            "`{f}` over a table must type as that same table, in:\n{out}"
        );
    }
    // Axis indices: a rank-1 integer array.
    for f in ["indicesof", "indicesof0"] {
        let out = pir(f);
        assert!(
            out.contains("(%bind s (%meta ((%array 1 (4) (%scalar integer))"),
            "`{f}` over a table must type as an integer index array, in:\n{out}"
        );
    }
    // Row count.
    let out = pir("lengthof");
    assert!(
        out.contains("(%bind s (%meta ((%scalar integer) %fixed nonnegintegers)"),
        "`lengthof` over a table must type as a nonneg integer, in:\n{out}"
    );
    // The four §07 table reductions now give the record the paragraph defines:
    // "returns a record whose fields are the column names and values are the
    // per-column reductions". Previously `sum`/`mean` were `%deferred` and `var`
    // MISTYPED as `(%scalar real)`; both were pinned as-measured here, and this
    // assertion replacing those pins is the deliberate change they existed to force.
    for f in ["sum", "mean"] {
        let out = pir(f);
        assert!(
            out.contains(&format!(
                "(%bind s (%meta ((%record (x (%scalar real)) (y (%scalar real))) \
                 %fixed (cartprod (%field x reals) (%field y reals))) ({f}"
            )),
            "`{f}` over a table must give a record of per-column sums, in:\n{out}"
        );
    }
    // `var`/`std` reduce each column to a NON-NEGATIVE real, so the per-field set is
    // `nonnegreals` — their catalogue row's `result_set`, applied per column instead
    // of to the whole result.
    for f in ["var", "std"] {
        let out = pir(f);
        assert!(
            out.contains(&format!(
                "(%bind s (%meta ((%record (x (%scalar real)) (y (%scalar real))) \
                 %fixed (cartprod (%field x nonnegreals) (%field y nonnegreals))) ({f}"
            )),
            "`{f}` over a table must give a record of per-column variances, in:\n{out}"
        );
    }
}

/// §07 **Table reductions**: "When `sum`, `mean`, or `var` is applied to a table,
/// the reduction operates column-wise and returns a record whose fields are the
/// column names and values are the per-column reductions. Every column must support
/// the reduction operation." `std` joins that list by the owner ruling of 2026-08-10
/// (flatppl-design `4c93237`) — a commit which is NOT on design `main`, so `std`'s
/// membership here rests on unmerged spec, exactly as its splat exemption does.
///
/// The per-column value is whatever the reduction gives for an ARRAY of that
/// column's element type, so the table and array forms agree by construction:
/// `sum`/`mean` keep the column's element type, `var`/`std` give a real. That is
/// derived from the existing array rules rather than being a second set of rules.
#[test]
fn a_table_reduction_gives_a_record_of_per_column_reductions() {
    let two = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals, y = reals), 4))\n";
    // Field NAMES come from the columns, in column order.
    for f in ["sum", "mean", "var", "std"] {
        let out = ir(&format!("{two}s = {f}(d)"));
        assert!(
            out.contains("(%record (x (%scalar real)) (y (%scalar real)))"),
            "`{f}` must give a record keyed by the column names, in:\n{out}"
        );
    }
    // `sum` keeps the column's own element type — an INTEGER column sums to an
    // integer, exactly as `sum` of an integer array does.
    let mixed = "d = load_data(\"d.csv\", cartpow(cartprod(k = integers, y = reals), 4))\n";
    let out = ir(&format!("{mixed}s = sum(d)"));
    assert!(
        out.contains("(%record (k (%scalar integer)) (y (%scalar real)))"),
        "`sum` must keep each column's element type, in:\n{out}"
    );
    // `mean` does NOT: §07 defines it as (1/n)Σxᵢ, and the mean of `[1, 2]` is `1.5`,
    // so an integer column means to a REAL. An earlier revision of this test asserted
    // the opposite, having derived `mean` from `sum` instead of from §07's formula —
    // which reproduced a pre-existing bug in the ARRAY path (`mean([1, 2, 3])` also
    // typed integer) and dressed the agreement up as a correctness argument. Both
    // paths now share `reduced_scalar`, checked against the formula.
    let out = ir(&format!("{mixed}s = mean(d)"));
    assert!(
        out.contains("(%record (k (%scalar real)) (y (%scalar real)))"),
        "`mean` of an integer column must be real, in:\n{out}"
    );
    // `var`/`std` give a real per column whatever the column's type, mirroring their
    // catalogue row's `result: Scalar(Real)` on the array form.
    for f in ["var", "std"] {
        let out = ir(&format!("{mixed}s = {f}(d)"));
        assert!(
            out.contains("(%record (k (%scalar real)) (y (%scalar real)))"),
            "`{f}` must give a real per column, in:\n{out}"
        );
    }
}

/// A one-column table reduction still returns a one-field record.
#[test]
fn a_one_column_table_reduces_to_a_one_field_record() {
    let one = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals), 4))\n";
    for f in ["sum", "mean"] {
        let out = ir(&format!("{one}s = {f}(d)"));
        assert!(
            out.contains("(%record (x (%scalar real)))")
                && !out.contains("%bind s (%meta ((%scalar"),
            "`{f}` over a one-column table is a one-field record, not a scalar, in:\n{out}"
        );
    }
    for f in ["var", "std"] {
        let out = ir(&format!("{one}s = {f}(d)"));
        assert!(
            out.contains("(%record (x (%scalar real)))"),
            "`{f}` over a one-column table is a one-field record, in:\n{out}"
        );
    }
}

/// The value-set must describe the same value the TYPE does. Before this rule
/// `var(<table>)` carried `nonnegreals` — a scalar set on what is now a record — and
/// nothing reading the set rather than the type could tell.
#[test]
fn a_table_reduction_carries_a_matching_record_value_set() {
    let two = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals, y = reals), 4))\n";
    for (f, want) in [
        ("sum", "(cartprod (%field x reals) (%field y reals))"),
        ("mean", "(cartprod (%field x reals) (%field y reals))"),
        (
            "var",
            "(cartprod (%field x nonnegreals) (%field y nonnegreals))",
        ),
        (
            "std",
            "(cartprod (%field x nonnegreals) (%field y nonnegreals))",
        ),
    ] {
        let out = ir(&format!("{two}s = {f}(d)"));
        assert!(
            out.contains(want),
            "`{f}` must carry the value-set {want}, in:\n{out}"
        );
    }
}

/// A deliberate NON-extension, spec-grounded: a column whose per-row type is NOT a
/// scalar leaves the whole call `%deferred`. §07 says only "Every column must
/// support the reduction operation" and does not say what reducing a column of
/// vectors yields, so inventing one would be guessing. Covers all seven table
/// reductions, including the three design PR #79 adds.
#[test]
fn table_reductions_leave_non_scalar_columns_deferred() {
    let vec_col = "d = load_data(\"d.csv\", cartpow(cartprod(v = cartpow(reals, 3)), 4))\n";
    for f in ["sum", "mean", "var", "std", "prod", "maximum", "minimum"] {
        let out = ir(&format!("{vec_col}s = {f}(d)"));
        assert!(
            out.contains(&format!("(%bind s (%meta (%deferred %fixed %unknown) ({f}")),
            "`{f}` over a vector-column table must stay %deferred, in:\n{out}"
        );
    }
}

/// `prod` keeps the column's own element type (spec §07: a product of integers is
/// an integer, same as `sum`). `maximum`/`minimum` also keep the element type —
/// they return an ELEMENT of the column rather than a computed aggregate,
/// matching their catalogue row's `ElemScalarKind` result on the array form.
#[test]
fn prod_maximum_minimum_reduce_a_table_column_wise() {
    let two = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals, y = reals), 4))\n";
    for f in ["prod", "maximum", "minimum"] {
        // The unary interface passes the table whole to its domain rule.
        assert!(
            errors(&format!("{two}s = {f}(d)")).is_empty(),
            "`{f}` must accept a whole table"
        );
        let out = ir(&format!("{two}s = {f}(d)"));
        assert!(
            out.contains(&format!(
                "(%bind s (%meta ((%record (x (%scalar real)) (y (%scalar real))) \
                 %fixed (cartprod (%field x reals) (%field y reals))) ({f}"
            )),
            "`{f}` over a table must give a record keyed by the column names, in:\n{out}"
        );
    }
    // Element type is preserved per column, including for an INTEGER column —
    // unlike `mean`/`var`/`std`, none of the three computes a new numeric kind.
    let mixed = "d = load_data(\"d.csv\", cartpow(cartprod(k = integers, y = reals), 4))\n";
    for f in ["prod", "maximum", "minimum"] {
        let out = ir(&format!("{mixed}s = {f}(d)"));
        assert!(
            out.contains("(%record (k (%scalar integer)) (y (%scalar real)))"),
            "`{f}` must keep each column's own element type, in:\n{out}"
        );
    }
    // A ONE-column table still gives a ONE-field record, not a scalar — the same
    // case `a_one_column_table_reduces_to_a_one_field_record` pins for sum/mean.
    let one = "d = load_data(\"d.csv\", cartpow(cartprod(x = reals), 4))\n";
    for f in ["prod", "maximum", "minimum"] {
        let out = ir(&format!("{one}s = {f}(d)"));
        assert!(
            out.contains("(%record (x (%scalar real)))")
                && !out.contains("%bind s (%meta ((%scalar"),
            "`{f}` over a one-column table is a one-field record, not a scalar, in:\n{out}"
        );
    }
    // The value-set must describe the record, not the scalar set `maximum`/
    // `minimum`'s catalogue row would give for a bare array argument — the same
    // drift `a_table_reduction_carries_a_matching_record_value_set` guards for
    // `var`/`std`.
    for f in ["prod", "maximum", "minimum"] {
        let out = ir(&format!("{two}s = {f}(d)"));
        assert!(
            out.contains("(cartprod (%field x reals) (%field y reals))"),
            "`{f}` must carry the value-set (cartprod (%field x reals) (%field y reals)), in:\n{out}"
        );
    }
}

/// §07's `mean` is $\bar{x} = \frac{1}{n}\sum_i x_i$, so the mean of an INTEGER array
/// is a REAL — the mean of `[1, 2]` is `1.5`. `reduce_type` returned the element type
/// for all of `sum`/`prod`/`mean`, so `mean([1, 2, 3])` typed integer. That is
/// arithmetic, so it outranks the previous code.
///
/// This was pre-existing on `main`, not introduced here; it surfaced because the
/// table rule was derived from it and inherited it. `sum` and `prod` of integers
/// stay integers, which is correct, and complex stays complex for all three.
#[test]
fn mean_of_an_integer_array_is_real() {
    let out = ir("v = [1, 2, 3]\ns = mean(v)");
    assert!(
        out.contains("(%bind s (%meta ((%scalar real) %fixed reals) (mean"),
        "`mean` of an integer array must be real, in:\n{out}"
    );
    // The reductions whose element type IS the answer keep it.
    for f in ["sum", "prod"] {
        let out = ir(&format!("v = [1, 2, 3]\ns = {f}(v)"));
        assert!(
            out.contains(&format!(
                "(%bind s (%meta ((%scalar integer) %fixed integers) ({f}"
            )),
            "`{f}` of an integer array stays an integer, in:\n{out}"
        );
    }
}

/// The ARRAY forms are otherwise untouched — the table rule is guarded on the
/// argument being a table, so every non-table call keeps the arm it had.
#[test]
fn array_reductions_are_unchanged_by_the_table_rule() {
    let v = "v = [1.0, 2.0, 3.0]\n";
    for (f, want) in [
        ("sum", "((%scalar real) %fixed reals)"),
        ("mean", "((%scalar real) %fixed reals)"),
        ("prod", "((%scalar real) %fixed reals)"),
        ("var", "((%scalar real) %fixed nonnegreals)"),
        ("std", "((%scalar real) %fixed nonnegreals)"),
    ] {
        let out = ir(&format!("{v}s = {f}(v)"));
        assert!(
            out.contains(want),
            "`{f}` over an array must still be {want}, in:\n{out}"
        );
    }
}
