use flatppl_syntax::{Syntax, parse, print_with};

fn assert_equivalent(sugared: &str, explicit: &str) {
    let module = parse(sugared).unwrap();
    let expected = flatppl_flatpir::write(&parse(explicit).unwrap());
    assert_eq!(flatppl_flatpir::write(&module), expected);
    for syntax in [Syntax::Full, Syntax::Minimal] {
        let printed = print_with(&module, syntax);
        let reparsed = parse(&printed).unwrap();
        assert_eq!(flatppl_flatpir::write(&reparsed), expected);
        assert_eq!(print_with(&reparsed, syntax), printed);
    }
}

#[test]
fn parenthesized_records_preserve_fields_and_nested_expressions() {
    assert_equivalent(
        "x = 3\ny = 5\nr = (b = (only_field = x), a = ((x, y), (x + y)))\nz = r.b.only_field",
        "x = 3\ny = 5\nr = record(b = record(only_field = x), a = ((x, y), (x + y)))\nz = r.b.only_field",
    );
}

#[test]
fn bare_reification_boundaries_preserve_written_order() {
    assert_equivalent(
        "a = elementof(reals)\nb = elementof(reals)\nf = functionof(a + 2 * b, b, a)\ny ~ Normal(a + 2 * b, 2.0)\nk = base.kernelof((y = y), b, a)",
        "a = elementof(reals)\nb = elementof(reals)\nf = functionof(a + 2 * b, b = b, a = a)\ny ~ Normal(a + 2 * b, 2.0)\nk = base.kernelof(record(y = y), b = b, a = a)",
    );
}

#[test]
fn keyword_boundaries_preserve_tuple_values_and_lambda_scopes() {
    assert_equivalent(
        "a = elementof(reals)\nb = elementof(reals)\nt = (a, b)\nf = base.functionof(t, pair = t, a = a, beta = b)\ng = x -> (value = x + a)\nh = fn((value = _))\nz = f(t, a, b)",
        "a = elementof(reals)\nb = elementof(reals)\nt = (a, b)\nf = base.functionof(t, pair = t, a = a, beta = b)\ng = x -> record(value = x + a)\nh = fn(record(value = _))\nz = f(t, a, b)",
    );
}

#[test]
fn bare_boundaries_preserve_tuple_values() {
    assert_equivalent(
        "a = elementof(reals)\nb = elementof(reals)\nt = (a, b)\nf = base.functionof(t, t, a, b)\nz = f(t, a, b)",
        "a = elementof(reals)\nb = elementof(reals)\nt = (a, b)\nf = base.functionof(t, t = t, a = a, b = b)\nz = f(t, a, b)",
    );
}

/// Spec §04: "Boundary inputs are given either all as bare names or all as
/// keyword arguments; mixing the two forms is a static error."
#[test]
fn mixed_bare_and_keyword_boundaries_are_refused() {
    for operation in ["functionof", "kernelof"] {
        for boundaries in [
            "pair = t, a, beta = b",
            "a, beta = b",
            "a = a, b",
            "a, _x_, beta = b",
        ] {
            let source = format!(
                "a = elementof(reals)\nb = elementof(reals)\nt = (a, b)\n\
                 f = {operation}(draw(Normal(a + b, 1.0)) + _x_, {boundaries})"
            );
            let err = parse(&source).expect_err(&format!("accepted {source}"));
            assert!(
                err.to_string().contains("mixing the two forms"),
                "{source}: {err}"
            );
        }
    }
}

/// Spec §04: in a bare boundary list "a placeholder `_x_` expands to `x = _x_`".
#[test]
fn a_bare_placeholder_expands_to_its_stripped_name() {
    assert_equivalent(
        "a = elementof(reals)\nd = elementof(reals)\nh = functionof(a * d + _x_, a, d, _x_)",
        "a = elementof(reals)\nd = elementof(reals)\nh = functionof(a * d + _x_, a = a, d = d, x = _x_)",
    );
    assert_equivalent(
        "a = elementof(reals)\nk = kernelof(draw(Normal(a + _shift_, 1.0)), _shift_, a)",
        "a = elementof(reals)\nk = kernelof(draw(Normal(a + _shift_, 1.0)), shift = _shift_, a = a)",
    );
}

/// Spec §04: "Derived names must be legal argument names".
#[test]
fn a_bare_placeholder_naming_a_reserved_word_is_refused() {
    for placeholder in [
        "_in_", "_all_", "_only_", "_true_", "_false_", "_self_", "_base_",
    ] {
        let source =
            format!("a = elementof(reals)\nf = functionof(a + {placeholder}, a, {placeholder})");
        let err = parse(&source).expect_err(&format!("accepted {source}"));
        assert!(err.to_string().contains("reserved word"), "{source}: {err}");
    }
}

#[test]
fn shadowed_construct_names_keep_ordinary_calls() {
    assert_equivalent(
        "functionof(x, y) = x + 2 * y\nkernelof(x, y) = x - y\nrecord(x) = x\nf = functionof(3, 5)\nk = kernelof(3, 5)\na = elementof(reals)\ng = base.functionof((value = a), a)\nr = (value = 13)",
        "functionof(x, y) = x + 2 * y\nkernelof(x, y) = x - y\nrecord(x) = x\nf = self.functionof(3, 5)\nk = self.kernelof(3, 5)\na = elementof(reals)\ng = base.functionof(base.record(value = a), a = a)\nr = base.record(value = 13)",
    );
}

#[test]
fn grouped_and_computed_boundaries_are_not_boundary_lists() {
    for operation in ["functionof", "kernelof"] {
        for boundary in ["(a, b)", "(a = a, b = b)", "(a)", "a + 1"] {
            let source = format!(
                "a = elementof(reals)\nb = elementof(reals)\nf = {operation}(a + b, {boundary})"
            );
            assert!(parse(&source).is_err(), "accepted {source}");
        }
    }
}

#[test]
fn ordinary_calls_do_not_accept_bare_names_after_keywords() {
    for source in [
        "x = Normal(mu = 1.0, 2.0)",
        "functionof(x, y) = x + y\nf = functionof(x = 1, 2)",
    ] {
        assert!(parse(source).is_err(), "accepted {source}");
    }
}
