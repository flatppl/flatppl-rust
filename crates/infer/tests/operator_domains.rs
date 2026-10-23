//! Spec §07 "Operator-equivalent functions": each row states a Domains cell, and an
//! operand outside it is a located static error whose type is `%failed`, never
//! `%deferred`. A record, tuple or table is in no row's domain.
use flatppl_core::Type;
use flatppl_infer::{Severity, infer};

fn errors(source: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(source).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn clean(source: &str) {
    let found = errors(source);
    assert!(found.is_empty(), "{source}: {found:?}");
}

fn refused(source: &str) {
    assert!(!errors(source).is_empty(), "{source}");
}

const BINARY: [&str; 11] = [
    "add", "sub", "mul", "divide", "pow", "equal", "unequal", "lt", "le", "gt", "ge",
];

const OUTSIDE: [&str; 6] = [
    "record(a = 1.0, b = 2.0)",
    "(1.0, 2.0)",
    "table(a = [1.0, 2.0], b = [3.0, 4.0])",
    "Normal(0.0, 1.0)",
    "(x -> x)",
    "fn(Normal(mu = _, sigma = 1.0))",
];

#[test]
fn a_record_tuple_table_or_object_operand_is_outside_every_row() {
    for op in BINARY {
        for operand in OUTSIDE {
            for call in [
                format!("{op}({operand}, 1.5)"),
                format!("{op}(1.5, {operand})"),
            ] {
                let found = errors(&format!("z = {call}\n"));
                assert!(!found.is_empty(), "{call}");
            }
        }
    }
    for operand in OUTSIDE {
        refused(&format!("z = neg({operand})\n"));
    }
}

#[test]
fn the_spec_example_record_plus_a_number_is_refused_with_a_located_failed_type() {
    for source in [
        "z = record(a = 1, b = 2) + 10.0\n",
        "z = 10.0 - record(a = 1, b = 2)\n",
        "z = (1.0, 2.0) * 2.0\n",
        "z = -record(a = 1.0)\n",
    ] {
        let mut module = flatppl_syntax::parse(source).unwrap();
        let diagnostics = infer(&mut module);
        let error = diagnostics
            .iter()
            .find(|d| d.severity == Severity::Error)
            .unwrap_or_else(|| panic!("{source}"));
        assert!(
            error.message.contains("§07") && error.message.contains("Operator-equivalent"),
            "{}",
            error.message
        );
        assert!(error.node.is_some(), "{source}: the error is not located");
        let rhs = module.bindings().next().unwrap().1.rhs;
        assert!(
            matches!(module.type_of(rhs), Some(Type::Failed(_))),
            "{source}: {:?}",
            module.type_of(rhs)
        );
    }
}

#[test]
fn a_sole_record_or_table_splats_and_is_not_itself_an_operand() {
    clean("z = add(record(a = 1.0, b = 2.0))\n");
    clean("z = add(table(a = [1.0], b = [2.0]))\n");
}

#[test]
fn the_scalar_kind_clauses_of_the_domains() {
    // `lt`..`ge`: `reals`, so a complex number is outside; `equal`: discrete domains.
    clean("z = lt(1, 2.5)\n");
    refused("z = lt(1.0 + 2.0*im, 1.0)\n");
    clean("z = equal(1, 2)\n");
    clean("z = equal(true, false)\n");
    refused("z = equal(1.5, 2.5)\n");
    refused("x = elementof(reals)\nz = x == 0.5\n");
    // §03 embeds booleans and integers in the arithmetic domains.
    clean("z = true + 1\n");
    clean("z = neg(true)\n");
    clean("z = divide([1.0, 2.0], 2.0)\n");
    clean("z = add([1.0, 2.0], [3.0, 4.0])\n");
    // An array of records is no array of numbers.
    refused("z = add([record(a = 1.0), record(a = 2.0)], 1.0)\n");
}

#[test]
fn the_per_cell_call_of_a_broadcast_obeys_the_row_domain() {
    // A table's cells are records.
    refused("t = table(a = [1.0, 2.0], b = [3.0, 4.0])\nz = broadcast(add, t, 1.0)\n");
    refused("z = [1.0, 2.0] .== 2.0\n");
    clean("z = [1.0, 2.0] .+ 1.0\n");
    clean("z = [1.0, 2.0] .< 2.0\n");
    clean("z = [1, 2] .== 2\n");
}
