//! Spec §04 "Additional arguments": an operation that calls a function or kernel
//! passes the additional arguments `...` positionally or by name, a sole record or
//! table by its fields. `scan`, `kscan` and `filter` traverse them as sequences.
use flatppl_core::{Dim, Type};
use flatppl_infer::{Severity, infer};

fn errors(source: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(source).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn type_of(source: &str, name: &str) -> Type {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{source}: {errors:?}");
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .unwrap()
        .1
        .rhs;
    module.type_of(rhs).cloned().unwrap()
}

#[test]
fn callbacks_take_several_positional_keyword_and_record_additional_arguments() {
    for source in [
        "f(x, a, b) = a * x + b\nM = weighted(f, Normal(0, 1), 2.0, 1.0)",
        "f(x, a, b) = a * x + b\nM = weighted(f, Normal(0, 1), b = 1.0, a = 2.0)",
        "f(x, a, b) = a * x + b\nM = logweighted(f, Normal(0, 1), record(a = 2.0, b = 1.0))",
        "f(x, a) = a * x\nM = pushfwd(f, Normal(0, 1), a = 2.0)",
        "step(s, d, k) = Normal(s + d, k)\nM = markovchain(step, 0.0, 3, 0.1, 0.5)",
        "step(s, d, k) = Normal(s + d, k)\nM = markovchain(step, 0.0, 3, k = 0.5, d = 0.1)",
        "step(s, d, k) = Normal(s + d, k)\nM = markovchain(step, 0.0, 3, record(d = 0.1, k = 0.5))",
        "x = reduce((a, b, p, q) -> a + b * p + q, [1.0, 2.0], 2.0, 1.0)",
        "x = reduce((a, b, p, q) -> a + b * p + q, [1.0, 2.0], q = 1.0, p = 2.0)",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn additional_arguments_must_bind_every_input_exactly_once() {
    for source in [
        // Unbound.
        "f(x, a, b) = a * x + b\nM = weighted(f, Normal(0, 1), 2.0)",
        // Unknown keyword.
        "f(x, a) = a * x\nM = pushfwd(f, Normal(0, 1), c = 2.0)",
        // Bound twice.
        "f(x, a) = a * x\nM = weighted(f, Normal(0, 1), 2.0, a = 3.0)",
        // Too many.
        "f(x, a) = a * x\nM = weighted(f, Normal(0, 1), 2.0, 3.0)",
        "step(s, d) = Normal(s + d, 1.0)\nM = markovchain(step, 0.0, 3)",
        "step(s, d) = Normal(s + d, 1.0)\nM = markovchain(step, 0.0, 3, e = 0.5)",
    ] {
        assert!(!errors(source).is_empty(), "{source}");
    }
}

#[test]
fn scan_traverses_its_sequences_together() {
    // scan(step, 0, [10, 20], [2, 3]) steps 0 -> 20 -> 80.
    let ty = type_of(
        "step(a, x, w) = a + x * w\nr = scan(step, 0.0, [10.0, 20.0], [2.0, 3.0])",
        "r",
    );
    let Type::Array { shape, .. } = ty else {
        panic!("{ty:?}");
    };
    assert_eq!(shape.as_ref(), [Dim::Static(2)]);
    // A length-one sequence repeats; the result length follows the other one.
    let ty = type_of(
        "step(a, x, w) = a + x * w\nr = scan(step, 0.0, [10.0, 20.0], [2.0])",
        "r",
    );
    let Type::Array { shape, .. } = ty else {
        panic!("{ty:?}");
    };
    assert_eq!(shape.as_ref(), [Dim::Static(2)]);
    // Keyword and sole-table forms bind the same inputs.
    for source in [
        "step(a, x, w) = a + x * w\nr = scan(step, 0.0, w = [2.0, 3.0], x = [10.0, 20.0])",
        "step(a, x, w) = a + x * w\nr = scan(step, 0.0, table(x = [10.0, 20.0], w = [2.0, 3.0]))",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn scan_sequences_must_agree_in_length() {
    for source in [
        // Unequal lengths other than one.
        "step(a, x, w) = a + x * w\nr = scan(step, 0.0, [1.0, 2.0, 3.0], [1.0, 2.0])",
        "k(s, x, w) = Normal(s + x, w)\nM = kscan(k, 0.0, [1.0, 2.0, 3.0], [1.0, 2.0])",
        "within(x, lo) = lo < x\nr = filter(within, [1, 4, 9], [0, 5])",
    ] {
        assert!(!errors(source).is_empty(), "{source}");
    }
}

#[test]
fn scalars_and_objects_are_held_constant_at_every_step() {
    // scan(step, 0, [10, 20], 2) steps 0 -> 20 -> 60.
    let ty = type_of(
        "step(a, x, w) = a + x * w\nr = scan(step, 0, [10, 20], 2)",
        "r",
    );
    let Type::Array { shape, .. } = ty else {
        panic!("{ty:?}");
    };
    assert_eq!(shape.as_ref(), [Dim::Static(2)]);
    for source in [
        "step(a, x, w) = a + x * w\nr = scan(step, 0.0, [10.0, 20.0], 2.0)",
        "step(a, x, w, v) = a + x * w + v\nr = scan(step, 0.0, [10.0, 20.0], 2.0, [1.0, 2.0])",
        "k(s, x, w) = Normal(s + x, w)\nM = kscan(k, 0.0, [1.0, 2.0], 0.5)",
        "within(x, lo) = lo < x\nr = filter(within, [1, 4, 9], 3)",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn sequence_callbacks_receive_record_parameters_whole() {
    // §04 "Explicit callback parameters": the final p is passed whole,
    // independently of the arrays or tables traversed by the operation.
    for (sequence, parameter) in [("", "record(w = 2.0)"), ("x = ", "p = record(w = 2.0)")] {
        for source in [
            format!(
                "step(a, x, p) = a + x * p.w\nr = scan(step, 0.0, {sequence}[10.0, 20.0], {parameter})"
            ),
            format!(
                "k(s, x, p) = Normal(s + x * p.w, 1.0)\nM = kscan(k, 0.0, {sequence}[1.0, 2.0], {parameter})"
            ),
            format!("within(x, p) = x > p.w\nr = filter(within, [1.0, 4.0, 9.0], {parameter})"),
        ] {
            assert_eq!(errors(&source), Vec::<String>::new(), "{source}");
        }
    }
    // Non-sequence operations retain the same whole-parameter rule.
    for source in [
        "step(a, p) = a + p.w\nr = markovchain(step, 0.0, 3, record(w = 2.0))",
        "w(x, p) = p.s * x\nm = weighted(w, Normal(0.0, 1.0), record(s = 2.0))",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn a_tuple_or_a_two_dimensional_array_input_is_a_static_error() {
    for source in [
        "step(a, x, p) = a + x\nr = scan(step, 0.0, [1.0, 2.0], (1.0, 2.0))",
        "step(a, x, p) = a + x\nr = scan(step, 0.0, [1.0, 2.0], rowstack([[1.0, 2.0], [3.0, 4.0]]))",
        "k(s, x, p) = Normal(s + x, 1.0)\nM = kscan(k, 0.0, [1.0, 2.0], (1.0, 2.0))",
        "within(x, p) = x < 3\nr = filter(within, [1, 4, 9], (1, 2))",
    ] {
        assert!(!errors(source).is_empty(), "{source}");
    }
}

#[test]
fn scan_and_kscan_need_one_array_or_table_input() {
    for source in [
        "step(a, w) = a + w\nr = scan(step, 0.0, 2.0)",
        "k(s, w) = Normal(s, w)\nM = kscan(k, 0.0, 2.0)",
        "step(a, w, p) = a + w\nr = scan(step, 0.0, 2.0, record(p = 1.0))",
    ] {
        assert!(!errors(source).is_empty(), "{source}");
    }
}

#[test]
fn kscan_traverses_its_sequences_together() {
    for source in [
        "k(s, x, w) = Normal(s + x, w)\nM = kscan(k, 0.0, [1.0, 2.0], [0.5, 0.25])",
        "k(s, x, w) = Normal(s + x, w)\nM = kscan(k, 0.0, [1.0, 2.0], [0.5])",
        "k(s, x, w) = Normal(s + x, w)\nM = kscan(k, 0.0, w = [0.5, 0.25], x = [1.0, 2.0])",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
    let ty = type_of(
        "k(s, x, w) = Normal(s + x, w)\nM = kscan(k, 0.0, [1.0, 2.0], [0.5])",
        "M",
    );
    let Type::Measure { domain, .. } = ty else {
        panic!("{ty:?}");
    };
    let Type::Array { shape, .. } = *domain else {
        panic!("{domain:?}");
    };
    assert_eq!(shape.as_ref(), [Dim::Static(2)]);
}

#[test]
fn filter_traverses_additional_inputs_with_the_data() {
    for source in [
        "within(x, low, high) = low < x && x < high\nr = filter(within, [1, 4, 9], low = [0, 5, 8], high = [10])",
        "within(x, low, high) = low < x && x < high\nr = filter(within, [1, 4, 9], [0, 5, 8], [10])",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
    assert!(!errors(
        "within(x, low, high) = low < x && x < high\nr = filter(within, [1, 4, 9], [0, 5], [10])"
    )
    .is_empty());
}

#[test]
fn chains_pass_additional_arguments_unchanged_to_every_step() {
    for source in [
        "f1(x, p) = x + p\nf2(y, p) = y * p\nc = fchain(f1, f2)\nr = c(1.0, 2.0)",
        "f1(x, p) = x + p\nf2(y, p) = y * p\nc = fchain(f1, f2)\nr = c(1.0, p = 2.0)",
        "k1(x, p) = Normal(x + p, 1.0)\nk2(y, p) = Normal(y + p, 2.0)\nc = kchain(k1, k2)\nr = c(1.0, 0.5)",
        "k1(x, p) = Normal(x + p, 1.0)\nk2(y, p) = Normal(y + p, 2.0)\nc = kchain(k1, k2)\nr = c(1.0, p = 0.5)",
    ] {
        assert_eq!(errors(source), Vec::<String>::new(), "{source}");
    }
    // A keyword the later step does not declare is not bound there.
    for source in [
        "f1(x, p) = x + p\nf2(y, q) = y * q\nc = fchain(f1, f2)\nr = c(1.0, p = 2.0)",
        "k1(x, p) = Normal(x + p, 1.0)\nk2(y, q) = Normal(y + q, 2.0)\nc = kchain(k1, k2)\nr = c(1.0, p = 0.5)",
    ] {
        assert!(!errors(source).is_empty(), "{source}");
    }
}

#[test]
fn a_fed_record_reaches_the_next_step_whole_when_the_chain_has_additional_arguments() {
    // Spec §06: in `kchain(K1, K2, K3)(x, ...)` each step "receives its fed value whole as
    // its first argument", so a call with additional arguments never auto-splats the fed
    // record onto the next step's inputs, even when the field names match them.
    let joint = "k1(x, p) = joint(a = Normal(x + p, 1.0), b = Normal(x, 1.0))\n";
    for chain in ["kchain", "jointchain"] {
        // The fed record binds `v` whole.
        assert_eq!(
            errors(&format!(
                "{joint}k2(v, p) = Normal(v.a + v.b + p, 1.0)\nc = {chain}(k1, k2)\nr = c(1.0, 10.0)"
            )),
            Vec::<String>::new(),
            "{chain}"
        );
        // Inputs named after the fields are not bound from it: `p` is left unbound.
        let found = errors(&format!(
            "{joint}k2(a, b, p) = Normal(a + b + p, 1.0)\nc = {chain}(k1, k2)\nr = c(1.0, 10.0)"
        ));
        assert!(
            found
                .iter()
                .any(|m| m.contains("input `p` is left unbound")),
            "{chain}: {found:?}"
        );
        // With no additional arguments the ordinary calling convention still splats.
        assert_eq!(
            errors(&format!(
                "k1(x) = joint(a = Normal(x, 1.0), b = Normal(x, 1.0))\n\
                 k2(a, b) = Normal(a + b, 1.0)\nc = {chain}(k1, k2)\nr = c(1.0)"
            )),
            Vec::<String>::new(),
            "{chain} without extras"
        );
    }
    let f1 = "f1(x, p) = record(a = x, b = p)\n";
    assert_eq!(
        errors(&format!(
            "{f1}f2(r, p) = r.a + r.b + p\nc = fchain(f1, f2)\ny = c(1.0, 10.0)"
        )),
        Vec::<String>::new()
    );
    assert!(
        !errors(&format!(
            "{f1}f2(a, b, p) = a + b + p\nc = fchain(f1, f2)\ny = c(1.0, 10.0)"
        ))
        .is_empty()
    );
    assert_eq!(
        errors("f1(x) = record(a = x, b = 2.0)\nf2(a, b) = a + b\nc = fchain(f1, f2)\ny = c(1.0)"),
        Vec::<String>::new()
    );
    // The fed record is typed as the step's record, so `r.a + r.b + p` is a real.
    assert_eq!(
        type_of(
            &format!("{f1}f2(r, p) = r.a + r.b + p\nc = fchain(f1, f2)\ny = c(1.0, 10.0)"),
            "y"
        ),
        flatppl_core::Type::Scalar(flatppl_core::ScalarType::Real)
    );
}
