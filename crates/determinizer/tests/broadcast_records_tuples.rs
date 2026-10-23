//! Spec §04 "Broadcasting", "Disallowed inputs": "Records and tuples are not allowed as
//! operands of `broadcast`; a table row reaching the callable as a record is unaffected."
//! Inference types such a call as failed, so the determinizer refuses it rather than
//! hold the record constant or iterate the tuple. A table operand gives the callable one
//! record per row, under the callable's own call rule.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn pir(src: &str) -> String {
    flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"))
}

#[test]
fn a_record_or_tuple_operand_is_refused() {
    let f = "f(a, p) = a * p.s\ng(a, c) = a * c\nrec = record(s = 0.5, u = 2.0)\n";
    for (call, source) in [
        (
            "record held constant",
            format!("{f}r = broadcast(f, [1.0, 2.0], rec)\n"),
        ),
        (
            "record in a kernel broadcast",
            format!(
                "{f}K(m, p) = Normal(mu = m, sigma = p.s)\nM = broadcast(K, [1.0, 2.0], rec)\n\
                 lp = logdensityof(M, [1.0, 2.0])\n"
            ),
        ),
        (
            "record in a sampled broadcast",
            format!(
                "{f}K(m, p) = Normal(mu = m, sigma = p.s)\nM = broadcast(K, [1.0, 2.0], rec)\n\
                 out = rand(M, rnginit(7))\n"
            ),
        ),
        ("tuple", format!("{f}r = broadcast(g, (2.0, 3.0), 10.0)\n")),
        ("tuple of builtin", "r = (2.0, 3.0) .+ 10.0\n".to_string()),
    ] {
        let err = determinize(&parse_infer(&source)).expect_err(call);
        // A kernel broadcast is refused earlier, for being a kernel outside a primitive, and
        // the sampling path for its non-scalar cells.
        assert!(
            ["operand", "kernel", "scalar cells"]
                .iter()
                .any(|part| err.reason.contains(part)),
            "{call}: {err:?}"
        );
    }
}

#[test]
fn a_table_row_reaches_the_callable_as_a_record() {
    // `f` of two inputs splats the row record; `h` of one input takes it whole.
    let table = "t = table(a = [1.0, 2.0], b = [3.0, 4.0])\n";
    for call in [
        "f(a, b) = a * b\nr = broadcast(f, t)\n",
        "f(a, b) = a * b\nr = broadcasted(f)(t)\n",
        "f(a, b) = a * b\nr = f.(t)\n",
        "h(row) = row.a * row.b\nr = broadcast(h, t)\n",
        "h(row) = row.a * row.b\nr = h.(t)\n",
    ] {
        let out = pir(&format!("{table}{call}"));
        assert!(
            out.contains("(%bind r (%meta ((%array 1 (2) (%scalar real))"),
            "{call}\n{out}"
        );
    }
}
