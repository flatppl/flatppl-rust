//! A bare-name alias chain (`S = S1`, `S1 = S0`) emits exactly what the direct
//! spelling emits.
//!
//! §04 *Name resolution* makes `S` name the very expression `S0` names. The
//! emitter reads a fixed operand (a set, a literal dimension, a data column,
//! a yields matrix) through ONE `(%ref self x)` hop, so `emit` first points
//! every alias reference at the end of its chain. Each family is checked on
//! FlatPDL written directly, which no canonicalization has touched, and on a
//! model lowered by the determiniser.

use flatppl_stablehlo::{EmitOptions, Mode};

/// Emit FlatPDL as written.
fn emit_flatpdl(source: &str) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    flatppl_stablehlo::emit(&model, Mode::LogDensity, &EmitOptions::default())
        .unwrap_or_else(|e| panic!("{e}\n{source}"))
}

/// Determinize a model, then emit it.
fn emit_model(source: &str) -> String {
    let options = EmitOptions::default();
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .unwrap();
    flatppl_stablehlo::emit(&model, Mode::LogDensity, &options).unwrap()
}

/// `head` binds `S`; `chain` binds `S` through `S1` to `S0`.
fn assert_chain_emits_direct(emit: Emit, direct: &str, chain: &str, tail: &str) {
    assert_eq!(
        emit(&format!("{chain}{tail}")),
        emit(&format!("{direct}{tail}"))
    );
}

/// `registry::literal_fixed_positive_int`: LKJ's fixed dimension `n`.
#[test]
fn lkj_dimension() {
    let (direct, chain) = ("n = 3\n", "n0 = 3\nn1 = n0\nn = n1\n");
    let head = "eta = elementof(posreals)\nc_obs = elementof(cartpow(reals, [3, 3]))\n\
                inputs = (eta, c_obs)\noutputs = (lp)\n";
    assert_chain_emits_direct(
        emit_flatpdl,
        direct,
        chain,
        &format!("{head}lp = builtin_logdensityof(LKJ, record(n = n, eta = eta), c_obs)\n"),
    );
    assert_chain_emits_direct(
        emit_model,
        direct,
        chain,
        &format!("{head}lp = logdensityof(LKJ(n = n, eta = eta), c_obs)\n"),
    );
}

/// `registry::uniform_bounds` / `uniform_logpdf`: the support set.
#[test]
fn uniform_support() {
    let (direct, chain) = (
        "S = interval(0.0, 1.0)\n",
        "S0 = interval(0.0, 1.0)\nS1 = S0\nS = S1\n",
    );
    let head = "x = elementof(reals)\ninputs = (x)\noutputs = (lp)\n";
    assert_chain_emits_direct(
        emit_flatpdl,
        direct,
        chain,
        &format!("{head}lp = builtin_logdensityof(Uniform, record(support = S), x)\n"),
    );
    assert_chain_emits_direct(
        emit_model,
        direct,
        chain,
        &format!("{head}lp = logdensityof(Uniform(S), x)\n"),
    );
}

/// `ops::lower_in` / `classify_elem_set`: a membership test's set.
#[test]
fn set_membership() {
    let tail = "x = elementof(reals)\nv = ifelse(x in S, 1.0, 0.0)\n\
                inputs = (x)\noutputs = (v)\n";
    for emit in [emit_flatpdl, emit_model] {
        assert_chain_emits_direct(
            emit,
            "S = interval(0.0, 2.0)\n",
            "S0 = interval(0.0, 2.0)\nS1 = S0\nS = S1\n",
            tail,
        );
    }
}

/// `Emitter::bind_column` / `column_arg`: a column of a promoted data input,
/// read through an alias of the container.
#[test]
fn data_column() {
    let tail = "data = load_data(\"data.json\", cartpow(cartprod(y = reals), 3))\n\
                mu = elementof(reals)\nv = sum(D.y) * mu\n\
                inputs = (mu, data)\noutputs = (v)\n";
    for emit in [emit_flatpdl, emit_model] {
        assert_chain_emits_direct(emit, "D = data\n", "D0 = data\nD1 = D0\nD = D1\n", tail);
    }
}

/// `pyhf::yields_call` / `yield_rows`: the nominal-yields matrix of a retained
/// `pyhf-helpers` call.
#[test]
fn pyhf_yields() {
    let tail = "pyhf = standard_module(\"pyhf-helpers\", \"0.1\")\n\
                theta = elementof(cartpow(reals, 2))\ninputs = theta\n\
                a = theta[1]\ns = theta[2]\n\
                n1 = [1.0, 2.0]\nn2 = [3.0, 4.0]\ndelta = [0.5, 0.25] .* s\n\
                shifts = array(cat(delta, fill(0.0, 2)), [2, 1, 2], [1, 2, 3])\n\
                factors = array(cat(fill(a, 2), fill(1.0, 2)), [2, 1, 2], [1, 2, 3])\n\
                outputs = pyhf.expected_counts(pyhf.sample_yields(R, shifts, factors))\n";
    for emit in [emit_flatpdl, emit_model] {
        assert_chain_emits_direct(
            emit,
            "R = rowstack([n1, n2])\n",
            "R0 = rowstack([n1, n2])\nR1 = R0\nR = R1\n",
            tail,
        );
    }
}

/// An emit path: determinize-then-emit, or emit FlatPDL as written.
type Emit = fn(&str) -> String;

/// `lp` as a model writes it and as FlatPDL writes it, scoring 0.5 under
/// `Normal(mu, sigma)`.
const PATHS: [(Emit, &str); 2] = [
    (
        emit_model,
        "lp = logdensityof(Normal(mu = MU, sigma = SIGMA), 0.5)\n",
    ),
    (
        emit_flatpdl,
        "lp = builtin_logdensityof(Normal, record(mu = MU, sigma = SIGMA), 0.5)\n",
    ),
];

fn score(template: &str, mu: &str, sigma: &str) -> String {
    template.replace("MU", mu).replace("SIGMA", sigma)
}

/// `modes::read_abi` / `bind_input`: an `inputs` entry behind an alias chain is
/// classified by the construct at the end of the chain (§04, §13), and an
/// output may read either name.
#[test]
fn aliased_inputs() {
    for (emit, lp) in PATHS {
        let direct = emit(&format!(
            "x = elementof(reals)\n{}inputs = (x)\noutputs = (lp)\n",
            score(lp, "x", "1.0")
        ));
        for mu in ["X2", "x"] {
            let aliased = emit(&format!(
                "x = elementof(reals)\nX = x\nX2 = X\n{}inputs = (X2)\noutputs = (lp)\n",
                score(lp, mu, "1.0")
            ));
            assert_eq!(aliased, direct, "mu = {mu}");
        }
    }
}

/// A fixed value behind an alias is promoted like the direct binding.
#[test]
fn aliased_fixed_input() {
    for (emit, lp) in PATHS {
        let tail = format!(
            "x = elementof(reals)\n{}inputs = (x, s)\noutputs = (lp)\n",
            score(lp, "x", "s")
        );
        assert_chain_emits_direct(
            emit,
            "s = external(posreals)\n",
            "s0 = external(posreals)\ns1 = s0\ns = s1\n",
            &tail,
        );
    }
}

/// An alias and its target are one object, so listing both is listing one
/// binding twice.
#[test]
fn an_alias_and_its_target_are_one_input() {
    let source = "x = elementof(reals)\nX = x\n\
                  lp = builtin_logdensityof(Normal, record(mu = x, sigma = 1.0), 0.5)\n\
                  inputs = (X, x)\noutputs = (lp)\n";
    let mut model = flatppl_syntax::parse(source).unwrap();
    let _ = flatppl_infer::infer(&mut model);
    let err = flatppl_stablehlo::emit(&model, Mode::LogDensity, &EmitOptions::default())
        .expect_err("one object, two argument positions");
    assert!(err.to_string().contains("more than once"), "{err}");
}

/// A retained `pyhf-helpers` call through an aliased module (§04 *Aliasing*).
#[test]
fn pyhf_module_alias() {
    let tail = "theta = elementof(cartpow(reals, 2))\ninputs = theta\n\
                a = theta[1]\ns = theta[2]\n\
                n1 = [1.0, 2.0]\nn2 = [3.0, 4.0]\ndelta = [0.5, 0.25] .* s\n\
                shifts = array(cat(delta, fill(0.0, 2)), [2, 1, 2], [1, 2, 3])\n\
                factors = array(cat(fill(a, 2), fill(1.0, 2)), [2, 1, 2], [1, 2, 3])\n\
                outputs = P.expected_counts(P.sample_yields(rowstack([n1, n2]), shifts, factors))\n";
    let load = "standard_module(\"pyhf-helpers\", \"0.1\")";
    for emit in [emit_flatpdl, emit_model] {
        assert_chain_emits_direct(
            emit,
            &format!("P = {load}\n"),
            &format!("P0 = {load}\nP1 = P0\nP = P1\n"),
            tail,
        );
    }
}

/// `bind_input`'s aggregate branch: a data input listed under an alias binds
/// the columns its uses read.
#[test]
fn aliased_data_input() {
    let data =
        "data = load_data(\"data.json\", cartpow(cartprod(y = reals), 3))\nmu = elementof(reals)\n";
    for emit in [emit_flatpdl, emit_model] {
        let direct = emit(&format!(
            "{data}v = sum(data.y) * mu\ninputs = (mu, data)\noutputs = (v)\n"
        ));
        for read in ["D", "data"] {
            let aliased = emit(&format!(
                "{data}D1 = data\nD = D1\nv = sum({read}.y) * mu\ninputs = (mu, D)\noutputs = (v)\n"
            ));
            assert_eq!(aliased, direct, "read = {read}");
        }
    }
}
