//! A sample query (`d, s2 = rand(lawof(z), s)` with the rng state `s` as an
//! input) emitted in logdensity mode. It panicked at `batching::shape`
//! ("expected numeric tensor shape"): the Enzyme static-shape check read the
//! rng key's `tensor<2xui64>` as a numeric tensor. Logdensity mode must refuse
//! it with a message instead; sample mode emits it as before. A key computed
//! from a seed reads no draw, so logdensity mode still returns it.

use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

const SRC: &str = "q = elementof(interval(0.0, 1.0))\nx ~ Binomial(10, q)\nz = x * 2 + 1\n\
                   s = rnginit(0)\nd, s2 = rand(lawof(z), s)\ninputs = (s, q)\noutputs = (d, s2)\n";

fn emit(mode: Mode, opts: EmitOptions) -> Result<String, String> {
    emit_source(SRC, mode, opts)
}

fn emit_source(src: &str, mode: Mode, opts: EmitOptions) -> Result<String, String> {
    let mut model = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(&model, mode, &opts).map_err(|e| e.to_string())
}

#[test]
fn logdensity_mode_refuses_a_sample_query_on_every_target() {
    for (target, restrict) in [
        (Target::Cpu, true),
        (Target::Gpu, true),
        (Target::Cpu, false),
    ] {
        let err = emit(
            Mode::LogDensity,
            EmitOptions {
                target,
                restrict_enzyme_compatible: restrict,
                ..Default::default()
            },
        )
        .expect_err("a sample query is not a logdensity program");
        let want = if restrict {
            "Enzyme restrictions require a deterministic query"
        } else {
            "rng state, which logdensity mode does not return"
        };
        assert!(err.contains(want), "{target:?}/{restrict}: {err}");
    }
}

#[test]
fn logdensity_mode_returns_a_key_from_a_seed() {
    let hlo = emit_source(
        "seed = elementof(integers)\ninputs = seed\noutputs = rnginit(seed)\n",
        Mode::LogDensity,
        EmitOptions::default(),
    )
    .expect("a seeded key reads no draw");
    assert!(hlo.contains("tensor<2xui64>"), "{hlo}");
}

#[test]
fn sample_mode_still_emits_the_query() {
    let hlo = emit(
        Mode::Sample,
        EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .expect("sample mode emits");
    assert!(
        hlo.contains("func.func @sample(%key: tensor<2xui64>"),
        "{hlo}"
    );
}

/// No mode, target or dtype panics on the query: each either emits or refuses
/// with a message.
#[test]
fn no_combination_panics() {
    for sample in [false, true] {
        for gpu in [false, true] {
            for f32 in [false, true] {
                for restrict in [true, false] {
                    let result = std::panic::catch_unwind(|| {
                        let mode = if sample {
                            Mode::Sample
                        } else {
                            Mode::LogDensity
                        };
                        let opts = EmitOptions {
                            target: if gpu { Target::Gpu } else { Target::Cpu },
                            dtype: if f32 { Dtype::F32 } else { Dtype::F64 },
                            restrict_enzyme_compatible: restrict,
                            ..Default::default()
                        };
                        emit(mode, opts)
                    });
                    assert!(
                        result.is_ok(),
                        "panicked: sample={sample} gpu={gpu} f32={f32} restrict={restrict}"
                    );
                }
            }
        }
    }
}
