//! A density output reads a draw only from its own point or through a `lawof`.
//!
//! §13 "Output reduction": "`draw` nodes take their values from the explicit
//! `point`, unless marginalized out". A density query whose measure reads a draw
//! outside every `lawof` has no value. The determinizer applies no implicit
//! marginal, and it never reads the draw from another output's point, because
//! §13 reduces per output.
use flatppl_core::Symbol;
use flatppl_determinizer::{determinize, determinize_with_roots};
use flatppl_infer::ModuleBundle;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn refusal(src: &str) -> String {
    determinize(&parse_infer(src))
        .expect_err("a free draw in a density query must refuse")
        .reason
}

const SHARED_PARAMETER: &str = "z = draw(Normal(mu = 0.4, sigma = 1.0))\n\
     lp = logdensityof(joint(f1 = Normal(mu = z, sigma = 0.5), \
     f2 = Normal(mu = z, sigma = 1.5)), record(f1 = 0.5, f2 = 0.7))\n";

#[test]
fn inference_accepts_a_stochastic_constructor_parameter() {
    // The model is legal: `lp` is a stochastic-phase value. Only a density OUTPUT
    // needs a deterministic value, so only determinization refuses.
    let mut m = flatppl_syntax::parse(SHARED_PARAMETER).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn a_stochastic_constructor_parameter_refuses_and_names_the_draw() {
    let reason = refusal(SHARED_PARAMETER);
    assert!(
        reason.contains("density query in `lp` reads draw `z`")
            && reason.contains("put `z` in the point")
            && reason.contains("lawof(...)"),
        "{reason}"
    );
}

#[test]
fn another_outputs_point_does_not_condition_the_query() {
    // §13 reduces per output, so `lp_latent`'s point 0.1 is not a value for `z`
    // in `lp`, in either statement order.
    let latent = "lp_latent = logdensityof(lawof(z), 0.1)\n";
    let (first, rest) = SHARED_PARAMETER.split_once('\n').unwrap();
    for src in [
        format!("{first}\n{latent}{rest}\n"),
        format!("{SHARED_PARAMETER}{latent}"),
    ] {
        let reason = refusal(&src);
        assert!(reason.contains("reads draw `z`"), "{src}\n{reason}");
    }
}

#[test]
fn an_inline_draw_parameter_refuses() {
    let reason = refusal(
        "lp = logdensityof(Normal(mu = draw(Normal(mu = 0.0, sigma = 1.0)), sigma = 1.0), 0.5)",
    );
    assert!(reason.contains("reads an inline `draw`"), "{reason}");
}

#[test]
fn a_query_in_a_helper_binding_of_an_output_refuses() {
    let src = "z = draw(Normal(mu = 0.0, sigma = 1.0))\n\
               a = logdensityof(Normal(mu = z, sigma = 1.0), 0.5)\n\
               lp = a + 1.0";
    let mut m = parse_infer(src);
    let lp: Symbol = m.intern("lp");
    let err = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[lp]))
        .expect_err("the output depends on a free-draw query");
    assert!(
        err.reason.contains("density query in `a` reads draw `z`"),
        "{}",
        err.reason
    );
}

#[test]
fn a_law_or_a_boundary_input_still_lowers() {
    // `lawof` integrates `z`; the point carries the record law's own draws.
    determinize(&parse_infer(
        "z = draw(Normal(mu = 0.4, sigma = 1.0))\n\
         f1 = draw(Normal(mu = z, sigma = 0.5))\n\
         lp = logdensityof(lawof(f1), 0.5)\n\
         lq = logdensityof(lawof(record(z = z, f1 = f1)), record(z = 0.1, f1 = 0.5))",
    ))
    .expect("lawof spellings lower");
    // A reification boundary turns `z` into an input of `F`, so `score` is scored
    // at that input, not at the draw.
    let mut m = parse_infer(&format!(
        "{SHARED_PARAMETER}F = functionof(lp, z = z)\nout = F(0.3)"
    ));
    let out: Symbol = m.intern("out");
    determinize_with_roots(&m, &ModuleBundle::new(), Some(&[out]))
        .expect("a boundary input is not a free draw");
}
