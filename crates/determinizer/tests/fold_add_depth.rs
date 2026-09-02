//! An `iid` over N observations must lower to a sum that nests ⌈log2 N⌉ deep.
//!
//! `density::fold_add` used to fold its terms LEFT, so the emitted FlatPDL was as
//! deep as the dataset is long. A 2000-point likelihood printed a 1998-term `+`
//! chain, and `flatppl-js` could not score it at all: its recursive IR walkers
//! overflowed Node's default stack inside `processSource`, before any
//! materialisation, with `Maximum call stack size exceeded`. FlatPDL is a target
//! other tools consume, so the depth has to stay within a normal recursive
//! walker's reach.
//!
//! `13-determinization.md` "Output reduction" fixes only that `iid` SUMS its
//! component densities, so the association is the determiniser's to choose. What
//! is NOT free is the term sequence: these tests pin that the terms are the same
//! multiset in the same left-to-right order, and that N <= 2 is byte-identical to
//! the old left fold, since a pairwise tree cannot differ there by construction
//! and a change that made it differ would be a bug elsewhere.

use flatppl_determinizer::determinize;

/// `logdensityof` of an iid Normal likelihood over `n` observations.
fn iid_model(n: usize) -> String {
    let obs: Vec<String> = (0..n).map(|i| format!("{}.0", i % 7)).collect();
    format!(
        "g = Normal(mu = 0.0, sigma = 1.0)\n\
         obs = [{}]\n\
         lk = likelihoodof(iid(g, lengthof(obs)), obs)\n\
         lp = logdensityof(lk, record())",
        obs.join(", ")
    )
}

fn lower(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("must lower, not refuse");
    flatppl_determinizer::is_flatpdl(&out).expect("output must be FlatPDL");
    flatppl_flatpir::write(&out)
}

/// The greatest number of `add` calls on any root-to-leaf path of `pir`.
///
/// This is the quantity a consumer's recursive walker pays for, so it is what the
/// bound has to be stated in. Counted by tracking, for each open paren, whether it
/// began an `add` call.
fn add_nesting_depth(pir: &str) -> usize {
    let bytes = pir.as_bytes();
    let mut stack: Vec<bool> = Vec::new();
    let mut adds = 0usize;
    let mut max = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => {
                let is_add = pir[i..].starts_with("(add ") || pir[i..].starts_with("(add\n");
                stack.push(is_add);
                if is_add {
                    adds += 1;
                    max = max.max(adds);
                }
            }
            b')' => {
                if let Some(true) = stack.pop() {
                    adds -= 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    assert!(stack.is_empty(), "unbalanced parens in:\n{pir}");
    max
}

#[test]
fn an_iid_over_2000_observations_nests_logarithmically() {
    let pir = lower(&iid_model(2000));

    // Every observation contributes exactly one density term, so the tree this
    // has to balance really does have 2000 leaves.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2000,
        "expected one density term per observation"
    );

    // 2000 -> 1000 -> 500 -> 250 -> 125 -> 63 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    // is 11 pairings. The left fold produced 1999, which is the defect.
    let depth = add_nesting_depth(&pir);
    assert_eq!(
        depth, 11,
        "an iid over 2000 points must nest 11 `add` levels (ceil(log2 2000)), \
         not {depth}; a left fold gives 1999 and overflows the JS scorer"
    );
}

#[test]
fn the_depth_bound_holds_across_sizes_including_non_powers_of_two() {
    // An odd count carries its tail term forward unpaired, which is the case a
    // power-of-two-only test would miss.
    for n in [3usize, 5, 7, 17, 100, 1023, 1025] {
        let pir = lower(&iid_model(n));
        assert_eq!(
            pir.matches("builtin_logdensityof").count(),
            n,
            "n={n}: expected one density term per observation"
        );
        let want = usize::BITS as usize - (n - 1).leading_zeros() as usize;
        assert_eq!(
            add_nesting_depth(&pir),
            want,
            "n={n}: expected ceil(log2 n) = {want} `add` levels"
        );
    }
}

#[test]
fn the_terms_keep_their_observation_order() {
    // Pairing adjacent terms must not permute them. The observations are distinct
    // here, so the order the density terms' points appear in the emission IS the
    // order of `obs`, whatever the tree above them looks like.
    let src = "\
g = Normal(mu = 0.0, sigma = 1.0)
obs = [10.0, 11.0, 12.0, 13.0, 14.0]
lk = likelihoodof(iid(g, lengthof(obs)), obs)
lp = logdensityof(lk, record())";
    let pir = lower(src);
    let seen: Vec<&str> = ["10.0", "11.0", "12.0", "13.0", "14.0"]
        .into_iter()
        .filter(|lit| pir.contains(lit))
        .collect();
    assert_eq!(
        seen.len(),
        5,
        "every observation must appear exactly once in the emission:\n{pir}"
    );
    let mut at = 0usize;
    for lit in ["10.0", "11.0", "12.0", "13.0", "14.0"] {
        let found = pir[at..]
            .find(lit)
            .unwrap_or_else(|| panic!("`{lit}` out of observation order in:\n{pir}"));
        at += found + lit.len();
    }
}

#[test]
fn one_term_passes_through_and_two_terms_make_one_add() {
    // N <= 2 must be BYTE-IDENTICAL to the left fold, so this change cannot move
    // any existing single- or two-term emission. The expected text is the
    // pre-change output, captured verbatim.
    let one = lower(&iid_model(1));
    assert_eq!(
        add_nesting_depth(&one),
        0,
        "one term needs no `add`:\n{one}"
    );
    assert_eq!(
        one, ONE_TERM_PIR,
        "single-term emission changed; it must pass the term through untouched"
    );

    let two = lower(&iid_model(2));
    assert_eq!(
        add_nesting_depth(&two),
        1,
        "two terms are one `add`:\n{two}"
    );
    assert_eq!(
        two, TWO_TERM_PIR,
        "two-term emission changed; it must stay `add(t0, t1)`"
    );
}

const ONE_TERM_PIR: &str = include_str!("golden/fold_add_one_term.pir");
const TWO_TERM_PIR: &str = include_str!("golden/fold_add_two_terms.pir");
