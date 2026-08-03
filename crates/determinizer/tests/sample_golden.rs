use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

// A DESTRUCTURED rand (`v, s2 = rand(...)`) now lowers to the full spec §07
// `tuple(value, advanced_rng)` — the parser's `v, s2 = rand(...)` sugar
// desugars to `__0x1 = rand(...); v = get(__0x1, 1); s2 = get(__0x1, 2)`
// (1-based integer-literal `get` projections), which now resolve against a
// real tuple instead of hitting the (former) destructure refusal.
#[test]
fn destructured_rand_lowers_to_tuple() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
v, s2 = rand(s, lawof(record(x = x)))
out = v";
    let m = parse_infer(src);
    let out = determinize(&m).expect("destructured rand must lower to a tuple");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(tuple "),
        "expected tuple(value, advanced_rng) for a destructured rand:\n{pir}"
    );
    assert!(
        pir.contains("builtin_sample"),
        "expected a builtin_sample under the tuple:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
}

// rand(rng, lawof(record(x = draw(Normal)))) samples the one draw via
// builtin_sample, threads the rng, and eliminates the measure/stochastic layer.
#[test]
fn single_draw_samples_via_builtin_sample() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-draw rand must lower to builtin_sample");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_sample"),
        "emits builtin_sample:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_sample").count(),
        1,
        "one sample per draw:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("(lawof ") && !pir.contains("(rand "),
        "measure/sample-surface layer eliminated:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
    // Strengthened per the {pir} dump: the rng is threaded in as-is (no fresh
    // rngstate is fabricated), the constructor symbol survives bare (`Normal`,
    // not re-wrapped), and the sampled value is read out via `get0(sample, 0)`
    // (there is no separate `get1` primitive in this codebase — see
    // `sample::build_sample_term`).
    assert!(
        pir.contains("(builtin_sample (%ref self s) Normal ("),
        "builtin_sample threads the rng and carries the bare Normal ctor:\n{pir}"
    );
    assert!(
        pir.contains("(get0"),
        "sampled value is projected via get0(sample, 0):\n{pir}"
    );
    // The draw-binding `x` is now dead (the sampled value is a fresh inline
    // node, not a ref to `x`) and swept by `sweep_dead_measure_bindings`.
    assert!(
        pir.contains("(%bind x 0.0)"),
        "orphaned draw binding x is swept to a harmless literal:\n{pir}"
    );
}

// A positional-arg constructor `Normal(0.0, 1.0)` is equivalent to the keyword
// form `Normal(mu = 0.0, sigma = 1.0)` (spec §04 calling conventions: positional
// args bind to the ordered parameter names). The sample leaf must lower it —
// producing the identical FlatPDL as the keyword form — not refuse. Regression
// for buffy #143 (@sample path).
#[test]
fn sample_draw_positional_constructor_lowers_same_as_keyword() {
    let positional = "\
s = rnginit(0)
x = draw(Normal(0.0, 1.0))
draws = rand(s, lawof(record(x = x)))";
    let keyword = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(x = x)))";
    let out_pos = determinize(&parse_infer(positional)).expect("positional sample leaf must lower");
    let out_kw = determinize(&parse_infer(keyword)).expect("keyword sample leaf must lower");
    let pir_pos = flatppl_flatpir::write(&out_pos);
    let pir_kw = flatppl_flatpir::write(&out_kw);
    assert!(
        pir_pos.contains("builtin_sample")
            && pir_pos.contains("(record (%field mu 0.0) (%field sigma 1.0))"),
        "positional lowers to builtin_sample with the named kernel-input record:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword forms lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
}

// Generality across distributions on the @sample path: `Beta` has params
// ["alpha", "beta"], so `Beta(2.0, 5.0)` positional binds alpha=2.0, beta=5.0 and
// samples via the same builtin_sample as the keyword form. Confirms the sample
// leaf is not Normal-specific. Regression for buffy #143 (non-Gaussian sample
// leaf).
#[test]
fn sample_draw_positional_beta_lowers_same_as_keyword() {
    let positional = "\
s = rnginit(0)
x = draw(Beta(2.0, 5.0))
draws = rand(s, lawof(record(x = x)))";
    let keyword = "\
s = rnginit(0)
x = draw(Beta(alpha = 2.0, beta = 5.0))
draws = rand(s, lawof(record(x = x)))";
    let out_pos =
        determinize(&parse_infer(positional)).expect("positional Beta sample leaf must lower");
    let out_kw = determinize(&parse_infer(keyword)).expect("keyword Beta sample leaf must lower");
    let pir_pos = flatppl_flatpir::write(&out_pos);
    let pir_kw = flatppl_flatpir::write(&out_kw);
    assert!(
        pir_pos.contains("builtin_sample")
            && pir_pos.contains("(record (%field alpha 2.0) (%field beta 5.0))"),
        "positional Beta binds to its ordered params alpha/beta:\n{pir_pos}"
    );
    assert_eq!(
        pir_pos, pir_kw,
        "positional and keyword Beta lower to identical FlatPDL:\npositional:\n{pir_pos}\nkeyword:\n{pir_kw}"
    );
}

// Two independent draws: two builtin_sample calls (field `a`, field `b`), the
// second consuming the first's advanced rng — sequential threading, not two
// fresh streams. There is no separate `get1` primitive in this codebase (see
// `sample::build_sample_term`); the advanced-rng slot is `get0(sample, 1)`.
//
// `a`'s `builtin_sample` node is referenced twice in the graph — once for its
// own sampled value (`get0(sample_a, 0)`, field `a`'s value) and once for the
// rng it hands to `b` (`get0(sample_a, 1)`, threaded into `sample_b`'s first
// arg). The FlatPIR writer is a plain tree printer with no common-subexpression
// sharing (`flatpir::writer` renders full subtrees at every reference site —
// see its module doc), so that shared node's `(builtin_sample (%ref self s)
// Normal ...)` text is re-expanded both places: 3 syntactic "builtin_sample"
// occurrences for 2 logical sample calls. That duplication is a printing
// artifact, not evidence of two independent (unthreaded) rng reads.
#[test]
fn two_independent_draws_thread_the_rng() {
    let src = "\
s = rnginit(0)
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 5.0, sigma = 1.0))
draws = rand(s, lawof(record(a = a, b = b)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("two draws must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "2 logical samples; the first's shared (value, rng) node is re-expanded \
         once more where its rng feeds the second sample (no CSE in the writer):\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "FlatPDL:\n{pir}"
    );
    // `a`'s sample reads the raw `s` rng directly — its call text
    // `(builtin_sample (%ref self s) Normal ...)` appears twice (its own
    // value and, re-expanded, inside `b`'s threaded rng argument).
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s) Normal").count(),
        2,
        "a's sample seeds from the raw rng `s`, appearing at both its reference sites:\n{pir}"
    );
    // `b`'s sample does NOT read a bare/atomic rng argument (which would
    // render unwrapped, e.g. `(builtin_sample (%ref ...` or a literal) — its
    // first argument is the composite `get0(...)` projection of `a`'s
    // advanced rng, which the writer wraps in `%meta` because it is a
    // composite (non-atomic) expression (see `flatpir::writer`'s render_node
    // doc: atomic leaves render bare, composites get a `%meta` wrapper).
    assert_eq!(
        pir.matches("(builtin_sample (%meta").count(),
        1,
        "b's sample takes a composite (threaded) rng argument, not a bare ref:\n{pir}"
    );
    // And that composite argument is specifically a `get0` projection whose
    // own subject is `a`'s `builtin_sample` — i.e. `get0(sample_a, 1)`,
    // exactly the advanced-rng slot `build_sample_term` returns.
    assert!(
        pir.contains("(builtin_sample (%meta (%rngstate %fixed rngstates) (get0"),
        "b's rng argument is a get0(...) projection:\n{pir}"
    );
    let threaded_rng_start = pir
        .find("(builtin_sample (%meta (%rngstate %fixed rngstates) (get0")
        .expect("threaded rng argument located above");
    assert!(
        pir[threaded_rng_start..].contains("(builtin_sample (%ref self s) Normal"),
        "the get0(...) threaded into b's sample projects a's builtin_sample(s, ...):\n{pir}"
    );
}

// Shared ancestor: mu feeds y1 AND y2. mu must be sampled ONCE (3 builtin_sample
// total, not 4), and y1/y2's kernel inputs reference the single mu sample.
//
// Unlike the independent-draws path (which inlines each sample), a SHARED latent
// is rewritten to a single named `builtin_sample` binding (`__sample_mu`) whose
// slots are read by-name — inlining it per consumer would re-draw `mu`, breaking
// shared-ancestor identity (measure-algebra-audit H7/M4). Because the FlatPIR
// writer has no CSE, a by-name ref prints the underlying `builtin_sample` exactly
// once, so the count is a faithful "mu sampled once" check (a 4th occurrence would
// mean `mu` was inlined and re-expanded / re-drawn).
#[test]
fn shared_ancestor_sampled_once() {
    let src = "\
s = rnginit(0)
mu = draw(Normal(mu = 0.0, sigma = 10.0))
y1 = draw(Normal(mu = mu, sigma = 1.0))
y2 = draw(Normal(mu = mu, sigma = 1.0))
draws = rand(s, lawof(record(mu = mu, y1 = y1, y2 = y2)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("hierarchical model must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "mu sampled ONCE + y1 + y2 = 3 (a 4th means mu was re-sampled):\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("(lawof ") && !pir.contains("(rand "),
        "measure/sample-surface layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "FlatPDL:\n{pir}"
    );
    // Strengthened per the {pir} dump. `mu`'s draw-BINDING is rewritten in place to
    // slot 0 of its single sample tuple — not inlined into y1/y2 — so the latent is
    // a shared, named node.
    assert!(
        pir.contains(
            "(%bind mu (%meta ((%scalar real) %fixed reals) (get0 (%ref self __sample_mu) 0)))"
        ),
        "mu's binding is rewritten to slot 0 of its single sample tuple:\n{pir}"
    );
    // `mu` is the ONLY sample seeded from the raw rng `s`; it reads `s` exactly once
    // (no per-consumer re-sample of the shared latent).
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s) Normal").count(),
        1,
        "the shared latent mu draws from the seed rng exactly once:\n{pir}"
    );
    // All THREE consumers of the latent bind `mu = (%ref self mu)` to the one
    // shared `mu` binding by name — y1's kernel input, y2's kernel input, and the
    // output record's `mu` field — never an inlined copy of mu's sample.
    assert_eq!(
        pir.matches("(%field mu (%ref self mu))").count(),
        3,
        "y1, y2, and the output record all reference the single shared mu latent:\n{pir}"
    );
    // The rng threads in dependency order s → mu → y1 → y2: y1's sample consumes
    // mu's advanced rng, y2's consumes y1's.
    assert!(
        pir.contains(
            "(%bind __sample_y1 (%meta ((%tuple (%scalar real) %rngstate) %fixed %unknown) \
             (builtin_sample (%meta (%rngstate %fixed rngstates) (get0 (%ref self __sample_mu) 1))"
        ),
        "y1's sample threads mu's advanced rng (get0(__sample_mu, 1)):\n{pir}"
    );
    assert!(
        pir.contains(
            "(%bind __sample_y2 (%meta ((%tuple (%scalar real) %rngstate) %fixed %unknown) \
             (builtin_sample (%meta (%rngstate %fixed rngstates) (get0 (%ref self __sample_y1) 1))"
        ),
        "y2's sample threads y1's advanced rng (get0(__sample_y1, 1)):\n{pir}"
    );
}

// `lower_record_of_draws_sample`'s M1 guard: a POSITIONAL measure record
// (`record(a)`, no field name) has no key to fold field-by-field under, so it
// is not a field-keyed product — mirrors `density::match_independent_record`'s
// identical guard on the density side (see
// `refuse.rs::measure_record_with_positional_args_refuses`). Unlike the
// density-side companion guard for a non-`%field` named arg (which the parser
// can never produce inside `record(...)` — see `syntax::parser`'s hardcoded
// `NamedKind::Field` for `record`/`table`/`joint`/`jointchain`/`cartprod`
// heads — and so is untested here), this one IS reachable via valid surface
// syntax on the sample path too.
#[test]
fn positional_measure_record_sample_refuses() {
    let src = "\
s = rnginit(0)
a = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(a)))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a positional measure record is not a field-keyed product — refuse");
    assert!(
        err.reason.contains("field-keyed product"),
        "refusal explains the record is not field-keyed: {err:?}"
    );
}

// The other shared-latent shape: one draw-binding `mu` bound to TWO record fields
// (`record(a = mu, b = mu)`). No hierarchy — but the naive per-field fold would
// still sample `mu` once per field (twice), the same shared-ancestor break as the
// hierarchical case (measure-algebra-audit H7/M4). The binding-rewrite path samples
// `mu` ONCE and both fields reference it by name.
#[test]
fn latent_shared_by_two_fields_sampled_once() {
    let src = "\
s = rnginit(0)
mu = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(a = mu, b = mu)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("shared-by-two-fields must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        1,
        "mu bound to two fields is still sampled exactly once:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("(lawof ") && !pir.contains("(rand "),
        "measure/sample-surface layer gone:\n{pir}"
    );
    // mu's draw-binding is rewritten to slot 0 of its single sample; both record
    // fields reference the one `mu` latent by name.
    assert!(
        pir.contains("(get0 (%ref self __sample_mu) 0)"),
        "mu is rewritten to slot 0 of its single sample tuple:\n{pir}"
    );
    assert!(
        pir.contains("(record (%field a (%ref self mu)) (%field b (%ref self mu)))"),
        "both fields a and b reference the single shared mu latent:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "FlatPDL:\n{pir}"
    );
}

// CHAINED rand CALLS (not chained draws under one rand): the second `rand`'s
// draw must consume the FIRST `rand`'s *advanced* rng (destructured out as
// `s2`), never the original `rnginit(...)` source. `refuse.rs`'s
// `destructured_rand_rng_threaded_into_second_rand_lowers` already proves a
// chain of this shape LOWERS (builtin_sample count == 3), but its second
// `rand` is value-terminal and it asserts no threading structure beyond that
// count. This test destructures BOTH rands (spec §07's (value, new_rstate)
// contract applied twice in a row) and proves the load-bearing structural
// piece: the second sample's rng argument resolves through a `get0(...)`
// projection of the FIRST sample, not a second read of `s`.
//
// This is architecturally distinct from `two_independent_draws_thread_the_rng`
// above, which threads rng across two DRAWS folded under a single `rand(...)`
// call (one `lower_rand` invocation, record-fold path); here there are two
// separate top-level `rand(...)` surface calls (two `lower_rand` invocations),
// and the driver's fixpoint (`driver.rs`'s re-scan for `rand` nodes) is what
// must pick up the second one once the first has lowered.
#[test]
fn chained_rand_threads_advanced_rng_not_source() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
v, s2 = rand(s, lawof(record(x = x)))
y = draw(Normal(mu = 1.0, sigma = 1.0))
w, s3 = rand(s2, lawof(record(y = y)))
out = record(v = v, w = w)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("chained destructured rand must lower");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
    // 2 logical samples, 4 textual occurrences — NOT the 3 seen in
    // `two_independent_draws_thread_the_rng`. There, only the VALUE-terminal
    // `rand(...)` result is queried, so a's sample is inlined at just 2 sites
    // (its own value, and where its rng threads into b). HERE both rands are
    // DESTRUCTURED, so `lower_rand` builds `tuple(value, rng_out)` per Task 2
    // — and since the writer has no CSE, materialising a tuple's own two
    // slots (get0(_,0) for value, get0(_,1) for rng) already re-expands its
    // underlying `builtin_sample` TWICE, independent of how many times `v`/
    // `s2` are used downstream. Two destructured rands => 2×2 = 4.
    assert_eq!(
        pir.matches("builtin_sample").count(),
        4,
        "2 destructured rands, each re-expanded twice building its own tuple:\n{pir}"
    );
    // The FIRST sample is the only one seeded from the raw source rng `s` —
    // both occurrences are its own tuple's value/rng slot expansions; `s` is
    // never threaded past this point.
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s) Normal").count(),
        2,
        "only the first sample ever reads the raw source rng `s`:\n{pir}"
    );
    // The SECOND sample's builtin_sample calls all read `s2` — the FIRST
    // rand's destructured advanced-rng output — never `s` directly. Because
    // `s2` is itself a genuine top-level binding (not an anonymous internal
    // projection), the writer treats `(%ref self s2)` as an atomic leaf and
    // prints it bare at the use site (no further inlining there); the proof
    // that `s2` really is the first sample's advanced rng lives in `s2`'s own
    // `%bind` line, checked below.
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s2) Normal").count(),
        2,
        "the second sample's rng argument is always `s2`, never the raw source `s`:\n{pir}"
    );
    // The two counts above must exhaust the total: no third, independent rng
    // source ever feeds a builtin_sample call.
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s) Normal").count()
            + pir.matches("(builtin_sample (%ref self s2) Normal").count(),
        pir.matches("builtin_sample").count(),
        "every builtin_sample call reads either `s` (first sample) or `s2` (second):\n{pir}"
    );
    // The anti-mislowering guard: `s2` is not a fresh/independent rngstate —
    // it is defined as slot 2 (1-based `get`) of `__0x1`, the FIRST rand's
    // own tuple binding. This is the load-bearing check that the second
    // `rand`'s draw is threaded through the first `rand`'s *returned*
    // rngstate, never re-forking from the source `s`.
    assert!(
        pir.contains("(%bind s2 (%meta (%rngstate %fixed rngstates) (get (%ref self __0x1) 2)))"),
        "s2 must be defined as slot 2 (the advanced rng) of the first rand's own tuple:\n{pir}"
    );
    // And `__0x1` (the tuple `s2` projects from) is itself built from a
    // builtin_sample seeded by the raw source `s` — closing the chain
    // s -> __0x1 -> s2 -> (second sample's rng argument).
    let first_tuple_start = pir
        .find("(%bind __0x1 ")
        .expect("the first rand's tuple binding is present");
    let first_tuple_end = pir
        .find("(%bind v ")
        .expect("the tuple binding is followed by v's binding");
    assert!(
        pir[first_tuple_start..first_tuple_end].contains("(builtin_sample (%ref self s) Normal"),
        "__0x1 (the tuple s2 projects from) is seeded from the raw source rng `s`:\n{pir}"
    );
}

// `draw(iid(K, n))` with a FIXED kernel `K` and a STATIC length `n` fans out to
// ONE batched `builtin_sample(rng, ctor, input, n)` — the spec §07
// measure-eval-prims size-dims form: a SINGLE call over the fixed kernel
// produces the length-`n` iid array and ONE advanced rngstate, not one
// `builtin_sample` per element. The writer has no common-subexpression
// sharing (see `lower_shared_record_sample`'s doc), so the ONE logical sample
// node re-expands textually at each `get0` projection (value slot, rng
// slot) — hence the substring count below is 2, not 1, exactly like the
// existing single-scalar destructured-rand goldens.
#[test]
fn iid_fixed_kernel_sample_fans_out() {
    let src = "\
s = rnginit(0)
xs ~ iid(Normal(mu = 0.0, sigma = 1.0), 10)
draws, s2 = rand(s, lawof(xs))
out = draws";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid(K,n) sample must fan out to one builtin_sample");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(%array 1 (10) (%scalar real))"),
        "the fanned variate must carry the array type (not a wrongly-scalar one), \
         mirroring infer's builtin_sample_fanned_variate_is_array:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_sample").count(),
        2,
        "the single iid fan-out sample re-expands at each get0 projection (no CSE in \
         the writer):\n{pir}"
    );
    assert!(
        pir.contains("(builtin_sample (%ref self s) Normal"),
        "expected a Normal builtin_sample seeded by the source rng:\n{pir}"
    );
    assert!(
        pir.contains("1.0))) 10))"),
        "expected the size dim 10 as builtin_sample's trailing arg, right after the \
         kernel_input record:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ")
            && !pir.contains("(iid ")
            && !pir.contains("(lawof ")
            && !pir.contains("(rand "),
        "measure/sample-surface layer eliminated:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
}

// A genuinely non-static `n` (an `external` count) has no compile-time length
// to unroll/batch by — refuse rather than guess a size (refuse-don't-mislower),
// mirroring `density::iid_dynamic_size_refuses`'s density-side counterpart.
#[test]
fn iid_nonstatic_n_sample_refuses() {
    let src = "\
s = rnginit(0)
n = external(posintegers)
xs ~ iid(Normal(mu = 0.0, sigma = 1.0), n)
draws, s2 = rand(s, lawof(xs))
out = draws";
    let m = parse_infer(src);
    determinize(&m).expect_err("non-static iid length must refuse rather than mislower");
}

// `draw(iid(broadcast(Normal, mus, 1.0), n))` — a per-element-differing-params
// kernel (Tier 3, spec §04 broadcasting: an array-of-kernels measure, NOT a
// fixed kernel) — must refuse rather than be silently fanned out as if every
// row drew from the SAME kernel input. `split_constructor` rejects the
// `broadcast(...)` head (it carries positional args, so it is not a bare
// built-in constructor call), which is what turns this into a refusal.
#[test]
fn iid_broadcast_kernel_sample_refuses() {
    let src = "\
s = rnginit(0)
mus = [0.0, 1.0, 2.0]
xs ~ iid(broadcast(Normal, mus, 1.0), 3)
draws, s2 = rand(s, lawof(xs))
out = draws";
    let m = parse_infer(src);
    determinize(&m)
        .expect_err("iid over a broadcast (differing per-element params) kernel must refuse");
}

// ---------------------------------------------------------------------------
// Sample path: composed measures in measure position, and derived record fields
// ---------------------------------------------------------------------------

// §07 `rand(rstate, m)`: "generates a random value from a closed measure `m`".
// A primitive constructor call is a nullary kernel — a CLOSED measure (§06
// "Fundamental measures and measure algebra": operations that map measures to
// values "require closed measures (i.e. nullary kernels)") — so it needs no
// `lawof` wrapper, and §07's only stated exclusions are non-constant weighting
// and multivariate truncation. A `refuse.rs` test previously pinned the
// opposite ("its second argument must be a lawof(...)"), which was the gate
// defect stated as a requirement. This is also the base case that terminates
// the `pushfwd` arm's recursion on its base.
#[test]
fn rand_of_a_bare_closed_measure_samples() {
    let src = "\
s = rnginit([42, 0, 0, 0])
draws = rand(s, Normal(mu = 0.0, sigma = 1.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("a bare closed measure is admissible per §07");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        1,
        "one sample from the closed measure:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// §07 types `rand`'s second argument as a "closed measure", and §06 makes
// `pushfwd(f, M)` of a closed measure closed. Sampling a pushforward is the
// textbook easy case — sample the base, apply the map, no inverse and no Jacobian
// — but `lower_rand` gated on a literal `lawof` head before dispatching, so no
// composed measure was reachable. This is §06's own log-normal example.
#[test]
fn rand_of_pushfwd_samples_base_then_applies_map() {
    let src = "\
s = rnginit([42, 0, 0, 0])
draws = rand(s, pushfwd(exp, Normal(mu = 0.0, sigma = 1.0)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("rand of a pushfwd must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        1,
        "one base sample:\n{pir}"
    );
    assert!(pir.contains("exp"), "map applied forward:\n{pir}");
    assert!(
        !pir.contains("pushfwd") && !pir.contains("lawof"),
        "measure layer gone:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// §13 "Output reduction": "A sampled output resolves its measure's `draw` nodes
// through `rand`" and "Other deterministic expressions pass through unchanged".
// A derived field over two sampled latents is exactly such an expression. Note the
// shared `mu` must still be sampled ONCE (shared-ancestor identity) — three latents
// means three `builtin_sample` calls, not four.
#[test]
fn derived_record_field_samples_over_shared_latents() {
    let src = "\
s = rnginit([42, 0, 0, 0])
mu = draw(Normal(mu = 0.0, sigma = 10.0))
y1 = draw(Normal(mu = mu, sigma = 1.0))
y2 = draw(Normal(mu = mu, sigma = 1.0))
d = y1 - y2
draws = rand(s, lawof(record(mu = mu, y1 = y1, y2 = y2, d = d)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("a derived record field must sample");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "mu sampled ONCE and shared by y1/y2; d is deterministic:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// §07's own worked `rand` example is a bare `iid` in measure position:
//
//   random_data, rstate2 = rand(rstate, iid(Normal(0, 1), 10))
//   more_random_data, rstate3 = rand(rstate2, iid(Exponential(1), 5))
//
// A fixed-kernel `iid(K, n)` is itself a nullary kernel, hence a closed measure —
// the same §07 clause that admits a bare constructor. It must lower, and it must
// lower to the SAME batched term as the `xs ~ iid(...); rand(s, lawof(xs))`
// spelling: ONE `builtin_sample` carrying the size dim, with a SINGLE advanced
// rngstate (§07 measure-eval-prims), never one sample per element.
#[test]
fn rand_of_bare_iid_matches_the_draw_spelling() {
    let measure_position = "\
s = rnginit(0)
draws, s2 = rand(s, iid(Normal(mu = 0.0, sigma = 1.0), 10))
out = draws";
    let draw_position = "\
s = rnginit(0)
xs ~ iid(Normal(mu = 0.0, sigma = 1.0), 10)
draws, s2 = rand(s, lawof(xs))
out = draws";

    let pir_of = |src: &str| {
        let m = parse_infer(src);
        let out = determinize(&m).expect("a fixed-kernel iid must sample in either spelling");
        assert!(
            flatppl_determinizer::is_flatpdl(&out).is_ok(),
            "must be FlatPDL:\n{}",
            flatppl_flatpir::write(&out)
        );
        flatppl_flatpir::write(&out)
    };
    let measure_pir = pir_of(measure_position);
    let draw_pir = pir_of(draw_position);

    // The emitted sample term must be byte-identical between the two spellings —
    // same kernel, same kernel_input record, same trailing size dim, same rng, and
    // (since the `%meta` annotations are compared too) the same inferred types.
    // Both spellings route through `sample.rs`'s single `lower_iid_sample`; this is
    // what would catch them drifting apart.
    let measure_term = first_sample_term(&measure_pir);
    let draw_term = first_sample_term(&draw_pir);
    assert_eq!(
        measure_term, draw_term,
        "the two spellings must emit the identical batched sample term"
    );
    assert!(
        measure_term.contains("(builtin_sample (%ref self s) Normal"),
        "seeded by the source rng:\n{measure_term}"
    );
    assert!(
        measure_term.ends_with(" 10)"),
        "the size dim 10 is builtin_sample's trailing arg:\n{measure_term}"
    );

    // ONE batched sample, not ten. The count is 2 rather than 1 because the
    // writer has no CSE, so the single sample node re-expands at each of the two
    // `get0` projections (value + advanced rng) — exactly what the pre-existing
    // `iid_fixed_kernel_sample_fans_out` golden pins for the draw spelling.
    assert_eq!(
        measure_pir.matches("builtin_sample").count(),
        draw_pir.matches("builtin_sample").count(),
        "the two spellings must emit the same number of samples:\n{measure_pir}\n---\n{draw_pir}"
    );
    assert_eq!(
        measure_pir.matches("builtin_sample").count(),
        2,
        "one batched fan-out sample, re-expanded at its two get0 projections — NOT \
         one sample per element:\n{measure_pir}"
    );
    assert!(
        measure_pir.contains("(%array 1 (10) (%scalar real))"),
        "the fanned variate carries the array type:\n{measure_pir}"
    );
    assert!(
        !measure_pir.contains("(iid ") && !measure_pir.contains("(rand "),
        "measure layer gone:\n{measure_pir}"
    );
}

/// The first `(builtin_sample …)` sub-term of `pir`, as a balanced substring —
/// used to compare what two spellings of the same measure actually emitted,
/// including their `%meta` type annotations.
fn first_sample_term(pir: &str) -> String {
    let start = pir
        .find("(builtin_sample")
        .unwrap_or_else(|| panic!("no builtin_sample term in:\n{pir}"));
    let mut depth = 0usize;
    for (i, b) in pir.bytes().enumerate().skip(start) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return pir[start..=i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced builtin_sample term in:\n{pir}");
}

// The capability this whole vertical exists for, in the shape the guards around it
// are scoped against: `pushfwd` over a RECORD law, with a map whose parameter names
// are the record's fields (§04 auto-splatting). The scalar-base golden above does
// not cover it, and three `refuse.rs` tests fence this neighbourhood — so without
// this golden a future tightening of `record_splat_mismatch` could make the correct
// spelling refuse with every other test still green, silently removing the feature.
//
// Two INDEPENDENT draws in the base: 2 logical samples, 3 syntactic occurrences.
// The first sample's node is shared — its value is `sub`'s left operand and its
// advanced rng seeds the second sample — and the FlatPIR writer has no
// common-subexpression sharing, so it re-expands at both sites. Exactly the
// 3-for-2 artifact `two_independent_draws_thread_the_rng` documents.
#[test]
fn pushfwd_over_independent_record_law_applies_map_to_both_draws() {
    let src = "\
s = rnginit([42, 0, 0, 0])
y1 = draw(Normal(mu = 0.0, sigma = 1.0))
y2 = draw(Normal(mu = 0.0, sigma = 1.0))
f = functionof(_a_ - _b_, y1 = _a_, y2 = _b_)
draws = rand(s, pushfwd(f, lawof(record(y1 = y1, y2 = y2))))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("pushfwd over a record law with a field-named map must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "2 logical samples; the first's shared (value, rng) node re-expands where its \
         rng seeds the second (no CSE in the writer):\n{pir}"
    );
    assert!(
        pir.contains("(sub "),
        "the map is applied FORWARD to the sampled record:\n{pir}"
    );
    // The threading is what distinguishes 2 logical samples from 2 independent
    // reads of `s`. Same idiom as `two_independent_draws_thread_the_rng`: the
    // first sample reads the raw `s` and its text appears at both its reference
    // sites; the second takes a COMPOSITE (`%meta`-wrapped) rng argument, which is
    // the `get0(sample_1, 1)` advanced slot.
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s) Normal").count(),
        2,
        "the first sample seeds from the raw rng `s`, at both its reference sites:\n{pir}"
    );
    assert_eq!(
        pir.matches("(builtin_sample (%meta").count(),
        1,
        "the second sample takes a threaded (composite) rng argument, not a bare ref:\n{pir}"
    );
    assert!(
        pir.contains("(builtin_sample (%meta (%rngstate %fixed rngstates) (get0"),
        "and that argument is the get0(...) advanced-rng projection:\n{pir}"
    );
    assert!(
        !pir.contains("(pushfwd ") && !pir.contains("(lawof ") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// Same map, but the base record law has a SHARED latent (`mu` feeding both `y1`
// and `y2`). This is where a re-inlined latent would hide, and where the earlier
// version of these goldens had no `builtin_sample` count assertion at all.
//
// 3 latents, 3 samples, and here the count is EXACT rather than inflated: the
// shared-latent path binds each sample to its own synthetic name, so each appears
// once and consumers read it by name. `mu` re-drawn per consumer would show up as
// a fourth occurrence.
#[test]
fn pushfwd_over_shared_latent_record_law_samples_each_latent_once() {
    let src = "\
s = rnginit([42, 0, 0, 0])
mu = draw(Normal(mu = 0.0, sigma = 10.0))
y1 = draw(Normal(mu = mu, sigma = 1.0))
y2 = draw(Normal(mu = mu, sigma = 1.0))
f = functionof(_a_ - _b_, y1 = _a_, y2 = _b_)
draws = rand(s, pushfwd(f, lawof(record(y1 = y1, y2 = y2))))";
    let m = parse_infer(src);
    let out =
        determinize(&m).expect("pushfwd over a shared-latent record law must lower, sampling once");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "mu, y1, y2 — one sample each, mu NOT re-drawn per consumer:\n{pir}"
    );
    // The shared latent is bound to one synthetic name and read by name; two
    // `builtin_sample` bindings for `mu` would mean shared-ancestor identity broke.
    assert_eq!(
        pir.matches("%bind __sample_mu ").count(),
        1,
        "mu's sample is a single named binding:\n{pir}"
    );
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s)").count(),
        1,
        "only the root latent reads the source rng; the rest are threaded:\n{pir}"
    );
    assert!(
        pir.contains("(sub "),
        "the map is applied FORWARD over the shared-latent draws:\n{pir}"
    );
    assert!(
        !pir.contains("(pushfwd ") && !pir.contains("(lawof ") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}
