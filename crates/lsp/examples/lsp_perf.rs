//! Dependency-free performance measurement for the LSP hot paths changed on
//! the `lsp/correctness-perf` branch. Run with:
//!
//!     cargo run --example lsp_perf --release -p flatppl-lsp
//!
//! Reports, on a large synthetic model:
//!   1. per-hover offset→node lookup: old `Module::node_at_offset` linear scan
//!      vs new `node_at_offset_indexed` over the memoized span index;
//!   2. the once-per-revision span-index build cost (the per-edit overhead the
//!      lookup win is amortized against);
//!   3. single `analyze` (parse + infer) latency — quantifies the debounce win,
//!      since a coalesced burst of N keystrokes saves (N-1) of these.
//!
//! Timing is min-of-repeats (the cleanest signal for a CPU-bound op); methodology
//! is printed so the numbers are reproducible, not asserted.

use flatppl_core::Idx;
use flatppl_lsp::db::{Catalogues, Database, FileSet, SourceFile};
use flatppl_lsp::queries::{analyze, node_at_offset_indexed, node_span_index};
use std::hint::black_box;
use std::time::Instant;

/// A large synthetic model: `n` chained bindings, each a nested arithmetic
/// expression, so node_count grows ~linearly into the thousands.
fn big_source(n: usize) -> String {
    let mut s = String::from("x0 = 1.0\n");
    for i in 1..n {
        let p = i - 1;
        s.push_str(&format!("x{i} = add(mul(x{p}, 2.0), sqrt(abs(x{p})))\n"));
    }
    s
}

/// Minimum wall-clock of `iters` runs of `f`, in nanoseconds.
fn min_ns(iters: u32, mut f: impl FnMut()) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_nanos());
    }
    best
}

fn main() {
    let n_bindings = 1500;
    let src = big_source(n_bindings);
    let len = src.len() as u32;

    let db = Database::default();
    let file = SourceFile::new(&db, "perf.flatppl".to_string(), src.clone());
    let fs = FileSet::new(&db, Vec::new());
    let cats = Catalogues::new(&db, Vec::new());

    let analyzed = analyze(&db, file, fs, cats);
    let module = analyzed.module(&db).expect("model analyzes");
    let node_count = module.node_count();

    // Sample offsets spread across the source — what a user's cursor hits.
    let n_offsets = 2000u32;
    let offsets: Vec<u32> = (0..n_offsets).map(|i| i * len / n_offsets).collect();

    // (1a) OLD: linear scan, called fresh per hover (no index).
    let old_lookup = min_ns(20, || {
        let mut acc = 0usize;
        for &off in &offsets {
            if let Some(id) = module.node_at_offset(black_box(off)) {
                acc += id.index();
            }
        }
        black_box(acc);
    });

    // (1b) NEW: lookup over the prebuilt, memoized span index.
    let index = node_span_index(&db, file, fs, cats);
    let new_lookup = min_ns(20, || {
        let mut acc = 0usize;
        for &off in &offsets {
            if let Some(id) = node_at_offset_indexed(black_box(&index), black_box(off)) {
                acc += id.index();
            }
        }
        black_box(acc);
    });

    // (2) Span-index MARGINAL build cost over the analyze that already runs for
    // diagnostics. analyze is needed regardless; the index adds only a span
    // collect + sort. Measure the first node_span_index call on a db where
    // analyze is already warm (salsa memoizes both, so this is a one-shot).
    let build_marginal_ns = {
        let db = Database::default();
        let f = SourceFile::new(&db, "perf.flatppl".to_string(), src.clone());
        let fs = FileSet::new(&db, Vec::new());
        let cats = Catalogues::new(&db, Vec::new());
        black_box(
            analyze(&db, f, fs, cats)
                .module(&db)
                .map(|m| m.node_count()),
        ); // warm analyze
        let t = Instant::now();
        black_box(node_span_index(&db, f, fs, cats));
        t.elapsed().as_nanos()
    };

    // (3) Single analyze (parse + infer) on a fresh db — the cost a debounced
    // burst pays once instead of per keystroke.
    let analyze_ns = min_ns(10, || {
        let db = Database::default();
        let f = SourceFile::new(&db, "perf.flatppl".to_string(), src.clone());
        let fs = FileSet::new(&db, Vec::new());
        let cats = Catalogues::new(&db, Vec::new());
        let a = analyze(&db, f, fs, cats);
        black_box(a.module(&db).map(|m| m.node_count()));
    });

    let per = |total: u128| total as f64 / n_offsets as f64;
    println!("model: {n_bindings} bindings, {node_count} nodes, {len} bytes");
    println!("offset lookups per run: {n_offsets} (min of 20 runs)\n");
    println!("(1) offset->node lookup, per call:");
    println!("    old linear scan      : {:.0} ns", per(old_lookup));
    println!("    new indexed lookup   : {:.0} ns", per(new_lookup));
    println!(
        "    speedup              : {:.1}x",
        per(old_lookup) / per(new_lookup).max(0.001)
    );
    println!();
    println!(
        "(2) span-index marginal build (collect+sort, over the analyze that runs anyway): {:.3} ms",
        build_marginal_ns as f64 / 1e6
    );
    let saving_per_lookup = (per(old_lookup) - per(new_lookup)).max(0.001);
    println!(
        "    break-even: {:.0} hovers per edit amortize the build",
        build_marginal_ns as f64 / saving_per_lookup
    );
    println!();
    println!(
        "(3) single analyze (parse+infer): {:.2} ms",
        analyze_ns as f64 / 1e6
    );
    println!(
        "    debounced 10-keystroke burst saves ~{:.1} ms (9 x analyze)",
        9.0 * analyze_ns as f64 / 1e6
    );
}
