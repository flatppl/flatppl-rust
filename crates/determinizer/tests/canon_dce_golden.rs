use flatppl_determinizer::determinize_with_roots;
use flatppl_infer::ModuleBundle;

fn determinize_roots(src: &str, roots: &[&str]) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let syms: Vec<flatppl_core::Symbol> = roots.iter().map(|r| m.intern(r)).collect();
    determinize_with_roots(&m, &ModuleBundle::new(), Some(&syms)).expect("must lower")
}

// With a requested-output root, DCE removes bindings unreachable from it — the
// dead measure-layer stubs and unreferenced value bindings vanish entirely
// (not zeroed to 0.0). The root itself and its transitive deps survive. #263 Pass 4-A.
#[test]
fn dce_removes_unreachable_keeps_root_and_deps() {
    let src = "\
a = draw(Normal(0.0, 1.0))
dead1 = 42.0
dead2 = record(x = 1.0)
__score__ = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_roots(src, &["__score__"]);
    let pir = flatppl_flatpir::write(&out);
    assert!(pir.contains("__score__"), "root kept:\n{pir}");
    assert!(
        !pir.contains("dead1") && !pir.contains("dead2"),
        "unreachable bindings removed, not zeroed:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// roots=None (via the existing determinize()) preserves today's keep-all
// behavior — dead bindings survive (zeroed by Pass 1's sweep or as-is). No regression.
#[test]
fn dce_none_roots_keeps_all_bindings() {
    let src = "\
a = draw(Normal(0.0, 1.0))
keeper = 42.0
__score__ = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = flatppl_determinizer::determinize(&m).expect("must lower");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("keeper"),
        "roots=None keeps all bindings:\n{pir}"
    );
}

// DCE is deterministic: determinizing the same source with the same roots
// twice produces byte-identical FlatPIR (after the first run, every survivor
// is reachable, so a second `retain_reachable` is a no-op). #263 Pass 4-A.
#[test]
fn dce_is_idempotent() {
    let src = "\
a = draw(Normal(0.0, 1.0))
dead1 = 42.0
__score__ = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let once = flatppl_flatpir::write(&determinize_roots(src, &["__score__"]));
    let twice = flatppl_flatpir::write(&determinize_roots(src, &["__score__"]));
    assert_eq!(once, twice, "DCE output is a fixpoint");
}
