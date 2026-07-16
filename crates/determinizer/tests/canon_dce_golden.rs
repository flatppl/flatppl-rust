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

// Pass 4 Task A review Fix 1: a binding reachable ONLY through a `functionof`
// reification `Inputs` boundary entry — never in an ordinary RHS subtree —
// must survive root-based DCE. Mirrors the scar `driver.rs` guards against for
// the mid-loop dead-binding sweep
// (`sweep_preserves_binding_referenced_only_via_reification_input`), here
// exercised through the public `determinize_with_roots` entry point and the
// final canon-DCE pass instead. `k`'s explicit boundary spec `g = g` (spec §04
// "a boundary node ... becomes disconnected from the output in the substituted
// graph ... permitted, not an error") closes over `g` as a genuinely UNUSED
// input: `k`'s reified body (`2.0`) never references `g` at all, so after
// `resolve_alias_refs`/DCE the only surviving reference to `g` anywhere in the
// module is the `(g (%ref self g))` `%specinputs` entry — `children()`
// deliberately excludes a `Call`'s `Inputs` bucket, so a body-only reachability
// walk would judge `g` dead and drop it, stranding `k`'s surviving reification
// input as a dangling `%ref`. (Verified non-vacuous: temporarily disabling the
// `Inputs`-scanning arm in `collect_referenced_names` makes this test fail,
// with `g` dropped and `k`'s input left dangling.)
#[test]
fn dce_keeps_binding_referenced_only_via_reification_input() {
    let src = "\
g = 3.0
dead = 42.0
k = functionof(2.0, g = g)
__score__ = k";
    let out = determinize_roots(src, &["__score__"]);
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(%bind g 3.0)"),
        "g reachable only via k's reification Inputs boundary must survive:\n{pir}"
    );
    assert!(
        !pir.contains("dead"),
        "genuinely unreachable binding still removed:\n{pir}"
    );
}
