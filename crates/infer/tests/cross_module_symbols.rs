//! Names crossing the `load_module` boundary keep their spelling (spec §11).
//!
//! A `Symbol` is an index into ONE module's interner, and the importer and each
//! dependency intern independently, so a `Type` handed across the boundary
//! unchanged re-reads every name it carries as whatever the receiving interner
//! holds at that index. §11 is normative about the slot that publishes such a
//! list:
//!
//! > `(%kernel (%inputs <name> ...) (%mass <mass>))` — user-defined transition
//! > kernels. The `%inputs` names are the callable's input names
//!
//! Both directions are covered: the dependency's types travelling out to the
//! importer, and the importer's substitution types travelling in.

use std::sync::Arc;

use flatppl_infer::{Diagnostic, Level, ModuleBundle, infer_module};

fn xmod(dep_path: &str, dep_src: &str, model_src: &str) -> (String, Vec<Diagnostic>) {
    let dep = flatppl_syntax::parse(dep_src).expect("dependency parses");
    let mut bundle = ModuleBundle::new();
    bundle.insert(dep_path, Arc::new(dep));
    let mut model = flatppl_syntax::parse(model_src).expect("model parses");
    let diags = infer_module(&mut model, &bundle, Level::Shape);
    (flatppl_flatpir::write(&model), diags)
}

fn binding_line<'a>(out: &'a str, name: &str) -> &'a str {
    out.lines()
        .find(|l| l.contains(&format!("(%bind {name} ")))
        .unwrap_or_else(|| panic!("no `{name}` binding in:\n{out}"))
}

/// A `joint` over a local and a cross-module kernel publishes the union of both
/// input lists. The dependency's name must appear as the DEPENDENCY spelled it,
/// so the correct application types and the misspelled one is rejected.
///
/// Before the boundary re-intern, `d.FK`'s input `mu_unique_alpha` read as
/// `reals` (the importer's entry at the dependency's index), which rejected this
/// correct call with "`JJ` has no parameter `mu_unique_alpha` (declares: `zz`,
/// `reals`)" and accepted `JJ(zz = 0.0, reals = 1.0)` instead.
#[test]
fn cross_module_kernel_input_names_survive_the_boundary() {
    let dep = "mu_unique_alpha = elementof(reals)\n\
               FK = functionof(Normal(mu = mu_unique_alpha, sigma = 1.0), \
               mu_unique_alpha = mu_unique_alpha)";
    let importer = "d = load_module(\"dep.flatppl\")\n\
                    zz = elementof(reals)\n\
                    LK = functionof(Normal(mu = zz, sigma = 1.0), zz = zz)\n\
                    JJ = joint(p = LK, q = d.FK)\n\
                    appl = JJ(zz = 0.0, mu_unique_alpha = 1.0)";
    let (out, diags) = xmod("dep.flatppl", dep, importer);

    let jj = binding_line(&out, "JJ");
    assert!(
        jj.contains("(%inputs zz mu_unique_alpha)"),
        "§11: the `%inputs` names are the callable's input names; got:\n{jj}"
    );
    assert!(
        diags.iter().all(|d| !d.message.contains("no parameter")),
        "the correct application must not be rejected, got: {diags:?}"
    );

    // The check is not a blanket opt-out: the name the corruption used to
    // publish is now rejected.
    let (_, wrong) = xmod(
        "dep.flatppl",
        dep,
        "d = load_module(\"dep.flatppl\")\n\
         zz = elementof(reals)\n\
         LK = functionof(Normal(mu = zz, sigma = 1.0), zz = zz)\n\
         JJ = joint(p = LK, q = d.FK)\n\
         appl = JJ(zz = 0.0, reals = 1.0)",
    );
    assert!(
        wrong
            .iter()
            .any(|d| d.message.contains("has no parameter `reals`")),
        "a name no callable declares must still be rejected, got: {wrong:?}"
    );
}

/// The repo's own cross-module kernel fixture: `xmodule_kernel_helper` declares
/// `center`, which read as `reals` on the importing side.
#[test]
fn fixture_helper_declared_name_survives_the_boundary() {
    let (out, diags) = xmod(
        "xmodule_kernel_helper.flatppl",
        "center = elementof(reals)\n\
         obs_kernel = functionof(Normal(mu = center, sigma = 1.0), center = center)",
        "m = load_module(\"xmodule_kernel_helper.flatppl\")\n\
         loc = elementof(reals)\n\
         LKK = functionof(Normal(mu = loc, sigma = 1.0), loc = loc)\n\
         JU = joint(p = LKK, q = m.obs_kernel)",
    );
    assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    let ju = binding_line(&out, "JU");
    assert!(
        ju.contains("(%inputs loc center)"),
        "the helper's declared `center` must survive the crossing; got:\n{ju}"
    );
}

/// `Type::Record` field names corrupt the same way, and nothing errors — a
/// positional `joint` merging a cross-module record variate published the
/// importer's names at the dependency's indices.
#[test]
fn merged_cross_module_record_domain_keeps_its_field_names() {
    let (out, diags) = xmod(
        "dep5.flatppl",
        "MR = joint(zeta_alpha_uniq = Normal(mu = 0.0, sigma = 1.0), \
         eta_beta_uniq = Exponential(rate = 1.0))",
        "d = load_module(\"dep5.flatppl\")\n\
         LOC = joint(loc_one = Normal(mu = 0.0, sigma = 1.0))\n\
         JB = joint(d.MR, LOC)",
    );
    assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    let jb = binding_line(&out, "JB");
    assert!(
        jb.contains(
            "(%record (zeta_alpha_uniq (%scalar real)) \
             (eta_beta_uniq (%scalar real)) (loc_one (%scalar real)))"
        ),
        "the merged domain must keep the dependency's field names; got:\n{jb}"
    );
    assert!(
        jb.contains("(record (zeta_alpha_uniq reals) (eta_beta_uniq reals) (loc_one reals))"),
        "the merged value-set must keep them too; got:\n{jb}"
    );
}

/// The inbound direction: a substituted record's field names are the IMPORTER's
/// symbols, so they are re-interned into the dependency before its walk. Without
/// that, the dependency annotates with foreign indices and the outbound
/// translation reads them against the wrong interner and panics.
///
/// Unlike the three tests above, this one passes before the fix too — the
/// inbound corruption had no observable output of its own, and this pins that
/// the outbound translation does not turn it into a crash.
#[test]
fn substituted_record_names_do_not_leak_into_the_dependency() {
    let (out, diags) = xmod(
        "dep6.flatppl",
        "p = elementof(cartprod(zz1 = reals, zz2 = reals))\nq = p",
        "d = load_module(\"dep6.flatppl\", p = mk)\n\
         filler_one = 1.0\n\
         filler_two = 2.0\n\
         mk = record(alpha_u = elementof(reals), beta_u = elementof(reals))\n\
         usee = d.q",
    );
    assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
    assert!(out.contains("(%bind usee"), "usee missing from:\n{out}");
}
