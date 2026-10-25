//! A module bound to another name is the same module.
//!
//! §04 *Aliasing*: two `load_module` calls with the same source "denote the
//! same module, so binding each to a name is equivalent to loading once and
//! aliasing the module reference". So `H = h; H.kallen(…)` must infer and lower
//! exactly as `h.kallen(…)` does, through any chain of aliases.

use std::sync::Arc;

use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

fn bundle() -> ModuleBundle {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "sub.flatppl",
        Arc::new(flatppl_syntax::parse("a = 2.0\nd = Normal(mu = a, sigma = 2.0)").unwrap()),
    );
    bundle
}

/// The inferred type of each of `names`, and the determinized module text.
fn lower(source: &str, names: &[&str]) -> (Vec<String>, String) {
    let bundle = bundle();
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = infer_module(&mut module, &bundle, Level::Shape)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{source}");
    let types = names
        .iter()
        .map(|name| {
            let sym = module.intern(name);
            let rhs = module.binding(module.binding_by_name(sym).unwrap()).rhs;
            format!("{:?}", module.type_of(rhs))
        })
        .collect();
    let out = flatppl_determinizer::determinize_with(&module, &bundle)
        .unwrap_or_else(|e| panic!("{}\n{source}", e.reason));
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    (types, flatppl_flatpir::write(&out))
}

/// The binding `name` in determinized FlatPIR.
fn binding<'a>(pir: &'a str, name: &str) -> &'a str {
    let at = pir
        .find(&format!("(%bind {name} "))
        .unwrap_or_else(|| panic!("no `{name}` in\n{pir}"));
    pir[at..].lines().next().unwrap()
}

/// `direct` loads the module as `m`; `chain` binds `M = M1`, `M1 = m`.
fn assert_alias_lowers_like_direct(load: &str, body: &str, names: &[&str]) {
    let direct = format!("m = {load}\n{}", body.replace("M.", "m."));
    let chain = format!("m = {load}\nM1 = m\nM = M1\n{body}");
    let (direct_types, direct_pir) = lower(&direct, names);
    let (chain_types, chain_pir) = lower(&chain, names);
    assert_eq!(chain_types, direct_types);
    for name in names {
        assert_eq!(binding(&chain_pir, name), binding(&direct_pir, name));
    }
}

/// A standard-module function: `kallen(1, 2, 3) = 1 + 4 + 9 - 2(2 + 3 + 6) = -8`.
#[test]
fn standard_module_function_call() {
    assert_alias_lowers_like_direct(
        "standard_module(\"particle-physics\", \"0.1\")",
        "v = M.kallen(1.0, 2.0, 3.0)",
        &["v"],
    );
}

/// A standard-module distribution scored through the alias.
#[test]
fn standard_module_distribution() {
    assert_alias_lowers_like_direct(
        "standard_module(\"particle-physics\", \"0.1\")",
        "lp = logdensityof(M.CrystalBall(m0 = 0.0, sigma = 1.0, alpha = 1.5, n = 3.0), 0.5)",
        &["lp"],
    );
}

/// A loaded module's value and measure members. `logpdf(Normal(2, 2), 0.5) =
/// -1.8401465`, the direct spelling's number.
#[test]
fn loaded_module_members() {
    assert_alias_lowers_like_direct(
        "load_module(\"sub.flatppl\")",
        "v = M.a * 3.0\nlp = logdensityof(M.d, 0.5)",
        &["v", "lp"],
    );
}
