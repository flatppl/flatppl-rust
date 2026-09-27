//! Preserve pyhf parameter identity across FlatPPL's narrower name grammar.

use crate::builder::{Builder, check_binding_name, sanitize_ident};
use crate::model::{PyhfDocument, PyhfParam};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn rename_parameters(doc: &mut PyhfDocument) -> BTreeMap<String, String> {
    let declared: BTreeSet<_> = doc
        .channels()
        .iter()
        .flat_map(|c| &c.samples)
        .flat_map(|s| &s.modifiers)
        .filter_map(|m| m.effective_param())
        .collect();
    let mut used = declared.clone();
    // Unused config entries and dangling POIs must not capture a renamed binding.
    used.extend(doc.parameters().iter().map(|p| p.name.clone()));
    for measurement in doc.measurements() {
        used.extend(measurement.config.poi.iter().cloned());
        used.extend(measurement.config.parameters.iter().map(|p| p.name.clone()));
    }
    used.extend(["flatppl_compat", "hepphys", "likelihood"].map(str::to_owned));
    let mut names = BTreeMap::new();
    for source in declared {
        if check_binding_name(&source, "pyhf parameter").is_ok() {
            continue;
        }
        let base = sanitize_ident(&source);
        let mut target = base.clone();
        let mut suffix = 2;
        while used.contains(&target) || check_binding_name(&target, "pyhf parameter").is_err() {
            target = format!("{base}_{suffix}");
            suffix += 1;
        }
        used.insert(target.clone());
        names.insert(source, target);
    }
    if names.is_empty() {
        return names;
    }
    let channels = match doc {
        PyhfDocument::Model(model) => {
            rename_config(&mut model.parameters, &names);
            &mut model.channels
        }
        PyhfDocument::Workspace(workspace) => {
            for measurement in workspace.measurements.iter_mut().chain(
                workspace
                    .toplvl
                    .iter_mut()
                    .flat_map(|t| &mut t.measurements),
            ) {
                if let Some(poi) = &mut measurement.config.poi
                    && !poi.is_empty()
                {
                    rename(poi, &names);
                }
                rename_config(&mut measurement.config.parameters, &names);
            }
            &mut workspace.channels
        }
    };
    for modifier in channels
        .iter_mut()
        .flat_map(|c| &mut c.samples)
        .flat_map(|s| &mut s.modifiers)
    {
        if let Some(source) = modifier.effective_param()
            && let Some(target) = names.get(&source)
        {
            modifier.parameter = Some(target.clone());
        }
    }
    names
}

fn rename(name: &mut String, names: &BTreeMap<String, String>) {
    if let Some(target) = names.get(name) {
        *name = target.clone();
    }
}

fn rename_config(parameters: &mut [PyhfParam], names: &BTreeMap<String, String>) {
    for parameter in parameters {
        rename(&mut parameter.name, names);
    }
}

pub(crate) fn emit_metadata(b: &mut Builder, names: &BTreeMap<String, String>) {
    if names.is_empty() {
        return;
    }
    let fields: Vec<_> = names
        .iter()
        .map(|(source, target)| (target.as_str(), b.str_lit(source)))
        .collect();
    let record = b.call_fields("record", &fields);
    b.bind_unique_doc(
        "pyhf_parameter_names",
        record,
        "Original pyhf parameter names for renamed bindings.",
    );
}
