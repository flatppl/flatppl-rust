//! Documents that pyhf itself refuses, and so must the importer.
//!
//! Each document below was run through pyhf 0.7.6 (`pyhf.Workspace(spec).model()`);
//! every test names the exception pyhf raises and quotes its message. The oracle
//! is the foreign tool, so no spec decision is involved: a document pyhf will not
//! build a model from cannot have a faithful FlatPPL lowering either.
//!
//! Without these checks all three documents converted to exit 0 and wrote a
//! module that `infer` and `lint` also accepted.

/// Assert that `read_pyhf(json)` errors and the message contains every needle.
fn assert_err_pyhf(label: &str, json: &str, needles: &[&str]) {
    match flatppl_hs3::read_pyhf(json) {
        Ok(_) => panic!("{label}: expected Err, got Ok"),
        Err(e) => {
            let msg = e.to_string();
            for needle in needles {
                assert!(
                    msg.contains(needle),
                    "{label}: error message should mention `{needle}`, got: {msg}"
                );
            }
        }
    }
}

/// A `normfactor` and a `shapefactor` sharing the name `k`.
///
/// pyhf: `pyhf.exceptions.InvalidNameReuse` — "Multiple values for
/// 'n_parameters' ([1, 2]) were found for k. Use unique modifier names when
/// constructing the pdf."
///
/// The importer used to emit `k = elementof(reals)` for channel `chA` and
/// `k = elementof(cartpow(posreals, 2))` for channel `chB`: one name, two
/// contradictory domains, in one module.
#[test]
fn shared_name_with_incompatible_domains_errs() {
    assert_err_pyhf(
        "normfactor_vs_shapefactor",
        r#"{"channels":[
             {"name":"chA","samples":[{"name":"sig","data":[5.0,10.0],
                "modifiers":[{"name":"k","type":"normfactor","data":null}]}]},
             {"name":"chB","samples":[{"name":"sig","data":[7.0,3.0],
                "modifiers":[{"name":"k","type":"shapefactor","data":null}]}]}],
           "observations":[{"name":"chA","data":[5.0,10.0]},{"name":"chB","data":[7.0,3.0]}],
           "measurements":[{"name":"m","config":{"poi":"k","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["`k`", "normfactor", "shapefactor"],
    );
}

/// A `normfactor` and a `normsys` sharing the name `k`.
///
/// pyhf: `pyhf.exceptions.InvalidNameReuse` — "Multiple values for
/// 'paramset_type' (['constrained_by_normal', 'unconstrained']) were found for
/// k."
///
/// Both declare one real component, so the domains agree; the auxiliary
/// measurement is what differs. The importer emitted one `k = elementof(reals)`
/// and one `Normal(k, 1)` constraint, silently making the free normfactor
/// constrained.
#[test]
fn shared_name_with_incompatible_constraints_errs() {
    assert_err_pyhf(
        "normfactor_vs_normsys",
        r#"{"channels":[
             {"name":"chA","samples":[{"name":"sig","data":[5.0,10.0],
                "modifiers":[{"name":"k","type":"normfactor","data":null}]}]},
             {"name":"chB","samples":[{"name":"sig","data":[7.0,3.0],
                "modifiers":[{"name":"k","type":"normsys","data":{"hi":1.1,"lo":0.9}}]}]}],
           "observations":[{"name":"chA","data":[5.0,10.0]},{"name":"chB","data":[7.0,3.0]}],
           "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["`k`", "normfactor", "normsys"],
    );
}

/// One `shapesys` name in two channels.
///
/// pyhf: `pyhf.exceptions.InvalidModel` — "Trying to add paramset shapesys/k on
/// s sample in chB channel but other paramsets exist with the same name."
///
/// A shapesys paramset is per-channel, so the two occurrences are different
/// parameters. The importer emitted one declaration and one constraint, from
/// whichever channel came first, silently dropping the other channel's sigma.
#[test]
fn shapesys_name_shared_across_channels_errs() {
    assert_err_pyhf(
        "shapesys_cross_channel",
        r#"{"channels":[
             {"name":"chA","samples":[{"name":"s","data":[10.0,20.0],
                "modifiers":[{"name":"k","type":"shapesys","data":[5.0,6.0]}]}]},
             {"name":"chB","samples":[{"name":"s","data":[10.0,20.0],
                "modifiers":[{"name":"k","type":"shapesys","data":[1.0,2.0]}]}]}],
           "observations":[{"name":"chA","data":[10.0,20.0]},{"name":"chB","data":[10.0,20.0]}],
           "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["shapesys", "`k`", "chA", "chB"],
    );
}

/// A measurement whose `poi` names a parameter no modifier declares.
///
/// pyhf: `pyhf.exceptions.InvalidModel` — "The parameter of interest 'missing'
/// cannot be fit as it is not declared in the model specification."
///
/// The importer emitted `m = record(poi = missing)`, a dangling reference. The
/// CLI printed `error[unresolved-name]` about the file it had just written and
/// still exited 0.
#[test]
fn poi_naming_no_parameter_errs() {
    assert_err_pyhf(
        "undeclared_poi",
        r#"{"channels":[{"name":"ch","samples":[
             {"name":"bkg","data":[50.0,60.0],
              "modifiers":[{"name":"mu","type":"normfactor","data":null}]}]}],
           "observations":[{"name":"ch","data":[55.0,70.0]}],
           "measurements":[{"name":"m","config":{"poi":"missing","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["`missing`", "`m`"],
    );
}

/// A `shapesys` `data` array shorter than the sample.
///
/// pyhf: `pyhf.exceptions.InvalidModifier` — "The 'bkg' sample shapesys
/// modifier 'uncorr' has data shape inconsistent with the sample. bkg has 'data'
/// of length 2 but uncorr has 'data' of length 1."
///
/// The importer emitted a two-component `uncorr = elementof(cartpow(posreals,
/// 2))` against a one-element `uncorr_sigma = [5.0]`. `infer` left the mismatch
/// `%deferred` and `lint` was clean.
#[test]
fn shapesys_data_shorter_than_sample_errs() {
    assert_err_pyhf(
        "shapesys_short",
        r#"{"channels":[{"name":"ch","samples":[
             {"name":"bkg","data":[50.0,60.0],
              "modifiers":[{"name":"uncorr","type":"shapesys","data":[5.0]}]}]}],
           "observations":[{"name":"ch","data":[55.0,70.0]}],
           "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["shapesys", "`uncorr`", "1", "2"],
    );
}

/// A `shapesys` `data` array LONGER than the sample, the other side of the same
/// check.
///
/// pyhf: `pyhf.exceptions.InvalidModifier` — "The 'bkg' sample shapesys
/// modifier 'uncorr' has data shape inconsistent with the sample.\nbkg has
/// 'data' of length 2 but uncorr has 'data' of length 3."
#[test]
fn shapesys_data_longer_than_sample_errs() {
    assert_err_pyhf(
        "shapesys_long",
        r#"{"channels":[{"name":"ch","samples":[
             {"name":"bkg","data":[50.0,60.0],
              "modifiers":[{"name":"uncorr","type":"shapesys","data":[5.0,6.0,7.0]}]}]}],
           "observations":[{"name":"ch","data":[55.0,70.0]}],
           "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["shapesys", "`uncorr`", "3", "2"],
    );
}

// ---------------------------------------------------------------------------
// The compatible cases pyhf ACCEPTS must keep converting, and a shared name
// must yield exactly ONE declaration — pyhf gives it one paramset.
// ---------------------------------------------------------------------------

/// pyhf accepts one `normfactor` name across two channels and reports a single
/// `mu` in `par_order`. The importer used to emit `mu = elementof(reals)` twice,
/// because the declared-parameter set was rebuilt per channel.
#[test]
fn normfactor_shared_across_channels_declares_once() {
    let json = r#"{"channels":[
         {"name":"chA","samples":[{"name":"sig","data":[5.0,10.0],
            "modifiers":[{"name":"mu","type":"normfactor","data":null}]}]},
         {"name":"chB","samples":[{"name":"sig","data":[7.0,3.0],
            "modifiers":[{"name":"mu","type":"normfactor","data":null}]}]}],
       "observations":[{"name":"chA","data":[5.0,10.0]},{"name":"chB","data":[7.0,3.0]}],
       "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
       "version":"1.0.0"}"#;
    let m = flatppl_hs3::read_pyhf(json).expect("pyhf accepts this workspace");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert_eq!(
        text.matches("mu = elementof(reals)").count(),
        1,
        "a shared normfactor is one nuisance parameter, got:\n{text}"
    );
}

/// pyhf accepts a `normsys` and a `histosys` sharing one name: both are
/// `constrained_by_normal` with one component.
#[test]
fn normsys_and_histosys_may_share_a_name() {
    let json = r#"{"channels":[
         {"name":"chA","samples":[{"name":"s","data":[10.0,20.0],
            "modifiers":[{"name":"a","type":"normsys","data":{"hi":1.1,"lo":0.9}}]}]},
         {"name":"chB","samples":[{"name":"s","data":[10.0,20.0],
            "modifiers":[{"name":"a","type":"histosys",
              "data":{"hi_data":[11.0,21.0],"lo_data":[9.0,19.0]}}]}]}],
       "observations":[{"name":"chA","data":[10.0,20.0]},{"name":"chB","data":[10.0,20.0]}],
       "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
       "version":"1.0.0"}"#;
    let m = flatppl_hs3::read_pyhf(json).expect("pyhf accepts this workspace");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert_eq!(
        text.matches("a = elementof(reals)").count(),
        1,
        "the shared alpha is one nuisance parameter, got:\n{text}"
    );
    assert_eq!(
        text.matches("a_constraint_likelihood =").count(),
        1,
        "one shared parameter carries one auxiliary measurement, got:\n{text}"
    );
}

/// One `staterror` name across two channels.
///
/// pyhf accepts it and gives the name ONE paramset spanning both channels'
/// bins: for the workspace below, 4 components with sigmas
/// [0.5, 0.3, 0.0333.., 0.05] and auxdata [1, 1, 1, 1], each channel masking
/// its own two. A single `cartpow(posreals, 2)` parameter multiplied into both
/// channels instead correlates gammas pyhf keeps independent, and its
/// constraint covers only the first channel's bins. That is what the importer
/// used to emit, and it is silently wrong, so refuse until the spanning form is
/// implemented. pyhf's own workspaces name these per channel
/// (`staterror_channel1`), which converts.
#[test]
fn staterror_shared_across_channels_errs() {
    assert_err_pyhf(
        "staterror_two_channels",
        r#"{"channels":[
             {"name":"chA","samples":[{"name":"s","data":[10.0,20.0],
                "modifiers":[{"name":"mcstat","type":"staterror","data":[5.0,6.0]}]}]},
             {"name":"chB","samples":[{"name":"s","data":[30.0,40.0],
                "modifiers":[{"name":"mcstat","type":"staterror","data":[1.0,2.0]}]}]}],
           "observations":[{"name":"chA","data":[10.0,20.0]},{"name":"chB","data":[30.0,40.0]}],
           "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
           "version":"1.0.0"}"#,
        &[
            "`mcstat`",
            "spanning every channel's bins (4 here)",
            "mcstat_chB",
        ],
    );
}

/// A `shapefactor` name in two channels with different bin counts.
///
/// pyhf builds a model, then reads past the end of its own 2-component paramset
/// and takes the third channel-B component from whatever parameter follows. It
/// returns a number, but not the one the document describes, so refuse. An
/// equal-bin sharing is genuinely one parameter and converts.
#[test]
fn per_bin_name_shared_with_different_bin_counts_errs() {
    assert_err_pyhf(
        "shapefactor_unequal_bins",
        r#"{"channels":[
             {"name":"ca","samples":[{"name":"b","data":[50.0,60.0],"modifiers":[
                {"name":"mu","type":"normfactor","data":null},
                {"name":"k","type":"shapefactor","data":null}]}]},
             {"name":"cb","samples":[{"name":"b","data":[30.0,20.0,10.0],
                "modifiers":[{"name":"k","type":"shapefactor","data":null}]}]}],
           "observations":[{"name":"ca","data":[52.0,58.0]},
                           {"name":"cb","data":[31.0,19.0,11.0]}],
           "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["`k`", "has 2 bins in channel `ca` and 3 in channel `cb`"],
    );
}

/// An empty `poi` string means "no POI declared" and pyhf accepts it, so the
/// undeclared-POI check must not fire on it.
#[test]
fn empty_poi_is_accepted() {
    let json = r#"{"channels":[{"name":"ch","samples":[
         {"name":"bkg","data":[50.0,60.0],
          "modifiers":[{"name":"mu","type":"normfactor","data":null}]}]}],
       "observations":[{"name":"ch","data":[55.0,70.0]}],
       "measurements":[{"name":"m","config":{"poi":"","parameters":[]}}],
       "version":"1.0.0"}"#;
    let m = flatppl_hs3::read_pyhf(json).expect("an empty poi is not a dangling reference");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert!(
        !text.contains("record(poi"),
        "no POI record when `poi` is empty, got:\n{text}"
    );
}

/// Two channels named `c`.
///
/// pyhf: `pyhf.exceptions.InvalidModel` — "No parameters specified for the
/// Model." (pyhf keys channels by name, so the second collapses onto the first.)
///
/// The importer used to give each channel its own bindings while both looked up
/// the workspace's single `c` observation, so the observed counts entered the
/// likelihood twice.
#[test]
fn duplicate_channel_name_errs() {
    assert_err_pyhf(
        "duplicate_channel",
        r#"{"channels":[
             {"name":"c","samples":[{"name":"b","data":[50.0],
                "modifiers":[{"name":"mu","type":"normfactor","data":null}]}]},
             {"name":"c","samples":[{"name":"b","data":[30.0],"modifiers":[]}]}],
           "observations":[{"name":"c","data":[50.0]}],
           "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["channel name `c` appears twice"],
    );
}

/// Two samples named `b` in one channel.
///
/// pyhf: `pyhf.exceptions.InvalidModel` — "No parameters specified for the
/// Model." pyhf keys a channel's samples by name, so the repeat collides.
#[test]
fn duplicate_sample_name_errs() {
    assert_err_pyhf(
        "duplicate_sample",
        r#"{"channels":[
             {"name":"c","samples":[
                {"name":"b","data":[50.0],
                 "modifiers":[{"name":"mu","type":"normfactor","data":null}]},
                {"name":"b","data":[30.0],"modifiers":[]}]}],
           "observations":[{"name":"c","data":[80.0]}],
           "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["two samples named `b`"],
    );
}

/// A channel with an empty `samples` array, and a workspace with no channels.
///
/// pyhf: `pyhf.exceptions.InvalidSpecification` — "[] should be non-empty",
/// against `channels[0].samples` and `channels`.
#[test]
fn empty_channel_and_empty_channels_err() {
    assert_err_pyhf(
        "channel_without_samples",
        r#"{"channels":[{"name":"c","samples":[]}],
           "observations":[{"name":"c","data":[50.0]}],
           "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["channel `c` has no samples"],
    );
    assert_err_pyhf(
        "workspace_without_channels",
        r#"{"channels":[],"observations":[],
           "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["workspace has no channels"],
    );
}

/// A histosys whose `lo_data` is shorter than the sample.
///
/// pyhf: `pyhf.exceptions.InvalidModifier` — "The 'b' sample histosys modifier
/// 'h' has data shape inconsistent with the sample." The message must name the
/// parameter, which a pyhf modifier carries in `name`, not `parameter`.
#[test]
fn histosys_length_mismatch_names_the_parameter() {
    assert_err_pyhf(
        "histosys_short_lo",
        r#"{"channels":[
             {"name":"c","samples":[{"name":"b","data":[50.0,60.0],"modifiers":[
                {"name":"mu","type":"normfactor","data":null},
                {"name":"h","type":"histosys",
                 "data":{"lo_data":[45.0],"hi_data":[55.0]}}]}]}],
           "observations":[{"name":"c","data":[52.0,58.0]}],
           "measurements":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
           "version":"1.0.0"}"#,
        &["histosys `h`", "has 1 bins but the sample nominal has 2"],
    );
}
