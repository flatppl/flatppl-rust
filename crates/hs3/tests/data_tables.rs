//! Embedded `data` → FlatPPL `table` + `<name>_domain` cartprod (spec §03).
//!
//! Each HS3 dataset is lowered to a top-level `<name> = table(...)` binding (one
//! column per observable axis) and, when it declares axes, a companion
//! `<name>_domain = cartprod(axis = interval(min, max), ...)`. A single-axis
//! dataset feeding a likelihood is observed against its column vector under an
//! `iid` plate; a multi-axis dataset against the table itself.
use flatppl_syntax::{Syntax, parse, print_with};

fn convert(json: &str) -> String {
    let m = flatppl_hs3::read(json).expect("fixture must parse and convert");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        parse(&text).is_ok(),
        "emitted FlatPPL must re-parse cleanly:\n{text}"
    );
    text
}

/// Single-axis unbinned data: `table(axis = [...])` + `<name>_domain`, observed
/// against the column vector under an iid plate.
#[test]
fn single_axis_unbinned_embeds_table_domain_and_observes_column() {
    let text = convert(
        r#"{
            "distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"m"}],
            "data":[{"name":"obs","type":"unbinned",
                     "axes":[{"name":"m","min":-3.0,"max":3.0}],
                     "entries":[[0.1],[0.2],[0.3]]}],
            "likelihoods":[{"name":"L","distributions":["g"],"data":["obs"]}],
            "parameter_points":[{"name":"n","entries":[
                {"name":"mu","value":0.0},{"name":"s","value":1.0}]}]
        }"#,
    );
    assert!(
        text.contains("obs = table(m = [0.1, 0.2, 0.3])"),
        "single-column table mismatch, got:\n{text}"
    );
    assert!(
        text.contains("obs_domain = cartprod(m = interval(-3.0, 3.0))"),
        "data domain mismatch, got:\n{text}"
    );
    assert!(
        text.contains("L = likelihoodof(iid(g, 3), get(obs, \"m\"))"),
        "column observation wiring mismatch, got:\n{text}"
    );
}

/// Multi-axis unbinned data: one table column per axis, a multi-field domain,
/// and (a multivariate event sample) the table observed directly under iid.
#[test]
fn multi_axis_unbinned_embeds_multicolumn_table_and_observes_table() {
    let text = convert(
        r#"{
            "distributions":[
                {"name":"prod","type":"product_dist","factors":["gx","gy"]},
                {"name":"gx","type":"gaussian_dist","mean":"mux","sigma":"sx","x":"x"},
                {"name":"gy","type":"gaussian_dist","mean":"muy","sigma":"sy","x":"y"}],
            "data":[{"name":"d","type":"unbinned",
                     "axes":[{"name":"x","min":-5.0,"max":5.0},{"name":"y","min":-2.0,"max":2.0}],
                     "entries":[[0.1,1.0],[0.2,1.1]]}],
            "likelihoods":[{"name":"L","distributions":["prod"],"data":["d"]}],
            "parameter_points":[{"name":"n","entries":[
                {"name":"mux","value":0.0},{"name":"sx","value":1.0},
                {"name":"muy","value":0.0},{"name":"sy","value":1.0}]}]
        }"#,
    );
    assert!(
        text.contains("d = table(x = [0.1, 0.2], y = [1.0, 1.1])"),
        "multi-column table mismatch, got:\n{text}"
    );
    assert!(
        text.contains("d_domain = cartprod(x = interval(-5.0, 5.0), y = interval(-2.0, 2.0))"),
        "multi-axis data domain mismatch, got:\n{text}"
    );
    // Multivariate event sample: the table itself is the observation (each row
    // is one (x, y) event), plated iid over its 2 rows.
    assert!(
        text.contains("L = likelihoodof(iid(prod, 2), d)"),
        "table observation wiring mismatch, got:\n{text}"
    );
}

/// Unbinned data with no declared axes: columns are synthesized positionally
/// (`c1`, `c2`, …) and no domain is emitted. (Previously this multi-coordinate
/// case was rejected outright.)
#[test]
fn axisless_multidim_unbinned_synthesizes_columns() {
    let text = convert(
        r#"{
            "distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"gx"}],
            "data":[{"name":"obs","type":"unbinned","entries":[[1.0,2.0]]}],
            "parameter_points":[{"name":"n","entries":[
                {"name":"mu","value":0.0},{"name":"s","value":1.0}]}]
        }"#,
    );
    assert!(
        text.contains("obs = table(c1 = [1.0], c2 = [2.0])"),
        "synthesized-column table mismatch, got:\n{text}"
    );
    assert!(
        !text.contains("obs_domain"),
        "no axes ⇒ no domain binding, got:\n{text}"
    );
}

// A `point` datum is a single scalar observation: bound as a bare literal and
// observed without an iid plate (HS3 §2.3 "point" data).
#[test]
fn point_datum_scalar_observation() {
    let json = r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
        "data":[{"name":"d","type":"point","value":1.27}],
        "likelihoods":[{"name":"L","distributions":["g"],"data":["d"]}],
        "parameter_points":[]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("bare point datum converts");
    let out = print_with(&m, Syntax::Minimal);
    assert!(
        out.contains("d = 1.27"),
        "scalar binding expected, got:\n{out}"
    );
    assert!(
        out.contains("likelihoodof(g, d)"),
        "un-plated scalar observation expected, got:\n{out}"
    );
}

/// Binned data is embedded as a single `counts` column; its axes still yield a
/// companion domain.
#[test]
fn binned_data_embeds_counts_table_and_domain() {
    let text = convert(
        r#"{
            "data":[{"name":"bd","type":"binned",
                     "axes":[{"name":"obs","min":0.0,"max":10.0}],
                     "contents":[5.0,7.0,3.0]}],
            "parameter_points":[]
        }"#,
    );
    assert!(
        text.contains("bd = table(counts = [5.0, 7.0, 3.0])"),
        "binned counts table mismatch, got:\n{text}"
    );
    assert!(
        text.contains("bd_domain = cartprod(obs = interval(0.0, 10.0))"),
        "binned data domain mismatch, got:\n{text}"
    );
}
