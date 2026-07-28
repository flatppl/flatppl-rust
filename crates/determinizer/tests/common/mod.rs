//! Helpers shared by the determiniser golden tests.

/// The `(%bind <name> …)` form for `name`, delimited by its own matching paren.
/// Scoped to the binding rather than taken as the rest of the file, so an assertion
/// about one binding cannot be satisfied — or defeated — by text emitted elsewhere.
pub fn pir_binding(pir: &str, name: &str) -> String {
    let open = format!("(%bind {name} ");
    let start = pir
        .find(&open)
        .unwrap_or_else(|| panic!("no `{name}` binding in:\n{pir}"));
    let mut depth = 0usize;
    for (i, ch) in pir[start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return pir[start..start + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated `{name}` binding in:\n{pir}")
}
