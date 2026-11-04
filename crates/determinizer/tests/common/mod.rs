//! Helpers shared by the determiniser golden tests.
//!
//! Every test binary that declares `mod common` compiles the whole module, so a helper
//! only some of them use is dead code there.
#![allow(dead_code)]

/// The `n`th (0-based) argument of the FIRST `(<head> …)` call in `expr`, delimited by
/// its own matching paren — so a `%meta`-wrapped argument comes back whole. Panics if
/// there is no such call or it has too few arguments.
///
/// Lets an assertion about a GATED density scope to the gate's arm
/// (`call_arg(&out, "ifelse", 1)`) rather than to the whole emission, where the gate's
/// own condition would satisfy it.
pub fn call_arg(expr: &str, head: &str, n: usize) -> String {
    let open = format!("({head} ");
    let start = expr
        .find(&open)
        .unwrap_or_else(|| panic!("no `{head}` call in:\n{expr}"))
        + open.len();
    let rest = &expr[start..];
    let mut at = 0usize;
    for i in 0..=n {
        while rest[at..].starts_with(' ') {
            at += 1;
        }
        let end = group_end(&rest[at..]);
        if i == n {
            return rest[at..at + end].to_string();
        }
        at += end;
    }
    panic!("`{head}` has no argument {n} in:\n{expr}")
}

/// The head symbol of `expr`, with any `%meta` type annotation stripped — the name of
/// the operation the expression actually IS.
pub fn pir_head(expr: &str) -> String {
    let inner = expr.trim();
    let Some(body) = inner.strip_prefix('(') else {
        return inner.to_string();
    };
    let head: String = body
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != ')')
        .collect();
    if head != "%meta" {
        return head;
    }
    // `(%meta <annotation> <expr>)` — the annotated expression is argument 1.
    pir_head(&call_arg(inner, "%meta", 1))
}

/// The length of the balanced group at the start of `s` — a parenthesised form, or a
/// bare token up to the next space or closing paren.
fn group_end(s: &str) -> usize {
    if !s.starts_with('(') {
        return s
            .find([' ', ')'])
            .unwrap_or_else(|| panic!("unterminated token in:\n{s}"));
    }
    let mut depth = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced group in:\n{s}")
}

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

/// Number the density placeholders by first use, so equal lowerings of
/// different sources compare equal.
pub fn renumber_points(text: &str) -> String {
    const STEM: &str = "density_point_";
    let mut seen: Vec<String> = Vec::new();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(STEM) {
        let (head, tail) = rest.split_at(at + STEM.len());
        out.push_str(head);
        let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
        let serial = &tail[..digits];
        let id = match seen.iter().position(|s| s == serial) {
            Some(id) => id,
            None => {
                seen.push(serial.to_owned());
                seen.len() - 1
            }
        };
        out.push_str(&id.to_string());
        rest = &tail[digits..];
    }
    out.push_str(rest);
    out
}

/// Strip the transparent `(%meta (<triple>) <inner>)` type annotations the
/// FlatPIR writer interleaves, leaving the bare S-expression — so a test can
/// assert an exact nested structure without the type noise between nodes.
pub fn strip_meta(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut drop_close: Vec<bool> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if s[i..].starts_with("(%meta ") {
            // Drop `(%meta ` + the balanced `(<triple>)` group + one space; the
            // matching close paren is dropped when we reach it (drop_close = true).
            i += "(%meta ".len();
            let mut depth = 0i32;
            loop {
                match b[i] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                i += 1;
                if depth == 0 {
                    break;
                }
            }
            if i < b.len() && b[i] == b' ' {
                i += 1;
            }
            drop_close.push(true);
        } else {
            match b[i] {
                b'(' => {
                    out.push('(');
                    drop_close.push(false);
                }
                b')' => {
                    if !drop_close.pop().unwrap_or(false) {
                        out.push(')');
                    }
                }
                c => out.push(c as char),
            }
            i += 1;
        }
    }
    out
}
