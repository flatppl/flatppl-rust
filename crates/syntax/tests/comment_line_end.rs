//! Spec §05: a `#` line comment and a `%` doc-comment line end only at a line break. A
//! `;` inside one is comment text, not a statement separator.

fn bound(src: &str) -> Vec<String> {
    let module = flatppl_syntax::parse(src).unwrap();
    module
        .bindings()
        .map(|(_, b)| module.resolve(b.name).to_string())
        .collect()
}

#[test]
fn a_semicolon_in_a_line_comment_does_not_start_a_statement() {
    assert_eq!(bound("a = 1.0  # one; b = 2.0\n"), ["a"]);
    assert!(bound("# one; two\n").is_empty());
    assert_eq!(bound("###\none; b = 2.0\n###\na = 1.0\n"), ["a"]);
}

#[test]
fn a_semicolon_in_a_doc_line_stays_in_the_doc() {
    let module = flatppl_syntax::parse("% doc; text\na = 1.0\n").unwrap();
    let names: Vec<_> = module
        .bindings()
        .map(|(_, b)| module.resolve(b.name).to_string())
        .collect();
    assert_eq!(names, ["a"]);
    let (_, a) = module.bindings().next().unwrap();
    let doc = format!("{:?}", a.doc);
    assert!(doc.contains("doc; text"), "{doc}");
}
