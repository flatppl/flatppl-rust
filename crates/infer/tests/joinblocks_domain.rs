use flatppl_infer::{Severity, infer};

#[test]
fn joinblocks_requires_nesting_and_preserves_the_inverse_shape() {
    let mut invalid =
        flatppl_syntax::parse("m = rowstack([[1, 2], [3, 4]])\nx = joinblocks(m)\n").unwrap();
    assert!(
        infer(&mut invalid)
            .iter()
            .any(|d| d.severity == Severity::Error)
    );
    let mut ranks =
        flatppl_syntax::parse("m = rowstack([[1, 2], [3, 4]])\nx = joinblocks([m, m])\n").unwrap();
    assert!(
        infer(&mut ranks)
            .iter()
            .any(|d| d.severity == Severity::Error)
    );

    let mut valid =
        flatppl_syntax::parse("x = joinblocks(splitblocks([1, 2, 3, 4], 2))\n").unwrap();
    let diagnostics = infer(&mut valid);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    let output = flatppl_flatpir::write(&valid);
    assert!(
        output.contains("(%array 1 (%dynamic) (%scalar integer))"),
        "{output}"
    );

    let mut fixed = flatppl_syntax::parse("x = joinblocks([[1, 2], [3, 4], [5, 6]])\n").unwrap();
    let diagnostics = infer(&mut fixed);
    assert!(diagnostics.iter().all(|d| d.severity != Severity::Error));
    let output = flatppl_flatpir::write(&fixed);
    assert!(
        output.contains("(%array 1 (6) (%scalar integer))"),
        "{output}"
    );
}
