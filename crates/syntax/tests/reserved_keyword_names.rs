use flatppl_syntax::parse;

const RESERVED: [&str; 5] = ["in", "true", "false", "all", "only"];

/// §05: `KeywordArg ::= Name "=" Expression`, and the reserved words "are
/// recognized before `Name`", so none of them names a keyword argument.
#[test]
fn a_reserved_word_is_not_a_keyword_argument_name() {
    for word in RESERVED {
        for source in [
            format!("g(x) = x\nz = g({word} = 2.0)"),
            format!("y ~ Normal(mu = 0.0, {word} = 1.0)"),
            format!("r = record({word} = 1.0)"),
            format!("r = ({word} = 1.0)"),
            format!("r = (a = 2.0, {word} = 1.0)"),
            format!("f = functionof(_x_ + 1.0, {word} = _x_)"),
            format!("a = elementof(reals)\nk = kernelof(draw(Normal(a, 1.0)), {word} = a)"),
        ] {
            let error = parse(&source).expect_err(&source);
            assert!(
                error.to_string().contains(&format!(
                    "`{word}` is a reserved word and cannot be a keyword argument name"
                )),
                "{source}: {error}"
            );
            assert!(error.span.is_some(), "{source}: error is not located");
        }
    }
}

/// §04 bars `self` and `base` only as module bindings; as keyword names they
/// stay legal, as does a name that merely starts with a reserved word.
#[test]
fn self_base_and_reserved_prefixes_stay_legal_keyword_names() {
    for source in [
        "f = functionof(_x_ + 1.0, self = _x_)\nz = f(self = 2.0)",
        "r = (self = 1.0, base = 2.0)",
        "r = record(inside = 1.0, allx = 2.0, only_one = 3.0)",
    ] {
        parse(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}
