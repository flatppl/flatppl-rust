//! A minimal S-expression reader.
//!
//! FlatPIR's canonical text syntax is standard S-expressions (spec §11), so this
//! layer is deliberately generic: it knows nothing about `%module` / `%bind` /
//! types — only atoms, strings, lists, and `;` line comments. The [`reader`]
//! interprets the resulting [`Sexpr`] tree into a `flatppl-core` module.
//!
//! Every parsed form carries a [`Span`] (byte range + start line) so the reader
//! can localize its structural/semantic errors back to the source — the same
//! caret treatment the lexer below already gives unbalanced parens and bad
//! string escapes.
//!
//! [`reader`]: crate::reader

use crate::error::{Error, Result};

/// A source location for a parsed form: byte range `[start, end)` into the
/// source plus the 1-based line of the form's first character. Byte offsets
/// drive the source-annotated renderer; the line is the fallback used when a
/// renderer has no source to compute from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
    pub line: usize,
}

/// A parsed S-expression: its [`SexprKind`] plus the source [`Span`] it covers.
#[derive(Clone, Debug, PartialEq)]
pub struct Sexpr {
    pub kind: SexprKind,
    pub span: Span,
}

/// The shape of a parsed S-expression: a bare atom, a string literal, or a list.
#[derive(Clone, Debug, PartialEq)]
pub enum SexprKind {
    /// A bare atom — symbol, `%keyword`, number, or boolean — as its raw lexeme.
    Atom(String),
    /// A string literal, already unescaped (the surface had surrounding quotes).
    Str(String),
    /// A parenthesised list.
    List(Vec<Sexpr>),
}

impl Sexpr {
    /// The atom text, if this is a [`SexprKind::Atom`].
    pub fn as_atom(&self) -> Option<&str> {
        match &self.kind {
            SexprKind::Atom(s) => Some(s),
            _ => None,
        }
    }

    /// The list elements, if this is a [`SexprKind::List`].
    pub fn as_list(&self) -> Option<&[Sexpr]> {
        match &self.kind {
            SexprKind::List(items) => Some(items),
            _ => None,
        }
    }
}

/// Parse all top-level forms in `input`. FlatPIR files hold exactly one
/// `(%module …)`, but parsing is general (comments and blank space anywhere).
pub fn parse_top(input: &str) -> Result<Vec<Sexpr>> {
    let mut p = Parser::new(input);
    let mut forms = Vec::new();
    p.skip_trivia();
    while !p.at_end() {
        forms.push(p.parse_form()?);
        p.skip_trivia();
    }
    Ok(forms)
}

struct Parser {
    src: Vec<char>,
    pos: usize,
    line: usize,
    /// Byte offset of the cursor (UTF-8 aware), for error spans.
    byte: u32,
}

impl Parser {
    fn new(input: &str) -> Self {
        Parser {
            src: input.chars().collect(),
            pos: 0,
            line: 1,
            byte: 0,
        }
    }

    /// An error spanning the single character at the cursor.
    fn err_here(&self, message: impl Into<String>) -> Error {
        Error::at_span(self.line, (self.byte, self.byte + 1), message)
    }

    fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied();
        if let Some(ch) = c {
            self.pos += 1;
            self.byte += ch.len_utf8() as u32;
            if ch == '\n' {
                self.line += 1;
            }
        }
        c
    }

    /// Skip whitespace and `;`-to-end-of-line comments.
    fn skip_trivia(&mut self) {
        while let Some(c) = self.peek() {
            if c == ';' {
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.bump();
                }
            } else if c.is_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }

    /// Parse one form, wrapping its [`SexprKind`] with the [`Span`] it covers.
    /// The cursor is assumed to sit at the form's first character (callers skip
    /// trivia first), so `start` is the form's opening byte and `end` is the
    /// byte just past its last character.
    fn parse_form(&mut self) -> Result<Sexpr> {
        let start = self.byte;
        let line = self.line;
        let kind = match self.peek() {
            Some('(') => self.parse_list()?,
            Some(')') => return Err(self.err_here("unexpected `)`")),
            Some('"') => self.parse_string()?,
            Some(_) => self.parse_atom()?,
            None => return Err(self.err_here("unexpected end of input")),
        };
        let end = self.byte;
        Ok(Sexpr {
            kind,
            span: Span { start, end, line },
        })
    }

    fn parse_list(&mut self) -> Result<SexprKind> {
        let open_line = self.line;
        let open_byte = self.byte;
        self.bump(); // consume '('
        let mut items = Vec::new();
        loop {
            self.skip_trivia();
            match self.peek() {
                Some(')') => {
                    self.bump();
                    return Ok(SexprKind::List(items));
                }
                None => {
                    return Err(Error::at_span(
                        open_line,
                        (open_byte, open_byte + 1),
                        "unclosed `(`",
                    ));
                }
                _ => items.push(self.parse_form()?),
            }
        }
    }

    fn parse_string(&mut self) -> Result<SexprKind> {
        let open_line = self.line;
        let open_byte = self.byte;
        self.bump(); // consume opening '"'
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('"') => return Ok(SexprKind::Str(s)),
                Some('\\') => match self.bump() {
                    // (escape span: backslash is 1 byte, the escaped char follows)
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('0') => s.push('\0'),
                    Some(other) => {
                        let end = self.byte;
                        return Err(Error::at_span(
                            self.line,
                            (end - 1 - other.len_utf8() as u32, end),
                            format!("invalid string escape `\\{other}`"),
                        ));
                    }
                    None => {
                        return Err(Error::at_span(
                            open_line,
                            (open_byte, open_byte + 1),
                            "unterminated string",
                        ));
                    }
                },
                Some(c) => s.push(c),
                None => {
                    return Err(Error::at_span(
                        open_line,
                        (open_byte, open_byte + 1),
                        "unterminated string",
                    ));
                }
            }
        }
    }

    fn parse_atom(&mut self) -> Result<SexprKind> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_whitespace() || matches!(c, '(' | ')' | '"' | ';') {
                break;
            }
            s.push(c);
            self.bump();
        }
        debug_assert!(!s.is_empty(), "parse_atom called at a delimiter");
        Ok(SexprKind::Atom(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_atoms_strings_lists() {
        let forms = parse_top(r#"(a "b c" (nested 1 2))"#).unwrap();
        assert_eq!(forms.len(), 1);
        let items = forms[0].as_list().expect("expected list");
        assert_eq!(items[0].as_atom(), Some("a"));
        assert_eq!(items[1].kind, SexprKind::Str("b c".into()));
        assert!(items[2].as_list().is_some());
    }

    #[test]
    fn skips_comments() {
        let forms = parse_top("; leading comment\n(a b) ; trailing\n").unwrap();
        assert_eq!(forms.len(), 1);
    }

    #[test]
    fn string_escapes() {
        let forms = parse_top(r#""line1\nline2 \"q\"""#).unwrap();
        assert_eq!(forms[0].kind, SexprKind::Str("line1\nline2 \"q\"".into()));
    }

    #[test]
    fn unclosed_paren_errors() {
        assert!(parse_top("(a b").is_err());
    }

    #[test]
    fn spans_cover_their_forms() {
        let src = "(a\n  (b c))";
        let forms = parse_top(src).unwrap();
        let outer = &forms[0];
        let slice = |sp: Span| &src[sp.start as usize..sp.end as usize];
        assert_eq!(slice(outer.span), "(a\n  (b c))");
        assert_eq!(outer.span.line, 1);

        let inner = &outer.as_list().unwrap()[1];
        assert_eq!(slice(inner.span), "(b c)");
        assert_eq!(inner.span.line, 2, "nested form remembers its own line");
    }
}
