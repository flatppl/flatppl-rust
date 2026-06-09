//! A minimal S-expression reader.
//!
//! FlatPIR's canonical text syntax is standard S-expressions (spec §11), so this
//! layer is deliberately generic: it knows nothing about `%module` / `%bind` /
//! types — only atoms, strings, lists, and `;` line comments. The [`reader`]
//! interprets the resulting [`Sexpr`] tree into a `flatppl-core` module.
//!
//! [`reader`]: crate::reader

use crate::error::{Error, Result};

/// A parsed S-expression: a bare atom, a string literal, or a list.
#[derive(Clone, Debug, PartialEq)]
pub enum Sexpr {
    /// A bare atom — symbol, `%keyword`, number, or boolean — as its raw lexeme.
    Atom(String),
    /// A string literal, already unescaped (the surface had surrounding quotes).
    Str(String),
    /// A parenthesised list.
    List(Vec<Sexpr>),
}

impl Sexpr {
    /// The atom text, if this is an [`Sexpr::Atom`].
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            Sexpr::Atom(s) => Some(s),
            _ => None,
        }
    }

    /// The list elements, if this is an [`Sexpr::List`].
    pub fn as_list(&self) -> Option<&[Sexpr]> {
        match self {
            Sexpr::List(items) => Some(items),
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
}

impl Parser {
    fn new(input: &str) -> Self {
        Parser {
            src: input.chars().collect(),
            pos: 0,
            line: 1,
        }
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

    fn parse_form(&mut self) -> Result<Sexpr> {
        match self.peek() {
            Some('(') => self.parse_list(),
            Some(')') => Err(Error::at(self.line, "unexpected `)`")),
            Some('"') => self.parse_string(),
            Some(_) => self.parse_atom(),
            None => Err(Error::at(self.line, "unexpected end of input")),
        }
    }

    fn parse_list(&mut self) -> Result<Sexpr> {
        let open_line = self.line;
        self.bump(); // consume '('
        let mut items = Vec::new();
        loop {
            self.skip_trivia();
            match self.peek() {
                Some(')') => {
                    self.bump();
                    return Ok(Sexpr::List(items));
                }
                None => return Err(Error::at(open_line, "unclosed `(`")),
                _ => items.push(self.parse_form()?),
            }
        }
    }

    fn parse_string(&mut self) -> Result<Sexpr> {
        let open_line = self.line;
        self.bump(); // consume opening '"'
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('"') => return Ok(Sexpr::Str(s)),
                Some('\\') => match self.bump() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('0') => s.push('\0'),
                    Some(other) => {
                        return Err(Error::at(
                            self.line,
                            format!("invalid string escape `\\{other}`"),
                        ));
                    }
                    None => return Err(Error::at(open_line, "unterminated string")),
                },
                Some(c) => s.push(c),
                None => return Err(Error::at(open_line, "unterminated string")),
            }
        }
    }

    fn parse_atom(&mut self) -> Result<Sexpr> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_whitespace() || matches!(c, '(' | ')' | '"' | ';') {
                break;
            }
            s.push(c);
            self.bump();
        }
        debug_assert!(!s.is_empty(), "parse_atom called at a delimiter");
        Ok(Sexpr::Atom(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_atoms_strings_lists() {
        let forms = parse_top(r#"(a "b c" (nested 1 2))"#).unwrap();
        assert_eq!(forms.len(), 1);
        let Sexpr::List(items) = &forms[0] else {
            panic!("expected list");
        };
        assert_eq!(items[0], Sexpr::Atom("a".into()));
        assert_eq!(items[1], Sexpr::Str("b c".into()));
        assert!(matches!(&items[2], Sexpr::List(_)));
    }

    #[test]
    fn skips_comments() {
        let forms = parse_top("; leading comment\n(a b) ; trailing\n").unwrap();
        assert_eq!(forms.len(), 1);
    }

    #[test]
    fn string_escapes() {
        let forms = parse_top(r#""line1\nline2 \"q\"""#).unwrap();
        assert_eq!(forms[0], Sexpr::Str("line1\nline2 \"q\"".into()));
    }

    #[test]
    fn unclosed_paren_errors() {
        assert!(parse_top("(a b").is_err());
    }
}
