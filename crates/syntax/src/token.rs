//! Tokenizer for canonical FlatPPL surface syntax (spec §05).
//!
//! Produces a flat [`Token`] stream. A few spec rules are handled here rather
//! than in the parser:
//!
//! - **Newlines as separators only at depth 0.** Inside an unclosed `(`/`[`,
//!   a newline is whitespace (implicit line continuation, §05); at bracket
//!   depth 0 it becomes a [`TokenKind::Newline`] statement separator.
//! - **Maximal munch** for dotted operators (`.+`, `.==`, …) vs the field/axis
//!   dot, and for the trailing-dot real literal (`1.` is `1.0`).
//! - **Comments** (`#`, `###`) are discarded; **doc-comments** (`%`, `%%%`)
//!   become [`TokenKind::Doc`] tokens the parser attaches to bindings.

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    /// Byte offsets into the source `[start, end)`, for spans/diagnostics.
    pub start: u32,
    pub end: u32,
    pub line: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Literals and identifiers.
    Int(i64),
    Real(f64),
    Str(String),
    Name(String),

    // Grouping and separators.
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Newline,

    // Binding / structural punctuation.
    Assign,    // =
    Tilde,     // ~
    Walrus,    // :=
    Colon,     // :
    Arrow,     // ->
    Dot,       // .   (field access or axis prefix — parser decides by context)
    DotLParen, // .(  (dot-call broadcast)

    // Plain operators.
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Lt,
    Gt,
    EqEq,
    BangEq,
    Le,
    Ge,
    AmpAmp,
    PipePipe,
    Bang,

    // Dotted (broadcast) operators — the same set, dot-prefixed.
    DotPlus,
    DotMinus,
    DotStar,
    DotSlash,
    DotCaret,
    DotLt,
    DotGt,
    DotEqEq,
    DotBangEq,
    DotLe,
    DotGe,
    DotAmpAmp,
    DotPipePipe,
    DotBang,

    /// A doc-comment (`%`/`%%%`), with an optional markup tag and content lines.
    Doc(Doc),

    Eof,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Doc {
    pub tag: Option<String>,
    pub lines: Vec<String>,
    /// `true` for a trailing single-line `% …` (after a binding's RHS); leading
    /// otherwise. Block `%%%` forms are always leading.
    pub trailing: bool,
}

/// Tokenize `input` into a [`Token`] stream terminated by [`TokenKind::Eof`].
pub fn tokenize(input: &str) -> Result<Vec<Token>> {
    Lexer::new(input).run()
}

struct Lexer {
    chars: Vec<char>,
    /// Byte offset of `chars[i]`, for span reporting (`offsets[i]`).
    offsets: Vec<u32>,
    pos: usize,
    line: u32,
    bracket_depth: i32,
    /// Whether the last significant token could end an expression (so a `%` on
    /// the same line is a *trailing* doc-comment, and a newline is a separator).
    after_value: bool,
    tokens: Vec<Token>,
}

impl Lexer {
    fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let mut offsets = Vec::with_capacity(chars.len() + 1);
        let mut byte = 0u32;
        for c in &chars {
            offsets.push(byte);
            byte += c.len_utf8() as u32;
        }
        offsets.push(byte);
        Lexer {
            chars,
            offsets,
            pos: 0,
            line: 1,
            bracket_depth: 0,
            after_value: false,
            tokens: Vec::new(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if let Some(ch) = c {
            self.pos += 1;
            if ch == '\n' {
                self.line += 1;
            }
        }
        c
    }

    /// An error spanning byte offset `start` to the current position
    /// (widened to at least the character under the cursor, so an
    /// unconsumed offending character is still covered).
    fn err_span(&self, start: u32, line: u32, message: impl Into<String>) -> Error {
        let here = self.offsets[self.pos.min(self.chars.len())];
        let next = self.offsets[(self.pos + 1).min(self.chars.len())];
        let end = if here > start {
            here
        } else {
            next.max(start + 1)
        };
        Error::at_span(line, (start, end), message)
    }

    fn run(mut self) -> Result<Vec<Token>> {
        loop {
            self.skip_inline_ws_and_comments()?;
            let Some(c) = self.peek() else { break };

            if c == '\n' || c == '\r' {
                self.lex_newline();
                continue;
            }
            if c == '%' {
                self.lex_doc()?;
                continue;
            }
            self.lex_token(c)?;
        }
        let end = self.byte_pos();
        self.tokens.push(Token {
            kind: TokenKind::Eof,
            start: end,
            end,
            line: self.line,
        });
        Ok(self.tokens)
    }

    fn byte_pos(&self) -> u32 {
        self.offsets[self.pos.min(self.offsets.len() - 1)]
    }

    /// Skip spaces/tabs, `#` line comments, and `###` block comments. Does NOT
    /// consume newlines (those are separators, handled separately).
    fn skip_inline_ws_and_comments(&mut self) -> Result<()> {
        loop {
            match self.peek() {
                Some(' ' | '\t') => {
                    self.bump();
                }
                Some('#') => {
                    // `###` alone on a line opens a block comment; otherwise a
                    // line comment to end-of-line or `;` (spec §05).
                    if self.is_block_fence('#') {
                        self.skip_block_comment()?;
                    } else {
                        while let Some(c) = self.peek() {
                            if c == '\n' || c == ';' {
                                break;
                            }
                            self.bump();
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    /// Is the upcoming run exactly a `fence`×3 alone on its line (only
    /// horizontal whitespace before the newline)? Used for `###` / `%%%`.
    fn is_block_fence(&self, fence: char) -> bool {
        if self.peek() != Some(fence)
            || self.peek2() != Some(fence)
            || self.chars.get(self.pos + 2).copied() != Some(fence)
        {
            return false;
        }
        // After the three fence chars, only whitespace until the newline/EOF.
        let mut i = self.pos + 3;
        while let Some(&c) = self.chars.get(i) {
            if c == '\n' {
                return true;
            }
            if c != ' ' && c != '\t' && c != '\r' {
                return false;
            }
            i += 1;
        }
        true
    }

    fn skip_block_comment(&mut self) -> Result<()> {
        let open = self.line;
        let open_start = self.byte_pos();
        // Consume the opening `###` line.
        self.consume_to_newline();
        loop {
            if self.peek().is_none() {
                // Point at the opening fence, not the whole runaway block.
                return Err(Error::at_span(
                    open,
                    (open_start, open_start + 3),
                    "unterminated `###` block comment",
                ));
            }
            // A line whose trimmed content is exactly `###` closes the block.
            if self.at_line_start_fence('#') {
                self.consume_to_newline();
                return Ok(());
            }
            self.consume_to_newline();
        }
    }

    /// At the current position (start of a line, after leading ws), is the line
    /// exactly `fence`×3?
    fn at_line_start_fence(&mut self, fence: char) -> bool {
        let save = self.pos;
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.pos += 1;
        }
        let is_fence = self.is_block_fence(fence);
        self.pos = save;
        is_fence
    }

    fn consume_to_newline(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                self.bump();
                break;
            }
            self.bump();
        }
    }

    fn lex_newline(&mut self) {
        let start = self.byte_pos();
        let line = self.line;
        // Consume the line break (CRLF or LF or CR).
        if self.peek() == Some('\r') {
            self.bump();
        }
        if self.peek() == Some('\n') {
            self.bump();
        }
        // Inside brackets, newlines are whitespace (implicit continuation).
        if self.bracket_depth == 0 {
            self.push(TokenKind::Newline, start, line);
            self.after_value = false;
        }
    }

    fn push(&mut self, kind: TokenKind, start: u32, line: u32) {
        let end = self.byte_pos();
        // Track whether this token can end an expression (a value).
        self.after_value = matches!(
            kind,
            TokenKind::Int(_)
                | TokenKind::Real(_)
                | TokenKind::Str(_)
                | TokenKind::Name(_)
                | TokenKind::RParen
                | TokenKind::RBracket
        );
        self.tokens.push(Token {
            kind,
            start,
            end,
            line,
        });
    }

    fn lex_token(&mut self, c: char) -> Result<()> {
        let start = self.byte_pos();
        let line = self.line;
        match c {
            '0'..='9' => return self.lex_number(start, line),
            '"' => return self.lex_string(start, line),
            c if c == '_' || c.is_ascii_alphabetic() => return self.lex_name(start, line),
            '.' => return self.lex_dot(start, line),
            '(' => {
                self.bump();
                self.bracket_depth += 1;
                self.push(TokenKind::LParen, start, line);
            }
            ')' => {
                self.bump();
                self.bracket_depth -= 1;
                self.push(TokenKind::RParen, start, line);
            }
            '[' => {
                self.bump();
                self.bracket_depth += 1;
                self.push(TokenKind::LBracket, start, line);
            }
            ']' => {
                self.bump();
                self.bracket_depth -= 1;
                self.push(TokenKind::RBracket, start, line);
            }
            ',' => {
                self.bump();
                self.push(TokenKind::Comma, start, line);
            }
            ';' => {
                self.bump();
                self.push(TokenKind::Semi, start, line);
            }
            '+' => {
                self.bump();
                self.push(TokenKind::Plus, start, line);
            }
            '*' => {
                self.bump();
                self.push(TokenKind::Star, start, line);
            }
            '/' => {
                self.bump();
                self.push(TokenKind::Slash, start, line);
            }
            '^' => {
                self.bump();
                self.push(TokenKind::Caret, start, line);
            }
            '~' => {
                self.bump();
                self.push(TokenKind::Tilde, start, line);
            }
            '=' => {
                self.bump();
                let k = if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::EqEq
                } else {
                    TokenKind::Assign
                };
                self.push(k, start, line);
            }
            ':' => {
                self.bump();
                let k = if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::Walrus
                } else {
                    TokenKind::Colon
                };
                self.push(k, start, line);
            }
            '-' => {
                self.bump();
                let k = if self.peek() == Some('>') {
                    self.bump();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                };
                self.push(k, start, line);
            }
            '<' => {
                self.bump();
                let k = if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::Le
                } else {
                    TokenKind::Lt
                };
                self.push(k, start, line);
            }
            '>' => {
                self.bump();
                let k = if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                };
                self.push(k, start, line);
            }
            '!' => {
                self.bump();
                let k = if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::BangEq
                } else {
                    TokenKind::Bang
                };
                self.push(k, start, line);
            }
            '&' => {
                self.bump();
                if self.peek() == Some('&') {
                    self.bump();
                    self.push(TokenKind::AmpAmp, start, line);
                } else {
                    return Err(self.err_span(start, line, "unexpected `&` (did you mean `&&`?)"));
                }
            }
            '|' => {
                self.bump();
                if self.peek() == Some('|') {
                    self.bump();
                    self.push(TokenKind::PipePipe, start, line);
                } else {
                    return Err(self.err_span(start, line, "unexpected `|` (did you mean `||`?)"));
                }
            }
            other => {
                return Err(self.err_span(start, line, format!("unexpected character `{other}`")));
            }
        }
        Ok(())
    }

    /// `.` is overloaded: dot-call `.(`, a leading-dot real (`.5`), a dotted
    /// broadcast operator (`.+`, `.==`, …), or the plain field/axis dot.
    fn lex_dot(&mut self, start: u32, line: u32) -> Result<()> {
        match self.peek2() {
            Some('(') => {
                self.bump();
                self.bump();
                self.bracket_depth += 1;
                self.push(TokenKind::DotLParen, start, line);
                Ok(())
            }
            Some(d) if d.is_ascii_digit() => self.lex_number(start, line),
            Some(op) if is_op_start(op) => {
                self.bump(); // consume '.'
                self.lex_dotted_op(op, start, line)
            }
            _ => {
                self.bump();
                self.push(TokenKind::Dot, start, line);
                Ok(())
            }
        }
    }

    fn lex_dotted_op(&mut self, op: char, start: u32, line: u32) -> Result<()> {
        let kind = match op {
            '+' => {
                self.bump();
                TokenKind::DotPlus
            }
            '-' => {
                self.bump();
                TokenKind::DotMinus
            }
            '*' => {
                self.bump();
                TokenKind::DotStar
            }
            '/' => {
                self.bump();
                TokenKind::DotSlash
            }
            '^' => {
                self.bump();
                TokenKind::DotCaret
            }
            '<' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::DotLe
                } else {
                    TokenKind::DotLt
                }
            }
            '>' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::DotGe
                } else {
                    TokenKind::DotGt
                }
            }
            '=' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::DotEqEq
                } else {
                    return Err(self.err_span(start, line, "`.=` is not an operator"));
                }
            }
            '!' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                    TokenKind::DotBangEq
                } else {
                    TokenKind::DotBang
                }
            }
            '&' => {
                self.bump();
                if self.peek() == Some('&') {
                    self.bump();
                    TokenKind::DotAmpAmp
                } else {
                    return Err(self.err_span(start, line, "`.&` is not an operator"));
                }
            }
            '|' => {
                self.bump();
                if self.peek() == Some('|') {
                    self.bump();
                    TokenKind::DotPipePipe
                } else {
                    return Err(self.err_span(start, line, "`.|` is not an operator"));
                }
            }
            _ => return Err(self.err_span(start, line, format!("`.{op}` is not an operator"))),
        };
        self.push(kind, start, line);
        Ok(())
    }

    fn lex_number(&mut self, start: u32, line: u32) -> Result<()> {
        let begin = self.pos;

        // Hex integer `0xF7`.
        if self.peek() == Some('0') && matches!(self.peek2(), Some('x' | 'X')) {
            self.bump();
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_hexdigit() || c == '_') {
                self.bump();
            }
            let text: String = self.chars[begin..self.pos].iter().collect();
            let digits: String = text[2..].chars().filter(|&c| c != '_').collect();
            let v = i64::from_str_radix(&digits, 16)
                .map_err(|_| self.err_span(start, line, format!("invalid hex integer `{text}`")))?;
            self.push(TokenKind::Int(v), start, line);
            return Ok(());
        }

        let mut is_real = false;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '_') {
            self.bump();
        }
        if self.peek() == Some('.') {
            is_real = true;
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '_') {
                self.bump();
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            is_real = true;
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '_') {
                self.bump();
            }
        }

        let text: String = self.chars[begin..self.pos].iter().collect();
        let cleaned: String = text.chars().filter(|&c| c != '_').collect();
        let kind = if is_real {
            let v: f64 = cleaned.parse().map_err(|_| {
                self.err_span(start, line, format!("invalid real literal `{text}`"))
            })?;
            TokenKind::Real(v)
        } else {
            let v: i64 = cleaned.parse().map_err(|_| {
                self.err_span(start, line, format!("invalid integer literal `{text}`"))
            })?;
            TokenKind::Int(v)
        };
        self.push(kind, start, line);
        Ok(())
    }

    fn lex_string(&mut self, start: u32, line: u32) -> Result<()> {
        self.bump(); // opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('"') => {
                    self.push(TokenKind::Str(s), start, line);
                    return Ok(());
                }
                Some('\\') => match self.bump() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('0') => s.push('\0'),
                    Some(o) => {
                        return Err(self.err_span(
                            start,
                            line,
                            format!("invalid string escape `\\{o}`"),
                        ));
                    }
                    None => return Err(self.err_span(start, line, "unterminated string")),
                },
                Some(c) => s.push(c),
                None => return Err(self.err_span(start, line, "unterminated string")),
            }
        }
    }

    fn lex_name(&mut self, start: u32, line: u32) -> Result<()> {
        let begin = self.pos;
        while matches!(self.peek(), Some(c) if c == '_' || c.is_ascii_alphanumeric()) {
            self.bump();
        }
        let name: String = self.chars[begin..self.pos].iter().collect();
        self.push(TokenKind::Name(name), start, line);
        Ok(())
    }

    fn lex_doc(&mut self) -> Result<()> {
        let start = self.byte_pos();
        let line = self.line;
        let is_block = self.peek() == Some('%')
            && self.peek2() == Some('%')
            && self.chars.get(self.pos + 2).copied() == Some('%');

        if is_block {
            self.bump();
            self.bump();
            self.bump();
            let tag = self.read_doc_tag();
            self.consume_to_newline(); // ignore the rest of the opening line
            let mut lines = Vec::new();
            loop {
                if self.peek().is_none() {
                    // Point at the opening fence, not the whole runaway block.
                    return Err(Error::at_span(
                        line,
                        (start, start + 3),
                        "unterminated `%%%` doc block",
                    ));
                }
                if self.at_line_start_fence('%') {
                    self.consume_to_newline();
                    break;
                }
                let lstart = self.pos;
                while !matches!(self.peek(), Some('\n') | None) {
                    self.bump();
                }
                let text: String = self.chars[lstart..self.pos].iter().collect();
                lines.push(text.trim_end_matches('\r').to_string());
                self.bump(); // newline
            }
            self.push(
                TokenKind::Doc(Doc {
                    tag,
                    lines,
                    trailing: false,
                }),
                start,
                line,
            );
            return Ok(());
        }

        // Single-line `% …` (or `%md …`), running to newline or `;`.
        let trailing = self.after_value;
        self.bump(); // %
        let tag = self.read_doc_tag();
        let cstart = self.pos;
        while !matches!(self.peek(), Some('\n') | Some(';') | None) {
            self.bump();
        }
        let raw: String = self.chars[cstart..self.pos].iter().collect();
        let text = raw.trim_end_matches('\r').trim().to_string();
        self.push(
            TokenKind::Doc(Doc {
                tag,
                lines: vec![text],
                trailing,
            }),
            start,
            line,
        );
        Ok(())
    }

    /// Recognize an optional `md` / `typ` markup tag immediately after the
    /// leading `%` / `%%%` (no intervening space). Consumes the tag and one
    /// separating space if present.
    fn read_doc_tag(&mut self) -> Option<String> {
        for tag in ["md", "typ"] {
            let tag_chars: Vec<char> = tag.chars().collect();
            if self.chars[self.pos..].starts_with(&tag_chars[..]) {
                let after = self.chars.get(self.pos + tag_chars.len()).copied();
                if matches!(after, None | Some(' ' | '\t' | '\n' | ';')) {
                    for _ in 0..tag_chars.len() {
                        self.bump();
                    }
                    if self.peek() == Some(' ') {
                        self.bump();
                    }
                    return Some(tag.to_string());
                }
            }
        }
        None
    }
}

fn is_op_start(c: char) -> bool {
    matches!(
        c,
        '+' | '-' | '*' | '/' | '^' | '<' | '>' | '=' | '!' | '&' | '|'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(input: &str) -> Vec<TokenKind> {
        tokenize(input)
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != TokenKind::Eof)
            .collect()
    }

    #[test]
    fn numbers_and_names() {
        assert_eq!(
            kinds("x = 3.5"),
            vec![
                TokenKind::Name("x".into()),
                TokenKind::Assign,
                TokenKind::Real(3.5)
            ]
        );
        assert_eq!(kinds("0xF7"), vec![TokenKind::Int(0xF7)]);
        assert_eq!(kinds("1_000"), vec![TokenKind::Int(1000)]);
    }

    #[test]
    fn operators_and_dotted() {
        assert_eq!(
            kinds("a .+ b"),
            vec![
                TokenKind::Name("a".into()),
                TokenKind::DotPlus,
                TokenKind::Name("b".into())
            ]
        );
        assert_eq!(
            kinds("a ^ b == c"),
            vec![
                TokenKind::Name("a".into()),
                TokenKind::Caret,
                TokenKind::Name("b".into()),
                TokenKind::EqEq,
                TokenKind::Name("c".into())
            ]
        );
    }

    #[test]
    fn newlines_only_at_depth_zero() {
        // Newline inside `(` is continuation; at depth 0 it separates.
        assert_eq!(
            kinds("f(a,\n b)\nx"),
            vec![
                TokenKind::Name("f".into()),
                TokenKind::LParen,
                TokenKind::Name("a".into()),
                TokenKind::Comma,
                TokenKind::Name("b".into()),
                TokenKind::RParen,
                TokenKind::Newline,
                TokenKind::Name("x".into()),
            ]
        );
    }

    #[test]
    fn comments_discarded() {
        assert_eq!(
            kinds("x = 1 # comment\ny = 2"),
            vec![
                TokenKind::Name("x".into()),
                TokenKind::Assign,
                TokenKind::Int(1),
                TokenKind::Newline,
                TokenKind::Name("y".into()),
                TokenKind::Assign,
                TokenKind::Int(2),
            ]
        );
    }

    #[test]
    fn dot_call_and_field() {
        assert_eq!(
            kinds("f.(x)"),
            vec![
                TokenKind::Name("f".into()),
                TokenKind::DotLParen,
                TokenKind::Name("x".into()),
                TokenKind::RParen
            ]
        );
        assert_eq!(
            kinds("r.field"),
            vec![
                TokenKind::Name("r".into()),
                TokenKind::Dot,
                TokenKind::Name("field".into())
            ]
        );
    }
}
