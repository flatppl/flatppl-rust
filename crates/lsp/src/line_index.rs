//! Byte-offset ↔ (line, UTF-16 column) conversion. LSP `Position.character` is a
//! UTF-16 code-unit count within the line; FlatPPL spans are byte offsets.

use std::sync::Arc;

/// A 0-based (line, UTF-16 column) position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pos {
    pub line: u32,
    pub character: u32, // UTF-16 code units within the line
}

#[derive(Clone, Debug, PartialEq, Eq, salsa::Update)]
pub struct LineIndex {
    text: Arc<str>,
    line_starts: Vec<u32>, // byte offset of each line's first char; [0] = 0
}

impl LineIndex {
    pub fn new(text: &str) -> Self {
        let mut line_starts = vec![0u32];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push((i + 1) as u32);
            }
        }
        LineIndex {
            text: text.into(),
            line_starts,
        }
    }

    /// The largest char boundary `<= byte` within `text` (clamped to len).
    /// `str::floor_char_boundary` is unstable, so walk down manually.
    fn floor_boundary(&self, mut byte: usize) -> usize {
        let len = self.text.len();
        if byte >= len {
            return len;
        }
        while byte > 0 && !self.text.is_char_boundary(byte) {
            byte -= 1;
        }
        byte
    }

    /// The byte offset of the end of line `line`'s CONTENT.
    ///
    /// For a NON-final line this is the next line's start minus the trailing
    /// `\n` (and a preceding `\r` for CRLF). For the FINAL line it is
    /// `text.len()` with NOTHING stripped: the final line has no terminator —
    /// any trailing `\n` belongs to the *previous* line, making the final line
    /// an empty line that starts at `text.len()`. Stripping there would push
    /// the content end BELOW the line start and panic the `text[start..end]`
    /// slices in `position` / `offset` (e.g. a cursor at EOF of a file that
    /// ends in a newline).
    fn line_content_end(&self, line: usize) -> usize {
        match self.line_starts.get(line + 1) {
            // Not the final line: `next` is one past THIS line's '\n'.
            Some(&next) => {
                let mut end = next as usize;
                if end > 0 && self.text.as_bytes().get(end - 1) == Some(&b'\n') {
                    end -= 1;
                }
                if end > 0 && self.text.as_bytes().get(end - 1) == Some(&b'\r') {
                    end -= 1;
                }
                end
            }
            // Final line: no terminator to strip; content ends at text.len().
            None => self.text.len(),
        }
    }

    /// Byte offset → (line, UTF-16 column). Clamps to the end if out of range.
    pub fn position(&self, byte: u32) -> Pos {
        let byte = self.floor_boundary(byte as usize);
        let line = match self.line_starts.binary_search(&(byte as u32)) {
            Ok(l) => l,
            Err(next) => next - 1,
        };
        let line_start = self.line_starts[line] as usize;
        // Count UTF-16 units up to `byte`, but never past the line's content end
        // (so a byte pointing at a trailing \r/\n does not over-count).
        let col_end = byte.min(self.line_content_end(line));
        let character: u32 = self.text[line_start..col_end]
            .chars()
            .map(|c| c.len_utf16() as u32)
            .sum();
        Pos {
            line: line as u32,
            character,
        }
    }

    /// (line, UTF-16 column) → byte offset. Clamps within the line.
    pub fn offset(&self, pos: Pos) -> u32 {
        let line = (pos.line as usize).min(self.line_starts.len().saturating_sub(1));
        let line_start = self.line_starts[line] as usize;
        let content_end = self.line_content_end(line);
        let mut u16s = 0u32;
        for (off, c) in self.text[line_start..content_end].char_indices() {
            if u16s >= pos.character {
                return (line_start + off) as u32;
            }
            u16s += c.len_utf16() as u32;
        }
        content_end as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_to_position_and_back() {
        let li = LineIndex::new("ab\ncde");
        let p = li.position(4); // 'd' → line 1, col 1
        assert_eq!((p.line, p.character), (1, 1));
        assert_eq!(li.offset(p), 4);
    }

    #[test]
    fn utf16_columns() {
        // é: 2 bytes / 1 u16; 𐐷: 4 bytes / 2 u16; x at byte 6
        let li = LineIndex::new("é𐐷x");
        assert_eq!(li.position(6).character, 3); // 1 + 2 = 3 u16 units before x
    }

    #[test]
    fn first_and_last_positions() {
        let li = LineIndex::new("ab\ncde");
        assert_eq!(
            li.position(0),
            Pos {
                line: 0,
                character: 0
            }
        );
        let last = li.position(6); // EOF
        assert_eq!(last.line, 1);
    }

    #[test]
    fn line_start_byte_maps_to_col_zero() {
        let li = LineIndex::new("ab\ncde");
        assert_eq!(
            li.position(3),
            Pos {
                line: 1,
                character: 0
            }
        ); // 'c' at line start
    }

    #[test]
    fn crlf_columns_exclude_carriage_return() {
        // "ab\r\ncde": line 0 content is "ab" (2 u16), the \r and \n are the
        // terminator and must not count toward columns.
        let li = LineIndex::new("ab\r\ncde");
        // byte 2 = the '\r' → end of line 0 content → column 2, NOT 2-then-\r.
        assert_eq!(
            li.position(2),
            Pos {
                line: 0,
                character: 2
            }
        );
        // byte 5 = 'd' on line 1 (line starts at byte 4) → column 1.
        assert_eq!(
            li.position(5),
            Pos {
                line: 1,
                character: 1
            }
        );
        // round-trip: column past EOL on line 0 clamps to end-of-content (byte 2).
        assert_eq!(
            li.offset(Pos {
                line: 0,
                character: 99
            }),
            2
        );
    }

    #[test]
    fn position_on_non_char_boundary_does_not_panic() {
        // "é" is 2 bytes (0xC3 0xA9); byte 1 is INSIDE it. A diagnostic span
        // landing mid-char must clamp down to a boundary, not panic.
        let li = LineIndex::new("é");
        let p = li.position(1); // must not panic
        assert_eq!(
            p,
            Pos {
                line: 0,
                character: 0
            }
        ); // floored to byte 0
    }

    #[test]
    fn offset_overshoot_stays_on_requested_line() {
        // "ab\ncde": an overshoot column on line 0 must land at the end of
        // line 0 content (byte 2 = before '\n'), NOT byte 3 (start of line 1).
        let li = LineIndex::new("ab\ncde");
        assert_eq!(
            li.offset(Pos {
                line: 0,
                character: 99
            }),
            2
        );
    }

    #[test]
    fn final_empty_line_at_eof_does_not_panic() {
        // A document ending in a newline has a final EMPTY line that starts at
        // text.len(). position()/offset() on it must not slice text[len..len-1]
        // — the reversed-range panic that crashed the LSP ("byte range starts
        // at N but ends at N-1") for a cursor/diagnostic at EOF.
        for text in ["ab\n", "ab\r\n", "\n"] {
            let li = LineIndex::new(text);
            let last_line = (li.line_starts.len() - 1) as u32;
            let len = text.len() as u32;
            let p = li.position(len); // EOF on the final empty line
            assert_eq!(p.line, last_line);
            assert_eq!(p.character, 0);
            assert_eq!(li.offset(p), len);
            // Overshoot column on the empty final line clamps to len, no panic.
            assert_eq!(
                li.offset(Pos {
                    line: last_line,
                    character: 99
                }),
                len
            );
        }
    }
}
