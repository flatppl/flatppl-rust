//! Byte-offset ↔ (line, UTF-16 column) conversion. LSP `Position.character` is a
//! UTF-16 code-unit count within the line; FlatPPL spans are byte offsets.

/// A 0-based (line, UTF-16 column) position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pos {
    pub line: u32,
    pub character: u32, // UTF-16 code units within the line
}

pub struct LineIndex {
    text: String,
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
            text: text.to_string(),
            line_starts,
        }
    }

    /// Byte offset → (line, UTF-16 column). Clamps to the end if out of range.
    pub fn position(&self, byte: u32) -> Pos {
        let byte = byte.min(self.text.len() as u32);
        // line = last line_start <= byte
        let line = match self.line_starts.binary_search(&byte) {
            Ok(l) => l,
            Err(next) => next - 1,
        };
        let line_start = self.line_starts[line] as usize;
        // UTF-16 column = sum of len_utf16 over chars in [line_start, byte)
        let character: u32 = self.text[line_start..byte as usize]
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
        let line = (pos.line as usize).min(self.line_starts.len() - 1);
        let line_start = self.line_starts[line] as usize;
        let line_end = self
            .line_starts
            .get(line + 1)
            .map(|&s| s as usize)
            .unwrap_or(self.text.len());
        // walk chars accumulating UTF-16 units until we reach pos.character
        let mut u16s = 0u32;
        for (off, c) in self.text[line_start..line_end].char_indices() {
            if u16s >= pos.character {
                return (line_start + off) as u32;
            }
            u16s += c.len_utf16() as u32;
        }
        line_end as u32
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
}
