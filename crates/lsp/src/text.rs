//! Lexical skips for editor operations on incomplete FlatPPL source.

/// Skip §05 plain/doc comments without interpreting punctuation in their body.
/// An unfinished block is normal while editing and consumes the remaining prefix.
pub(crate) fn skip_comment(bytes: &[u8], start: usize, end: usize) -> usize {
    let closing = [bytes[start]; 3];
    let fence_start = bytes[start..end].starts_with(&closing)
        && bytes[..start]
            .iter()
            .rev()
            .find(|b| !matches!(b, b' ' | b'\t'))
            .is_none_or(|b| matches!(b, b'\r' | b'\n'));
    let line_comment_end = || {
        (start..end)
            .find(|&i| matches!(bytes[i], b';' | b'\r' | b'\n'))
            .unwrap_or(end)
    };
    if !fence_start {
        return line_comment_end();
    }
    let line_end = |from: usize| {
        (from..end)
            .find(|&i| matches!(bytes[i], b'\r' | b'\n'))
            .unwrap_or(end)
    };
    let first_end = line_end(start);
    let opening = bytes[start..first_end].trim_ascii();
    let block = matches!(opening, b"###" | b"%%%" | b"%%%md" | b"%%%typ");
    if !block {
        return line_comment_end();
    }
    let mut next = first_end;
    while next < end {
        next += 1;
        let stop = line_end(next);
        if bytes[next..stop].trim_ascii() == closing {
            return stop;
        }
        next = stop;
    }
    end
}

pub(crate) fn skip_string(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut at = start + 1;
    while at < end {
        match bytes[at] {
            b'\\' => at = (at + 2).min(end),
            b'"' => return at + 1,
            _ => at += 1,
        }
    }
    end
}

pub(crate) fn skip_trivia(bytes: &[u8], mut at: usize, end: usize) -> usize {
    loop {
        while at < end && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        if at < end && matches!(bytes[at], b'#' | b'%') {
            at = skip_comment(bytes, at, end);
        } else {
            return at;
        }
    }
}
