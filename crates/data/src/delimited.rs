//! CSV and WSV (§07): column names in the first row, then one row per record.
//! Fields are trimmed, and blank lines and lines starting with `#` are skipped.
//! Cells stay text; the reader that consumes them decides their types.

use crate::{Error, Result};
use std::collections::HashSet;

/// A delimited table as text: the header names and the rows of trimmed cells. An
/// empty cell is a missing value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextTable {
    pub names: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl TextTable {
    /// The cells of column `index`, top to bottom.
    pub fn column(&self, index: usize) -> impl Iterator<Item = &str> {
        self.rows.iter().map(move |row| row[index].as_str())
    }

    fn checked(names: Vec<String>, rows: Vec<Vec<String>>) -> Result<Self> {
        let mut seen = HashSet::new();
        if names
            .iter()
            .any(|name| name.is_empty() || !seen.insert(name.as_str()))
        {
            return Err(Error(
                "column names in the first row must be nonempty and distinct".into(),
            ));
        }
        if let Some((i, row)) = rows
            .iter()
            .enumerate()
            .find(|(_, row)| row.len() != names.len())
        {
            return Err(Error(format!(
                "row {} has {} fields, but the first row names {} columns",
                i + 1,
                row.len(),
                names.len()
            )));
        }
        Ok(Self { names, rows })
    }
}

/// Comma-separated values, with RFC 4180 quoting.
pub fn read_csv(bytes: &[u8]) -> Result<TextTable> {
    let error = |e: csv::Error| Error(format!("CSV: {e}"));
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .comment(Some(b'#'))
        .flexible(true)
        .from_reader(bytes);
    let names = reader
        .headers()
        .map_err(error)?
        .iter()
        .map(str::to_owned)
        .collect();
    let rows = reader
        .records()
        .map(|record| record.map(|r| r.iter().map(str::to_owned).collect()))
        .collect::<std::result::Result<_, _>>()
        .map_err(error)?;
    TextTable::checked(names, rows)
}

/// Whitespace-separated values: runs of spaces and tabs separate fields.
pub fn read_wsv(bytes: &[u8]) -> Result<TextTable> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("WSV: {e}")))?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });
    let names = lines
        .next()
        .ok_or_else(|| Error("WSV: missing column names".into()))?;
    TextTable::checked(names, lines.collect())
}
