//! The salsa database for the FlatPPL language server.
//!
//! salsa 0.26 (the "salsa-2022" rewrite). Inputs are declared with
//! `#[salsa::input]` on a struct; the database is a struct holding a
//! `salsa::Storage<Self>` and carrying `#[salsa::db]` plus an impl of
//! `salsa::Database`. As of salsa 0.26 `salsa::Database` has no required
//! methods (the event hook moved to `Storage::new`), so the impl is empty.
//! Inputs are constructed with `Field::new(&db, field0, field1, ...)` in
//! declaration order, and fields are read with `field(&db)`. A `#[returns(ref)]`
//! field returns a borrow (`&String`) instead of a clone, and `#[salsa::input(debug)]`
//! opts the generated id into `Debug` (no longer derived by default).

#[salsa::input(debug)]
pub struct SourceFile {
    pub path: String,
    #[returns(ref)]
    pub text: String,
}

/// The set of workspace source files cross-file `load_module` resolution may
/// resolve against. Stored as a `Vec<SourceFile>` — each `SourceFile` is a salsa
/// input (an interned id, hence `Copy + Eq + Hash`), so the `Vec` stores cleanly.
///
/// `analyze` takes this as a parameter so that, by reading `parse(db, dep)` for
/// each resolved dependency, the cross-file salsa dependency edge is recorded:
/// editing a dependency's text input invalidates the importer's analysis.
#[salsa::input]
pub struct FileSet {
    #[returns(ref)]
    pub files: Vec<SourceFile>,
}

/// Host-supplied external `standard_module` catalogues, stored as their RON
/// source strings.
///
/// Storage choice: `flatppl_infer::Catalogue` derives only `Clone + Debug` —
/// it is neither `Hash`/`Eq` nor `salsa::Update`, so a `Vec<Catalogue>` cannot
/// be a salsa input field. Rather than wrap it, we store the RON *sources*
/// (`Vec<String>`, trivially `Update`) and parse them inside `analyze` via
/// [`flatppl_infer::parse_catalogue`]. A RON parse failure becomes an `Error`
/// diagnostic on the analyzed file (offset 0) rather than a panic.
#[salsa::input]
pub struct Catalogues {
    #[returns(ref)]
    pub sources: Vec<String>,
}

#[salsa::db]
#[derive(Default, Clone)]
pub struct Database {
    storage: salsa::Storage<Self>,
}

#[salsa::db]
impl salsa::Database for Database {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_file_input_roundtrips() {
        let db = Database::default();
        let f = SourceFile::new(&db, "a.flatppl".to_string(), "x = 1".to_string());
        assert_eq!(f.text(&db), "x = 1");
    }

    /// VERIFY-FIRST gate for the off-main-thread request worker pool: a cloned
    /// `Database` handle must be `Send + 'static` to move onto a worker thread.
    /// salsa `Storage::clone` shares the `Arc<Zalsa>` memo, so this should hold.
    /// If this fails to compile the worker-pool design is blocked.
    #[test]
    fn database_is_send() {
        fn assert_send<T: Send + 'static>() {}
        assert_send::<Database>();
    }
}
