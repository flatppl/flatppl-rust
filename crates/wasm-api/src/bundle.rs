//! Byte-based ZIP adapters. The shared bundle crate owns validation and limits.

/// A mounted archive. Reading members never extracts files or fetches URLs.
pub struct ZipBundle(flatppl_bundle::Bundle);

impl ZipBundle {
    pub fn open(identity: &str, bytes: &[u8]) -> Result<Self, String> {
        flatppl_bundle::Bundle::open(identity, bytes.into(), flatppl_bundle::Limits::default())
            .map(Self)
            .map_err(|error| error.to_string())
    }

    pub fn entrypoint(&self) -> String {
        self.0.entrypoint().to_string()
    }

    pub fn identity(&self, path: &str) -> String {
        self.0.identity_for(path)
    }

    /// Member metadata as JSON. File contents remain available through `read`.
    pub fn members(&self) -> String {
        let members: Vec<_> = self
            .0
            .members()
            .map(|member| {
                let kind = match member.kind {
                    flatppl_bundle::MemberKind::File => "file",
                    flatppl_bundle::MemberKind::Directory => "directory",
                    flatppl_bundle::MemberKind::Symlink => "symlink",
                };
                serde_json::json!({"path": member.path, "size": member.size, "kind": kind})
            })
            .collect();
        serde_json::Value::Array(members).to_string()
    }

    pub fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        self.0
            .read(path)
            .map(|bytes| bytes.to_vec())
            .map_err(|error| error.to_string())
    }

    pub fn resolve(&self, importer: &str, source: &str) -> Result<String, String> {
        self.0
            .resolve(importer, source)
            .map_err(|error| error.to_string())
    }

    pub fn is_dir(&self, path: &str) -> Result<bool, String> {
        self.0.is_dir(path).map_err(|error| error.to_string())
    }

    pub fn nested(&self, path: &str) -> Result<Self, String> {
        self.0
            .nested(path)
            .map(Self)
            .map_err(|error| error.to_string())
    }

    pub fn subdirectory(&self, path: &str) -> Result<Self, String> {
        self.0
            .subdirectory(path)
            .map(Self)
            .map_err(|error| error.to_string())
    }
}

/// Write a file map (`{"main.flatppl":[...], "data.json":[...]}`) as ZIP bytes.
/// `compression` is `store` or `deflate`. File values contain unsigned bytes.
pub fn pack_bundle(files: &str, compression: &str) -> Result<Vec<u8>, String> {
    let files: std::collections::BTreeMap<String, Vec<u8>> =
        serde_json::from_str(files).map_err(|error| format!("bundle files: {error}"))?;
    let compression = match compression {
        "store" => flatppl_bundle::Compression::Stored,
        "deflate" => flatppl_bundle::Compression::Deflated,
        _ => return Err("bundle compression must be store or deflate".into()),
    };
    flatppl_bundle::pack(files, compression).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_api_preserves_files_and_nested_identities() {
        let bundle = ZipBundle::open(
            "portable.zip",
            include_bytes!("../../bundle/tests/fixtures/portable.zip"),
        )
        .unwrap();
        assert_eq!(bundle.entrypoint(), "main.flatppl");
        assert_eq!(bundle.read("data/raw.bin").unwrap(), [0, 1, 255, 128]);
        assert_eq!(bundle.read("data/observed-link.json").unwrap(), b"[1, 2]\n");
        let child = bundle.nested("child.zip").unwrap();
        assert_ne!(
            bundle.identity("main.flatppl"),
            child.identity("main.flatppl")
        );
        assert_eq!(
            child.read("main.flatppl").unwrap(),
            b"offset = external(reals)\nvalue = offset + 2\n"
        );
        let files = serde_json::json!({
            "main.flatppl": bundle.read("main.flatppl").unwrap(),
            "raw.bin": bundle.read("data/raw.bin").unwrap()
        });
        let exported = pack_bundle(&files.to_string(), "deflate").unwrap();
        let reopened = ZipBundle::open("export.zip", &exported).unwrap();
        assert_eq!(reopened.read("raw.bin").unwrap(), [0, 1, 255, 128]);
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct ZipBundle(super::ZipBundle);

    #[wasm_bindgen]
    impl ZipBundle {
        #[wasm_bindgen(constructor)]
        pub fn new(identity: &str, bytes: &[u8]) -> Result<ZipBundle, JsError> {
            super::ZipBundle::open(identity, bytes)
                .map(Self)
                .map_err(|error| JsError::new(&error))
        }

        #[wasm_bindgen(getter)]
        pub fn entrypoint(&self) -> String {
            self.0.entrypoint()
        }

        pub fn identity(&self, path: &str) -> String {
            self.0.identity(path)
        }

        pub fn members(&self) -> String {
            self.0.members()
        }

        pub fn read(&self, path: &str) -> Result<Vec<u8>, JsError> {
            self.0.read(path).map_err(|error| JsError::new(&error))
        }

        pub fn resolve(&self, importer: &str, source: &str) -> Result<String, JsError> {
            self.0
                .resolve(importer, source)
                .map_err(|error| JsError::new(&error))
        }

        pub fn is_dir(&self, path: &str) -> Result<bool, JsError> {
            self.0.is_dir(path).map_err(|error| JsError::new(&error))
        }

        pub fn nested(&self, path: &str) -> Result<ZipBundle, JsError> {
            self.0
                .nested(path)
                .map(Self)
                .map_err(|error| JsError::new(&error))
        }

        pub fn subdirectory(&self, path: &str) -> Result<ZipBundle, JsError> {
            self.0
                .subdirectory(path)
                .map(Self)
                .map_err(|error| JsError::new(&error))
        }
    }

    #[wasm_bindgen]
    pub fn pack_bundle(files: &str, compression: &str) -> Result<Vec<u8>, JsError> {
        super::pack_bundle(files, compression).map_err(|error| JsError::new(&error))
    }
}
