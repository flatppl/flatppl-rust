# flatppl-cli — the `flatppl` command-line driver

The `flatppl` binary: a thin surface over the library crates. Conversion logic lives in
[`flatppl-syntax`](../syntax) / [`flatppl-flatpir`](../flatpir) (FlatPPL ↔ FlatPIR) and
[`flatppl-hs3`](../hs3) (HS3 / pyhf import); inference in [`flatppl-infer`](../infer).

```sh
flatppl convert model.flatppl model.flatpir     # FlatPPL → FlatPIR
flatppl convert model.flatpir model.flatppl     # FlatPIR → FlatPPL (--syntax minimal for lowered form)
flatppl convert --from hs3  model.json m.flatppl  # import native HS3
flatppl convert --from pyhf ws.json    m.flatppl  # import a pyhf workspace
flatppl infer model.flatppl typed.flatpir       # type/phase-annotated FlatPIR (--level phase|type|…|shape)
flatppl completions zsh > ~/.zfunc/_flatppl     # shell completions (bash|zsh|fish|powershell|elvish)
```

Formats are inferred from file extensions. Generated files carry a single minimal banner comment
(`AUTOMATICALLY GENERATED - do not edit`) - deliberately no timestamp, user, host, platform, or
command line, which would leak personal/system information; model imports also stamp a leading
`flatppl_compat` binding recording the targeted FlatPPL version. `--no-header` omits the banner.
Verbs are opt-in cargo features (`infer`, `hs3`; both on by default).

Install from the repository:

```sh
cargo install --git https://github.com/flatppl/flatppl-rust flatppl-cli
```
