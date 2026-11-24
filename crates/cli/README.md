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
flatppl pack model.flatppl model.flatppl.zip    # include imports and data
flatppl unpack model.flatppl.zip working-copy  # extract into a new directory
flatppl completions zsh > ~/.zfunc/_flatppl     # shell completions (bash|zsh|fish|powershell|elvish)
```

Model inputs also accept directory and ZIP bundles containing `main.flatppl`.
Imports and data remain inside the bundle, including nested archives.
`pack` reads fixed source paths through the parser and includes binary data unchanged.
Remote dependencies must already be cached with `flatppl prepare`.
Both export and extraction require new destination paths.
Extract a working copy before formatting or editing archive members.

Formats are inferred from file extensions. Generated files carry a single minimal banner comment
(`AUTOMATICALLY GENERATED - do not edit`) - deliberately no timestamp, user, host, platform, or
command line, which would leak personal/system information; model imports also stamp a leading
`flatppl_compat` binding recording the targeted FlatPPL version. `--no-header` omits the banner.
Verbs are controlled by Cargo features. Conversion, inference, preparation, formatting,
linting, determinization, and StableHLO emission are enabled by default. HS3 and pyhf
import require the additional `hs3` feature.

Install from the repository:

```sh
cargo install --git https://github.com/flatppl/flatppl-rust flatppl-cli --features hs3
```

## Egglog rules for teaching output

Use `flatppl egglog-rules` to list the names of rules, rulesets and optimiser
passes. The `optimise`, `determinize`, `stablehlo`, and `stan` commands accept a
repeatable `--disable-egglog` option. Each occurrence takes `rule:<id>`,
`ruleset:<id>`, `pass:<id>`, or `all`. The list covers every stage. A selector
affects only the stages that use the selected rules. `pass:demand-prune` keeps
each read of a record, tuple or vector cell as written, instead of replacing it
by the cell's value. The `stan` command
requires the `stan` Cargo feature.

For example, keep both negations in a lesson while simplifying other
expressions, then render the result as Markdown.

```sh
cat > lesson.flatppl <<'EOF'
detected = external(booleans)
reading = external(reals)
confirmed = lnot(lnot(detected))
reported = identity(reading)
EOF
flatppl optimise lesson.flatppl --disable-egglog rule:portable.double-negation -o teaching.flatppl
flatppl convert teaching.flatppl teaching.md
```

Use `flatppl convert lesson.flatppl teaching.md` to render the authored graph
directly. `--disable-egglog all` suppresses every Egglog rule and optimiser
pass for the command. Ordinary compilation steps, including projection and mandatory target lowering,
still run.
