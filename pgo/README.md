# PGO corpus

Release builds of `flatppl` use profile-guided optimization on pushes to main and on manual runs of `.github/workflows/build.yml`. Pull requests build only the plain binary. `build.sh <target>` does the PGO build for one target:

1. It builds a plain release binary.
2. It builds an instrumented binary and runs the training cases on it, six times each.
3. It merges the profile and builds with `-Cprofile-use`.
4. It runs the held-out cases on the plain and the PGO binary and requires byte-identical stdout, stderr and output files.

Each CI job trains on its own target and runner, from the same commit and toolchain. Profiles are never committed.

- `train.cases` lists the 12 training cases. They cover inference, CPU and GPU StableHLO emission, pyhf conversion, formula conversion, and Arrow file and dictionary-delta input. Their inputs live in `train/`, plus two test fixtures under `crates/`.
- `holdout.cases` lists the 22 held-out cases for the exact-output check. Their inputs live in `holdout/`, plus one test fixture under `crates/`. No input appears in both lists.
- `generate-arrow.py` writes the Arrow inputs. Run it with pyarrow 26.0.0 on Linux. macOS libm writes other bytes.

The job fails when a case exits non-zero. Keep the cases current with the language: fix a model the CLI refuses, do not drop it. Measure throughput by hand on several machines before you change the training cases.
