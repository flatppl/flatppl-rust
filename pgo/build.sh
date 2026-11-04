#!/usr/bin/env bash
# Builds a PGO release `flatppl` for one target and checks it against a plain
# release build of the same tree. Usage: pgo/build.sh <target-triple>
# The plain binary lands in target/, the PGO binary in target-pgo/use/.
set -euo pipefail
target=${1:?target triple}
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=$repo/target-pgo
exe=flatppl
[[ $target == *windows* ]] && exe=flatppl.exe

# rustc and the profiling runtime need native paths on Windows, bash needs POSIX paths.
native() { cygpath -m "$1" 2>/dev/null || printf '%s\n' "$1"; }
posix() { cygpath -u "$1" 2>/dev/null || printf '%s\n' "$1"; }
sha256() { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; }
report() {
    printf '%s\n' "$1"
    if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then printf '%s\n\n' "$1" >> "$GITHUB_STEP_SUMMARY"; fi
}
build() { # <target-dir> <rustflags>
    RUSTFLAGS=$2 cargo build --release --locked --all-features -p flatppl-cli \
        --target "$target" --target-dir "$1"
}
# Runs every case of a manifest once. Outputs go to $work/out/<case>.
run_cases() { # <manifest> <binary>
    local name rest args
    rm -rf "$work/out"
    while read -r name rest; do
        [[ -z $name || $name == \#* ]] && continue
        mkdir -p "$work/out/$name"
        read -ra args <<< "${rest//\{out\}/$work/out/$name}"
        if ! (cd "$repo/pgo" && "$2" "${args[@]}") \
            > "$work/out/$name/stdout" 2> "$work/out/$name/stderr"; then
            echo "case $name failed with $2:" >&2
            cat "$work/out/$name/stderr" >&2
            return 1
        fi
    done < "$repo/pgo/$1"
}

cd "$repo"
# A reused use-phase target dir would keep dependencies built with an old profile.
rm -rf "$work"
mkdir -p "$work"
host=$(rustc -vV | sed -n 's/^host: //p')
profdata=$(posix "$(rustc --print sysroot)")/lib/rustlib/$host/bin/llvm-profdata
report "PGO \`$target\`: $(rustc -V), $("$profdata" --version | grep -i 'llvm version' | xargs)"

SECONDS=0
build target ""
plain=$repo/target/$target/release/$exe
report "Plain build: ${SECONDS} s"

# Rust-only instrumentation: cc 1.11 does not forward -Cprofile-* to the C
# compiler. Select instrumentation stays off in both phases (rust#163616).
SECONDS=0
build "$work/instrumented" "-Cprofile-generate=$(native "$work/profiles") -Cllvm-args=-pgo-instr-select=false"
report "Instrumented build: ${SECONDS} s"

# Six runs per case reproduce the profile weights the throughput gain was measured with.
SECONDS=0
for _ in 1 2 3 4 5 6; do run_cases train.cases "$work/instrumented/$target/release/$exe"; done
"$profdata" merge -o "$work/flatppl.profdata" "$work/profiles" 2>&1 | tee "$work/merge.log"
if [[ -s $work/merge.log ]]; then echo "llvm-profdata merge reported diagnostics" >&2; exit 1; fi
report "Training and merge: ${SECONDS} s, $("$profdata" show "$work/flatppl.profdata" | grep 'Total functions')"

SECONDS=0
if ! build "$work/use" "-Cprofile-use=$(native "$work/flatppl.profdata") -Cllvm-args=-pgo-instr-select=false -Cllvm-args=-pgo-warn-missing-function" \
    > "$work/use.log" 2>&1; then
    tail -n 50 "$work/use.log" >&2
    exit 1
fi
pgo=$work/use/$target/release/$exe
# Cargo caps lints for registry crates, so only workspace crates report here.
missing=$(grep -c 'no profile data available for function' "$work/use.log" || true)
if grep -v 'no profile data available for function' "$work/use.log" \
    | grep -i -E 'hash mismatch|corrupt|out of date|inconsistent|malformed|invalid profile' >&2; then
    echo "profile-use reported mismatched or corrupt profile data" >&2
    exit 1
fi
report "Profile-use build: ${SECONDS} s, $missing \"no profile data\" notices"

run_cases holdout.cases "$plain"
mv "$work/out" "$work/plain-out"
run_cases holdout.cases "$pgo"
if ! diff -r "$work/plain-out" "$work/out" >&2; then
    echo "PGO outputs differ from the plain build" >&2
    exit 1
fi
report "Held-out outputs match: $(grep -c -v -E '^(#|$)' pgo/holdout.cases) cases"
report "$(printf 'plain %s bytes, sha256 %s\n\npgo %s bytes, sha256 %s' \
    "$(wc -c < "$plain" | xargs)" "$(sha256 "$plain" | cut -d' ' -f1)" \
    "$(wc -c < "$pgo" | xargs)" "$(sha256 "$pgo" | cut -d' ' -f1)")"
