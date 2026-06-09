'use strict';

// emit-flatpir.cjs — JS-engine side of the cross-engine FlatPIR parity
// harness (driven by ../cross_engine.rs; not a test target itself).
//
//   node emit-flatpir.cjs <flatppl-js-repo> <file.flatppl>...
//
// For every input file: tokenize/parse/lower it with the flatppl-js
// reference engine and print the module's FlatPIR S-expression. Output
// is a sentinel-delimited stream (no JSON dependency on the Rust side):
//
//   <<<FLATPPL-PARITY:BEGIN ok <path>
//   ...FlatPIR text...
//   <<<FLATPPL-PARITY:END
//   <<<FLATPPL-PARITY:BEGIN err <path>
//   <single-line error message>
//   <<<FLATPPL-PARITY:END
//
// Requires node >= 22.18 (built-in TypeScript type-stripping; the engine
// is plain .ts requires) — the Rust harness probes the version first.

const fs = require('node:fs');
const path = require('node:path');

const [, , repoDir, ...files] = process.argv;
if (!repoDir || files.length === 0) {
  console.error('usage: node emit-flatpir.cjs <flatppl-js-repo> <file.flatppl>...');
  process.exit(2);
}

// Resolves via packages/engine/package.json "main" (./index.ts), exactly
// like the engine's own test suite does with require('..').
const engine = require(path.resolve(repoDir, 'packages', 'engine'));

const oneLine = (s) => String(s).replace(/\s+/g, ' ').trim();
const chunks = [];

for (const file of files) {
  let record;
  try {
    const src = fs.readFileSync(file, 'utf8');
    const res = engine.processSource(src);
    const errors = (res.diagnostics || []).filter((d) => d.severity === 'error');
    if (errors.length > 0) {
      record = ['err', errors.map((e) => oneLine(e.message)).join('; ')];
    } else {
      record = ['ok', engine.pirSexpr.toSexpr(res.loweredModule)];
    }
  } catch (e) {
    record = ['err', oneLine((e && e.stack) ? e.message : e)];
  }
  chunks.push(
    `<<<FLATPPL-PARITY:BEGIN ${record[0]} ${file}\n${record[1]}\n<<<FLATPPL-PARITY:END\n`,
  );
}

process.stdout.write(chunks.join(''));
