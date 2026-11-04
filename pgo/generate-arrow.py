"""Arrow inputs for the PGO training and held-out cases.

The committed files were written on Linux (glibc) with pyarrow 26.0.0.
macOS libm `cos` differs from glibc by 1 ULP, so other hosts write other bytes.
"""

import math
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).parent


def batch(start, count, levels=None):
    indices = range(start, start + count)
    columns = {
        "y": pa.array([1 + 0.3 * (i % 97) + math.sin(i) for i in indices]),
        "x": pa.array([(i % 97) / 10 for i in indices], type=pa.float32()),
        "z": pa.array([math.cos(i / 9) for i in indices]),
    }
    if levels:
        columns["f"] = pa.DictionaryArray.from_arrays(
            pa.array([i % levels for i in indices], type=pa.int16()),
            pa.array([f"group_{i}" for i in range(levels)]),
        )
    return pa.record_batch(columns)


def write(path, batches, file=False, deltas=False):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    new_writer = pa.ipc.new_file if file else pa.ipc.new_stream
    options = pa.ipc.IpcWriteOptions(emit_dictionary_deltas=deltas)
    with pa.OSFile(str(path), "wb") as sink:
        with new_writer(sink, batches[0].schema, options=options) as writer:
            for value in batches:
                writer.write_batch(value)
            stats = writer.stats
    assert not deltas or stats.num_dictionary_deltas > 0
    print(path.relative_to(ROOT), path.stat().st_size, stats.num_dictionary_deltas)


write("train/arrow/train.arrow", [batch(0, 4096, 4)], file=True)
write("train/arrow/train-delta.arrows", [batch(0, 2048, 4), batch(2048, 2048, 8)], deltas=True)
numeric = [batch(100003 + i * 1024, 1024) for i in range(2)]
write("holdout/arrow/numeric.arrow", numeric, file=True)
write("holdout/arrow/numeric.arrows", numeric)
write("holdout/arrow/dictionary.arrow", [batch(200003 + i * 1024, 1024, 12) for i in range(2)], file=True)
write("holdout/arrow/delta.arrows", [batch(300003 + i * 512, 512, 4 * (i + 1)) for i in range(4)], deltas=True)
