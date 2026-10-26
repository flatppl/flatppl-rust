"""Regenerate the `load_data` fixtures (spec §07 formats).

Arrow IPC comes from polars, an independent writer. Safetensors is packed by
hand from the format definition: an 8-byte little-endian header length, a JSON
header of `{key: {dtype, shape, data_offsets}}`, then the data buffer.

Usage: python generate.py  (writes next to this file)
"""

import json
import struct
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent

# One table in every format: a real, an integer and a boolean column, an extra
# column a selection may omit, and three rows of which tests read two.
events = {
    "a": [1.5, -2.0, 3.25],
    "b": [2, 0, 7],
    "flag": [True, False, True],
    "unused": [9.0, 9.0, 9.0],
}

frame = pl.DataFrame(events, schema={"a": pl.Float64, "b": pl.Int32, "flag": pl.Boolean, "unused": pl.Float32})
frame.write_ipc(HERE / "events.arrow")
frame.write_ipc_stream(HERE / "events.arrows")
(HERE / "events.json").write_text(json.dumps([dict(zip(events, row)) for row in zip(*events.values())]) + "\n")
(HERE / "events_columns.json").write_text(json.dumps(events) + "\n")
cells = lambda v: str(v).lower() if isinstance(v, bool) else str(v)
rows = [",".join(events)] + [",".join(cells(v) for v in row) for row in zip(*events.values())]
(HERE / "events.csv").write_text("\n".join(rows) + "\n")
rows = ["  ".join(events)] + ["\t".join(cells(v) for v in row) for row in zip(*events.values())]
(HERE / "events.wsv").write_text("# whitespace-separated\n" + "\n".join(rows) + "\n")
(HERE / "column.csv").write_text("x\n0.5\n-1\n2.75\n")

DTYPES = {np.float64: "F64", np.float32: "F32", np.float16: "F16", np.int64: "I64",
          np.int32: "I32", np.uint8: "U8", np.bool_: "BOOL"}


def safetensors(path, tensors, dtype_override=None):
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for key, array in tensors.items():
        blob = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
        dtype = (dtype_override or {}).get(key, DTYPES[array.dtype.type])
        header[key] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


safetensors(HERE / "net.safetensors", {
    "W": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]], dtype=np.float32),
    "b": np.array([0.25, -0.5, 8.0], dtype=np.float64),
    "enc.layer.weight": np.array([3, -4], dtype=np.int32),
    ".enc.layer.mask.": np.array([True, False]),
    "enc.layer.half": np.array([1.5, -0.125], dtype=np.float16),
    "count": np.array(5, dtype=np.int64),
})
one = np.array([1.0], dtype=np.float32)
for name, tensors in {
    "empty_segment": {"enc..weight": one},
    "empty_key": {"...": one},
    "duplicate_path": {"enc.weight": one, ".enc.weight": one},
    "prefix_and_leaf": {"enc": one, "enc.weight": one},
}.items():
    safetensors(HERE / f"{name}.safetensors", tensors)
safetensors(HERE / "fp8.safetensors", {"x": np.array([1], dtype=np.uint8)}, {"x": "F8_E4M3"})
