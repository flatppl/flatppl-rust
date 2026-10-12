"""Independent PyArrow wire fixtures for the Rust formula reader."""

from pathlib import Path

import pyarrow as pa


DESTINATION = Path(__file__).parent
training = pa.table(
    {
        "y": pa.array([1, 2, 4, 8], type=pa.float64()),
        "x": pa.array([0, 1, None, 3], type=pa.float32()),
        "f": pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 0, 1], type=pa.int8()),
            pa.array(["b", "a"]),
            ordered=True,
        ),
        "unused": pa.array([None] * 4, type=pa.float64()),
    }
)
prediction = pa.table(
    {
        "x": pa.array([2, None, 4], type=pa.float32()),
        "f": pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 1], type=pa.int8()),
            pa.array(["a", "b"]),
            ordered=True,
        ),
        "unused": pa.array([None] * 3, type=pa.float64()),
    }
)
for filename, table, writer in (
    ("training.arrow", training, pa.ipc.new_file),
    ("training.arrows", training, pa.ipc.new_stream),
    ("prediction.arrows", prediction, pa.ipc.new_stream),
):
    with pa.OSFile(str(DESTINATION / filename), "wb") as output:
        with writer(output, table.schema) as stream:
            stream.write_table(table)
    print(filename, (DESTINATION / filename).stat().st_size)
print("pyarrow", pa.__version__)
