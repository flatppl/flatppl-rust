# Arrow transport fixtures

Generated independently with PyArrow 25.0.0 using `generate.py`. The Rust test
suite reads the frozen bytes; it does not require Python or PyArrow.

The training file and stream encode the same table: a float32 predictor with a
missing row, an ordered dictionary factor with levels `b, a`, a float64 response,
and an unused all-null column. The prediction stream omits the response, changes
the row count and physical dictionary order, and includes a missing predictor.
Preparation must reuse the training recipe and report original retained rows.
