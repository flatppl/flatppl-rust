"""Check an f32 dot with cancellation through the public CLI.

Usage: python check.py /path/to/flatppl [cpu|gpu]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call

MODEL = Path(__file__).with_name("model.flatppl")


def compile_query(binary, query, target, *extra):
    ir = subprocess.check_output(
        [binary, "stablehlo", "--target", target, "--dtype", "f32", query, *extra],
        text=True,
    ).replace("@logdensity(", "@main(")
    return jax.jit(lambda *args: hlo_call(*args, source=ir))


def check(binary, target, derivative):
    # Every pair is (1 + 2^-13)(1 - 2^-13) - 1 = -2^-26.
    # The first product alone rounds to exactly 1 in f32.
    a = np.tile(np.r_[np.tile([1 + 2**-13, 1], 32), 0], (2, 1)).astype(np.float32)
    b = np.r_[np.tile([1 - 2**-13, -1], 32), 0].astype(np.float32)
    scale = np.float32(1)
    expected = np.float32(-2.0**-20)
    args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(scale))

    primal = compile_query(binary, MODEL, target)
    np.testing.assert_array_equal(primal(*args), [expected], strict=True)
    fused = compile_query(binary, MODEL, target, "--ad-wrt", "scale")
    printed = compile_query(binary, derivative, target)
    for query in (fused, printed):
        np.testing.assert_array_equal(query(*args), [expected, expected], strict=True)

    for first, second, right, result in (
        (np.inf, 1, 1, np.inf),
        (-np.inf, 1, 1, -np.inf),
        (np.nan, 1, 1, np.nan),
        (np.inf, -np.inf, 1, np.nan),
        (np.inf, 1, 0, np.nan),
    ):
        a = np.ones_like(a)
        b = np.ones_like(b)
        a[:, 0], a[:, 1], b[0] = first, second, right
        args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(scale))
        expected_nonfinite = np.asarray(result, dtype=np.float32)
        np.testing.assert_array_equal(primal(*args), [expected_nonfinite], strict=True)
        for query in (fused, printed):
            np.testing.assert_array_equal(query(*args), [expected_nonfinite] * 2, strict=True)


if __name__ == "__main__":
    binary = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) == 3 else "cpu"
    jax.config.update("jax_platforms", "cpu" if target == "cpu" else "cuda")
    with tempfile.TemporaryDirectory() as temporary:
        derivative = Path(temporary) / "derivative.flatppl"
        subprocess.run(
            [binary, "autodiff", MODEL, "--wrt", "scale", "-o", derivative], check=True
        )
        check(binary, target, derivative)
