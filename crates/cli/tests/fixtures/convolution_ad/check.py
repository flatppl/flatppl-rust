"""Check source AD and Enzyme against the exact derivatives of valid windows.

Run with: python check.py /path/to/flatppl
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call


def compile_query(binary, path, dtype):
    source = subprocess.check_output(
        [binary, "stablehlo", path, "--target", "cpu", "--dtype", dtype], text=True
    ).replace("@logdensity(", "@main(")
    return jax.jit(lambda *args: hlo_call(*args, source=source))


def check(binary, directory, op, width, dtype):
    number = np.float32 if dtype == "f32" else np.float64
    signal = np.array([2, -1, 3, 4, -2], dtype=number)
    kernel = np.array([1, -3, 2, 4, -2][:width], dtype=number)
    ds = np.array([3, -2, 1, 4, -1], dtype=number)
    dk = np.array([-2, 3, 1, -3, 2][:width], dtype=number)
    size = len(signal) - width + 1
    dy = np.arange(1, size + 1, dtype=number) * np.where(np.arange(size) % 2, -1, 1)
    dy = dy.astype(number)
    js = np.zeros((size, len(signal)), dtype=number)
    jk = np.zeros((size, width), dtype=number)
    for i in range(size):
        for j in range(width):
            k = width - 1 - j if op == "conv" else j
            js[i, i + j] = kernel[k]
            jk[i, k] = signal[i + j]

    primal = directory / "primal.flatppl"
    result = directory / "derivative.flatppl"
    prefix = (
        f"s = elementof(cartpow(reals, {len(signal)}))\n"
        f"k = elementof(cartpow(reals, {width}))\ninputs = (s, k)\n"
    )
    expr = f"{op}(s, k)"
    primal.write_text(prefix + f"outputs = {expr}\n")
    args = (jnp.asarray(signal), jnp.asarray(kernel))
    call = compile_query(binary, primal, dtype)
    value, pullback = jax.vjp(lambda s, k: call(s, k)[0], *args)
    np.testing.assert_array_equal(value, js @ signal)
    for got, want in zip(pullback(jnp.asarray(dy)), (js.T @ dy, jk.T @ dy), strict=True):
        np.testing.assert_array_equal(got, want)

    for mode in ("gradient", "jvp", "vjp"):
        if mode == "gradient":
            primal.write_text(prefix + f"outputs = sum({expr})\n")
            seeds = ()
            expected = ((js @ signal).sum(), js.sum(axis=0), jk.sum(axis=0))
        elif mode == "jvp":
            primal.write_text(prefix + f"outputs = {expr}\n")
            seeds = (ds, dk)
            expected = (value, js @ ds + jk @ dk)
        else:
            seeds = (dy,)
            expected = (value, js.T @ dy, jk.T @ dy)
        subprocess.run(
            [binary, "autodiff", primal, "--mode", mode, "--wrt", "s", "--wrt", "k", "-o", result],
            check=True,
        )
        actual = compile_query(binary, result, dtype)(*args, *map(jnp.asarray, seeds))
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)
    with tempfile.TemporaryDirectory() as workspace:
        for dtype in ("f32", "f64"):
            for operation in ("conv", "crosscorr"):
                for width in (1, 3, 5):
                    check(sys.argv[1], Path(workspace), operation, width, dtype)
    print("All 12 convolution and correlation cases passed.")
