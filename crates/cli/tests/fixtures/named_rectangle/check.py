"""Check a named rectangle against its analytic partition function and gradient.

Usage: python check.py /absolute/path/to/release/flatppl
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
MODEL = Path(__file__).with_name("model.flatppl")
COORDINATES = ("a", "hi", "xx", "yy")


def oracle(a, hi, x, y):
    mass_x = math.expm1(a * hi) / a
    mass_y = (math.exp(1.0) - math.exp(-2.0)) / 2
    gradient = (x - hi / -math.expm1(-a * hi) + 1 / a,
                -a / -math.expm1(-a * hi), a, 2.0)
    return (a*x + 2*y - math.log(mass_x * mass_y), *gradient)


def run(binary, model, derivative, dtype):
    wrt = [argument for name in COORDINATES for argument in ("--wrt", name)]
    subprocess.run(
        [binary, "autodiff", str(model), "--numerical-integrals", *wrt, "-o", str(derivative)],
        check=True,
    )
    options = ["--integration-rtol", "1e-10", "--integration-atol", "1e-14"] if dtype == "f64" else []
    source = subprocess.run(
        [binary, "stablehlo", str(derivative), "--numerical-integrals", "--dtype", dtype, *options],
        check=True, capture_output=True, text=True,
    ).stdout.replace("@logdensity(", "@main(")
    score = jax.jit(lambda *args: hlo_call(*args, source=source))
    scalar = np.float64 if dtype == "f64" else np.float32
    tolerance = 3e-10 if dtype == "f64" else 4e-5
    for inputs in [(0.4, 2.3, 0.6, -0.2), (0.8, 1.5, 1.2, -0.6)]:
        actual = score(*(jnp.asarray(value, dtype=scalar) for value in inputs))
        np.testing.assert_allclose(actual, oracle(*inputs), atol=tolerance, rtol=tolerance)
        print(f"{model.stem} {dtype} {inputs} passed")
    for inputs in [(0.4, 2.3, -0.1, -0.2), (0.4, 2.3, 0.6, 0.6)]:
        value, *_ = score(*(jnp.asarray(value, dtype=scalar) for value in inputs))
        assert np.isneginf(value), (inputs, value)


if __name__ == "__main__":
    text = MODEL.read_text()
    swapped = text.replace("y = interval(-1.0, 0.5), x = interval(0.0, hi)",
                           "x = interval(0.0, hi), y = interval(-1.0, 0.5)")
    assert swapped != text
    with tempfile.TemporaryDirectory() as temp:
        scratch = Path(temp)
        reverse = scratch / "reordered.flatppl"
        reverse.write_text(swapped)
        for model in (MODEL, reverse):
            for dtype in ("f64", "f32"):
                run(sys.argv[1], model, scratch / "derivative.flatppl", dtype)
