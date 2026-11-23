"""Check the score and derivative of an exponentially tilted square.

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


def oracle(theta):
    if theta == 0:
        return 0.0, 0.0
    # The partition function is [exp(-theta/2) * expm1(theta) / theta]^2.
    score = 2 * math.log(math.expm1(theta) / theta) - theta
    derivative = 2 * (1 / -math.expm1(-theta) - 1 / theta) - 1
    return score, derivative


def run(binary, derivative, dtype):
    flags = ["--numerical-integrals", "--integration-max-intervals", "128", "--dtype", dtype]
    if dtype == "f64":
        flags.extend(["--integration-rtol", "1e-10", "--integration-atol", "1e-14"])
    for query, ad in [(derivative, []), (MODEL, ["--ad-wrt", "theta"])]:
        source = subprocess.run(
            [binary, "stablehlo", str(query), *flags, *ad],
            check=True, capture_output=True, text=True,
        ).stdout.replace("@logdensity(", "@main(")
        score_and_grad = jax.jit(lambda theta: hlo_call(theta, source=source))
        scalar = np.float64 if dtype == "f64" else np.float32
        tolerance = 2e-10 if dtype == "f64" else 2e-6
        for theta in [-0.9, 0.0, 0.1, 1.2]:
            value = score_and_grad(jnp.asarray(theta, dtype=scalar))
            np.testing.assert_allclose(value, oracle(theta), atol=tolerance, rtol=0)
            print(f"{query.stem} {dtype} {theta} passed")


if __name__ == "__main__":
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        derivative = Path(tmp) / "derivative.flatppl"
        subprocess.run(
            [binary, "autodiff", str(MODEL), "--numerical-integrals", "--wrt", "theta",
             "-o", str(derivative)],
            check=True,
        )
        for dtype in ["f64", "f32"]:
            run(binary, derivative, dtype)
