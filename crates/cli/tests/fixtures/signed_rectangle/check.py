"""Check numerical scores and derivatives of tilted rectangles.

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


def normalization_oracle(theta, shift, amplitude, variation, hi, *, rectangle=True):
    if hi == 0:
        return shift, 0.0, 1.0, 0.0
    slope = theta * amplitude * variation

    def mean(width):
        if abs(slope * width) <= 1e-3:
            return width / 2 + slope * width**2 / 12 - slope**3 * width**4 / 720
        return width / -math.expm1(-slope * width) - 1 / slope

    def mass(width):
        return math.expm1(slope * width) / slope if slope else width

    mass_y = mass(1.0) if rectangle else 1.0
    mean_y = mean(1.0) if rectangle else 0.0
    score = shift + theta**2 + theta * amplitude + math.log(mass(hi) * mass_y)
    d_theta = 2 * theta + amplitude * (1 + variation * (mean(hi) + mean_y))
    d_hi = slope / -math.expm1(-slope * hi) if slope else 1 / hi
    return score, d_theta, 1.0, d_hi


def check_normalization(binary, derivative, dtype, *, rectangle):
    model = MODEL.with_name("normalization.flatppl" if rectangle else "normalization_1d.flatppl")
    coordinates = ("theta", "shift", "hi")
    wrt = [flag for name in coordinates for flag in ("--wrt", name)]
    subprocess.run(
        [binary, "autodiff", str(model), "--numerical-integrals", *wrt, "-o", str(derivative)],
        check=True,
    )
    flags = ["--numerical-integrals", "--dtype", dtype, "--integration-max-intervals", "128"]
    if dtype == "f64":
        flags.extend(["--integration-rtol", "1e-10", "--integration-atol", "1e-14"])
    else:
        flags.extend(["--integration-rtol", "1e-5", "--integration-atol", "0"])
    scalar = np.float64 if dtype == "f64" else np.float32
    tolerance = 2e-9 if dtype == "f64" else 6e-6
    points = [
        (0.0, 0.0, 1.0, 1e-6, 2.0),
        (0.0, -1e8, 1.0, 1e-6, 2.0),
        (0.0, -1e8, 2e38, 0.0, 2.0),
        (0.0, -1e4, -1e-20, -0.9, 2.0),
        (0.7, -1e4, 1.0, 0.7, 2.0),
        (0.7, -1e8, 1.0, 0.7, 2.0),
        (0.7, 1e4, 1.0, 0.7, 2.0),
        (0.0, -1e8, 0.0, 0.7, 2.0),
        (0.0, -1e4, 1.0, 0.0, 0.0),
    ]
    fused = [flag for name in coordinates for flag in ("--ad-wrt", name)]
    for query, ad in [(derivative, []), (model, fused)]:
        program = subprocess.run(
            [binary, "stablehlo", str(query), *flags, *ad],
            check=True, capture_output=True, text=True,
        ).stdout.replace("@logdensity(", "@main(")
        invoke = jax.jit(lambda *args: hlo_call(*args, source=program))
        for point in points:
            expected = normalization_oracle(*(float(scalar(x)) for x in point), rectangle=rectangle)
            actual = invoke(*(jnp.asarray(x, dtype=scalar) for x in point))
            score_tolerance = max(2 * abs(np.spacing(scalar(expected[0]))), tolerance)
            np.testing.assert_allclose(actual[0], scalar(expected[0]), atol=score_tolerance, rtol=0)
            np.testing.assert_allclose(actual[1:], expected[1:], atol=0, rtol=tolerance)
            print(f"{query.stem} normalization {dtype} {point} passed")


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
            for rectangle in [False, True]:
                check_normalization(binary, Path(tmp) / "normalized.flatppl", dtype, rectangle=rectangle)
