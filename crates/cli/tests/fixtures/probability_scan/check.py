"""Run the probability-scan VJP oracle in the StableHLO JAX/Enzyme environment.

Usage: python check.py /absolute/path/to/release/flatppl
"""

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
MODEL = Path(__file__).with_name("masked.flatppl")
LENGTH = 65


def oracle(q, high, low, enabled, seed):
    # With zero observations, Bayes' update is the identity on probability,
    # leaving the affine transition q_next = low + (high - low) * q.
    if not enabled:
        return (2.0, 0.0, np.zeros(LENGTH), 0.0, 0.0, seed)

    slope = high - low
    by_state = 1.0
    by_obs = np.zeros(LENGTH)
    by_high, by_low = 0.0, 0.0
    primal = total_state = total_high = total_low = 0.0
    total_obs = np.zeros(LENGTH)
    for i in range(LENGTH):
        by_state *= slope
        by_obs *= slope
        by_obs[i] = slope * q * (1.0 - q)
        by_high = q + slope * by_high
        by_low = 1.0 - q + slope * by_low
        q = low + slope * q
        primal += q
        total_state += by_state
        total_obs += by_obs
        total_high += by_high
        total_low += by_low
    return (
        primal,
        seed * total_state,
        seed * total_obs,
        seed * total_high,
        seed * total_low,
        0.0,
    )


def run(binary, dtype):
    text = subprocess.run(
        [
            binary,
            "stablehlo",
            str(MODEL),
            "--dtype",
            dtype,
            "--ad-mode",
            "vjp",
            "--ad-wrt",
            "q",
            "--ad-wrt",
            "obs",
            "--ad-wrt",
            "high",
            "--ad-wrt",
            "low",
            "--ad-wrt",
            "unused",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.replace("@logdensity(", "@main(")
    value = jax.jit(lambda *args: hlo_call(*args, source=text))
    typ = np.float32 if dtype == "f32" else np.float64
    seed = -0.75
    cases = [
        (0.0, 0.83, 0.14, False),
        (1.0, 0.83, 0.14, False),
        (0.35, 0.0, 0.0, False),
        (0.35, 0.83, 0.14, True),
    ]
    for q, high, low, enabled in cases:
        inputs = (q, np.zeros(LENGTH), high, low, 2.0, float(enabled), seed)
        actual = value(*(jnp.asarray(x, dtype=typ) for x in inputs))
        expected = oracle(q, high, low, enabled, seed)
        for got, want in zip(actual, expected, strict=True):
            if enabled:
                tol = 5e-6 if dtype == "f32" else 1e-12
                np.testing.assert_allclose(got, want, atol=tol, rtol=tol)
            else:
                np.testing.assert_array_equal(got, want)
        print(f"{dtype} q={q} high={high} low={low} enabled={enabled} seed={seed} passed")


if __name__ == "__main__":
    for precision in ("f32", "f64"):
        run(sys.argv[1], precision)
