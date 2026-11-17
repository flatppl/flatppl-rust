"""Execute the pinned special-function oracles using JAX/Enzyme.

Run in the existing testsuite StableHLO environment:
    python check.py /absolute/path/to/release/flatppl [cpu|gpu]
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)
fixtures = json.loads(Path(__file__).with_name("oracle.json").read_text())["functions"]
for dtype in ("f32", "f64"):
    typ = np.float32 if dtype == "f32" else np.float64
    rtol, atol = (5e-3, 2e-5) if dtype == "f32" else (5e-8, 3e-10)
    for row in fixtures:
        name = row["name"]
        names = [f"a{i}" for i in range(len(row["point"]))]
        expression = f"sf.{name}({', '.join(names)})"
        if name == "bessel_i_scaled":
            expression = f"real({expression})"
        source = 'sf=standard_module("special-functions", "0.1")\n'
        source += "\n".join(f"{a}=external(reals)" for a in names)
        source += f"\ninputs=({', '.join(names)})\noutputs={expression}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".flatppl") as model:
            model.write(source)
            model.flush()
            result = subprocess.run(
                [
                    sys.argv[1],
                    "stablehlo",
                    model.name,
                    "--mode",
                    "logdensity",
                    "--dtype",
                    dtype,
                    "--target",
                    sys.argv[2] if len(sys.argv) > 2 else "cpu",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        module = re.sub(r"@logdensity\b", "@main", result.stdout)

        def call(*args, source=module):
            return hlo_call(*args, source=source)[0]

        value = jax.jit(call)
        for case in row["cases"]:
            got = value(*[jnp.asarray(a, dtype=typ) for a in case["args"]])
            np.testing.assert_allclose(
                got, case["value"], rtol=rtol, atol=np.finfo(typ).tiny
            )
        for case in row.get("failure_cases", []):
            args = [jnp.asarray(a, dtype=typ) for a in case["args"]]
            raw = jax.jit(lambda *args, source=module: hlo_call(*args, source=source))(
                *args
            )
            status = re.search(r"flatppl.runtime_status\s*=\s*(\d+)", module)
            assert status is not None
            assert int(raw[int(status[1])]) == case["status"]
            assert np.isnan(raw[0])
        point = [jnp.asarray(a, dtype=typ) for a in row["point"]]
        gradient = jax.jit(jax.grad(call, argnums=tuple(range(len(point)))))
        gradient_cases = [{"args": row["point"], "gradient": row["gradient"]}]
        gradient_cases += row.get("gradient_cases", [])
        for case in gradient_cases:
            args = [jnp.asarray(a, dtype=typ) for a in case["args"]]
            np.testing.assert_allclose(
                gradient(*args), case["gradient"], rtol=rtol, atol=atol
            )
        batch = [jnp.stack([a, a]) for a in point]
        np.testing.assert_allclose(
            jax.jit(jax.vmap(call))(*batch), value(*point), rtol=rtol, atol=atol
        )
        print(name, dtype, "values, all-parameter AD and vmap passed", flush=True)
