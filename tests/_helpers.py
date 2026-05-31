"""Shared round-trip test helpers.

The round trip under test is platform -> studio -> platform: build a
``channel.Scene`` (CPU), run ``adapter.to_studio`` then ``adapter.to_platform``,
and assert the rebuilt scene reproduces the original's declarative state. Geometry is
compared through the frozen ``GeometryMap.to_studio`` (kind + params), so the helper
stays primitive-agnostic.
"""
from typing import Any, Dict, List

import numpy as np
from witwin_server.features.platform.bridge import GeometryMap

ATOL = 1e-4


class RT:
    """Round-trip utilities (platform <-> studio)."""

    @staticmethod
    def back(adapter: Any, scene: Any) -> Any:
        # Full platform -> studio -> platform round trip.
        return adapter.to_platform(adapter.to_studio(scene))

    @staticmethod
    def geom(geometry: Any) -> Dict[str, Any]:
        # Comparable geometry view: kind + numeric params + world position.
        studio = GeometryMap.to_studio(geometry)
        params = {k: RT._floats(v) for k, v in studio.params.items()}
        return {"kind": studio.kind, "params": params, "position": RT._floats(studio.position)}

    @staticmethod
    def assert_geom(actual: Any, expected: Any) -> None:
        # Geometry kind + params + position match within tolerance.
        a, e = RT.geom(actual), RT.geom(expected)
        assert a["kind"] == e["kind"], f"kind {a['kind']} != {e['kind']}"
        assert a["params"].keys() == e["params"].keys(), f"params {a['params']} vs {e['params']}"
        for key, exp in e["params"].items():
            act = a["params"][key]
            if isinstance(exp, str):
                assert act == exp, f"geom param {key}: {act} != {exp}"
            else:
                np.testing.assert_allclose(act, exp, atol=ATOL, err_msg=f"geom param {key}")
        np.testing.assert_allclose(a["position"], e["position"], atol=ATOL, err_msg="position")

    @staticmethod
    def assert_material(actual: Any, expected: Any) -> None:
        # Scalar material (eps_r real / sigma_e / name) matches within tolerance.
        np.testing.assert_allclose(float(actual.eps_r), float(expected.eps_r), atol=ATOL)
        np.testing.assert_allclose(float(actual.sigma_e), float(expected.sigma_e), atol=ATOL)
        assert actual.name == expected.name

    @staticmethod
    def by_name(items: List[Any]) -> Dict[str, Any]:
        # Index scene structures / endpoints by their unique name.
        return {item.name: item for item in items}

    @staticmethod
    def _floats(value: Any) -> Any:
        # Numeric scalar/array -> float(s); strings (e.g. geometry axis) pass through.
        if value is None or isinstance(value, str):
            return value
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return float(value)
