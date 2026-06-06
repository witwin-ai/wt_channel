"""Small shared conversion helpers for the channel adapter maps."""
from typing import Any, List, Optional, Sequence

TOL = 1e-6


def num(value: Any) -> float:
    """Component scalar (tensor or number) -> python float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def vec(value: Any) -> List[float]:
    """Component vector (tensor or list) -> python floats."""
    seq = value.tolist() if hasattr(value, "tolist") else value
    return [float(v) for v in seq]


def json_value(value: Any) -> Any:
    """Component/runtime value -> JSON-serializable Python value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        return json_value(value.item())
    return value


def device_name(raw: Any) -> str:
    """Normalize a torch/string device ('cuda:0', device(...)) -> 'cuda' | 'cpu'."""
    return "cuda" if str(raw).startswith("cuda") else "cpu"


def opt_vec3(value: Optional[Sequence[float]]) -> Optional[tuple]:
    """Component vector3 -> a 3-tuple of floats, or None when unset/empty."""
    if value is None:
        return None
    seq = value.tolist() if hasattr(value, "tolist") else value
    return (float(seq[0]), float(seq[1]), float(seq[2]))
