"""Channel scene components (auto-registered on import).

Each module's ``@component`` classes register themselves with the core component
registry when imported. The package ``__init__`` imports every module so a single
``from wt_channel import components`` pulls them all in. The plugin exposes four core
components — ``Tx``, ``Rx``, ``RadioMap``, ``PathSolver`` — plus the ``MaterialPreset``
companion for ITU structure materials.
"""
from . import materials  # noqa: F401  (MaterialPreset)
from . import endpoints  # noqa: F401  (ChannelTx / ChannelRx, antenna inline)
from . import radiomap  # noqa: F401  (RadioMap — grid + config + solve + 2D field viz)
from . import pathsolver  # noqa: F401  (PathSolver — path solve + CIR/CFR/taps + ray drawing)
