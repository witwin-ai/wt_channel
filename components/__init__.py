"""Channel scene components (auto-registered on import).

Each module's ``@component`` classes register themselves with the core component
registry when imported. The package ``__init__`` imports every module so a single
``from wt_channel import components`` pulls them all in.
"""
from . import settings  # noqa: F401  (ChannelSettings)
from . import materials  # noqa: F401  (MaterialPreset)
from . import antenna  # noqa: F401  (AntennaArraySpec, ChannelDefaultArray)
from . import endpoints  # noqa: F401  (Transmitter, Receiver, ReceiverGrid)
from . import result  # noqa: F401  (ChannelResult — solve + radiomap viz)
from . import path  # noqa: F401  (PathSolve — path solve + CIR/CFR/taps + ray drawing)
