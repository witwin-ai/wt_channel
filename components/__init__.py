"""Channel scene components (auto-registered on import).

Each module's ``@component`` classes register themselves with the core component
registry when imported. The package ``__init__`` imports every module so a single
``from wt_channel import components`` pulls them all in.
"""
from . import settings  # noqa: F401  (ChannelSettings)
