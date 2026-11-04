"""wt-channel — witwin.channel solver-scene round-trip + radiomap solve + 2D field viz.

Importing this package (eager: no ``activationEvents``):
  1. registers the four core channel components — ``Tx`` / ``Rx`` / ``RadioMap`` /
     ``PathSolver`` (plus the ``MaterialPreset`` companion) — via their ``@component``
     decorators, and
  2. registers :class:`ChannelAdapter` under the ``"channel"`` domain with
     ``register_platform_adapter`` so ``load_platform_scene`` /
     ``export_platform_scene`` dispatch channel scenes here.

No ``witwin.channel`` import happens at module load: the adapter maps lazy-import
it inside their methods, so the plugin still registers its (schema-only) components
on a machine without the channel solver (drjit / CUDA RayD) installed. The shared
base layer (the platform-bridge maps in ``witwin_server.api`` + the unified
``Material`` component) is in core and frozen; this plugin only reads/writes it.
"""
from ._compat import apply_platform_compat

apply_platform_compat()

from witwin_server.api import register_platform_adapter

from . import components  # noqa: F401  (side-effect: registers channel components)
from . import library_items  # noqa: F401  (side-effect: registers Channel library prefabs)
from .adapter import ChannelAdapter

register_platform_adapter("channel", ChannelAdapter())
