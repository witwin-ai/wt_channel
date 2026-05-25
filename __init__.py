"""wt-channel — witwin.channel solver-scene round-trip + radiomap solve + 2D field viz.

Importing this package (eager: no ``activationEvents``):
  1. registers the channel scene components (settings / materials / endpoints /
     result) via their ``@component`` decorators, and
  2. registers :class:`ChannelAdapter` under the ``"channel"`` domain on the core
     :class:`PlatformSceneHandler` so ``load_platform_scene`` /
     ``export_platform_scene`` dispatch channel scenes here.

No ``witwin.channel`` import happens at module load: the adapter maps lazy-import
it inside their methods, so the plugin still registers its (schema-only) components
on a machine without the channel solver (drjit / CUDA RayD) installed. The shared
base layer (``witwin_server.platform_bridge`` + the unified ``EMMaterial``
component) is in core and frozen; this plugin only reads/writes it.
"""
from witwin_server.handlers.platform_scene import PlatformSceneHandler

from . import components  # noqa: F401  (side-effect: registers channel components)
from . import library_items  # noqa: F401  (side-effect: registers Channel library prefabs)
from .adapter import ChannelAdapter

PlatformSceneHandler.register_adapter("channel", ChannelAdapter())
