---
name: radio-map
description: Compute and show a radio map (RF coverage / signal strength heatmap) of a place or of the current Scene with the witwin.channel.run_radiomap tool. Use when the user asks for a radio map, radiomap, coverage map, signal coverage or signal strength, including "show me the UCSD map and run a radio map".
version: 1
scope: extension
---

# Radio maps

The agent tool **`witwin.channel.run_radiomap`** ray-traces a radio map (path gain in
dB) over the geometry already in the Scene with the WiTwin Channel GPU solver and
shows it as a colored plane in the viewport.

## When to use

Any request for a radio map, radiomap, coverage map, wireless / signal coverage or
signal strength, e.g.:

- "Show me the UCSD map and run a radio map"
- "What does the 3.5 GHz coverage look like here?"
- "我要看一下 UCSD 的 map，再跑一个 radio map"

## How to chain it

1. If the Scene has no map or buildings for the place yet, call
   **`witwin.streetmap.import_map`** first. For UCSD use
   `{ "location": "University of California San Diego" }` and omit `radius_km`: the
   importer uses a curated 1 km campus area, so every run imports the same map (and
   reuses its cached download). Pass `radius_km` only when the user states a size.
2. Call **`witwin.channel.run_radiomap`** with `{}`; every argument is optional. The
   grid covers the imported map, the map plane sits 1.5 m above the local ground at
   the map center and one transmitter goes 30 m above the center (clear of rooftops).
   Pass `frequency_ghz`, `tx_height_m`, `size_m`, `resolution` or `samples` only when
   the user asks for them. The default solver is `deterministic` (reflections plus edge
   diffraction); pass `"solver": "montecarlo"` only when the user asks for Monte Carlo.

Do not ask the user to open panels or press Solve; the tool does both.

## Result

The tool returns the transmitter position, the grid, the solve time and path-gain
statistics (`min` / `median` / `max` dB over covered cells and the covered fraction).
`solver` names the solver that ran; a `fallback` entry means the deterministic solve
failed and Monte Carlo produced the map, so mention it.
Report those briefly. Terrain is not flat: cells on hills above the map plane are
inside the ground and read as uncovered.
