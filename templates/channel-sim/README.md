# Channel Sim

A WiTwin Studio Project. It is also an ordinary Python project: you can open,
edit and run it without the editor.

## Set up

```bash
uv sync                 # or: python -m venv .venv, activate it, pip install -e .
```

## Use from Python

```python
import witwin.studio as ws

project = ws.open_project(".")
scene = project.scene()          # the main scene
print(list(scene.objects))
```

## Commands

- `witwin check` validates the Project files.
- `witwin run scripts/<name>.py` runs a Script with the Project on its path.
- `witwin open` opens the Project in WiTwin Studio.

## Layout

- `witwin.toml`: the Studio manifest (Project id, main scene).
- `pyproject.toml`: the Python dependencies, including the Plugins this Project uses.
- `scenes/`: the Scene files (`.wtscene`, TOML). Studio and code edit the same
  files; saving from either side rewrites the file, so comments in it are not kept.
- `.witwin/`: local runtime state (caches, Agent Sessions, Result history). It
  is not shared and git ignores it.

## What is in the Scene

`scenes/main.wtscene` is a small street block: a concrete `Channel Ground`,
three buildings (`Building North` and `Building East` in concrete, `Building
South` in brick, each with an ITU `MaterialPreset`), a transmitter `Tx` at 3 m
and a receiver `Rx` at 1.5 m, 13 m apart. `RadioMap` solves a 2D coverage map
over the ground and `Path Solver` traces the rays between Tx and Rx, which
reflect off the buildings. Both solvers run at 3.5 GHz.

## Solve

Open the Project in WiTwin Studio, select **RadioMap** and press **Solve**, or
select **Path Solver** and press **Solve Paths**. Solving needs the Channel
solver runtime and a CUDA GPU.

## Run the example

```bash
uv sync
uv run python scripts/prepare_channel.py
uv run pytest
```

`scripts/prepare_channel.py` opens this Project with `witwin.studio`, checks the setup
the solver needs and prints it. Pass an option to change the setup; the script
saves `scenes/main.wtscene`, the same file Studio edits.
