# Channel ML

A WiTwin Studio Project that synthesizes radio channel data, trains a model on
it and evaluates the model. It is also an ordinary Python project: you can
open, edit and run it without the editor.

The task: from the receiver's position on a street block, predict the total
path gain from the transmitter in dB (a learned coverage map). The Scene is a
3.5 GHz transmitter `Tx` at 3 m, a receiver `Rx` at 1.5 m, three buildings and
the ground, each with an ITU material. Two Scene parameters place the
receiver: `rx_x` and `rx_z` (metres).

## Set up

```bash
uv sync
```

This installs the Channel Plugin with its solver stack (`witwin-channel`, CUDA
torch). Synthesizing data needs a CUDA GPU; training, evaluating and the tests
run on the CPU.

## The pipeline

| Step | File | What it does |
| --- | --- | --- |
| Synthesize | `studies/link-gain.toml` (dataset Study) | Monte Carlo draws 800 receiver positions. For each, `scripts/synthesize.py` traces the paths from Tx to Rx (reflections and diffraction, as the `Path Solver` is set) and stores the position, the total gain in dB, the RMS delay spread and the path count. Positions inside a building are dropped. Writes `datasets/link-gain/`, split 80/10/10 into train, val and test. |
| Train | `studies/train-gain.toml` (train Study) | `scripts/train.py` fits a small neural network (`scripts/model.py`, numpy) and logs `train/loss`, `val/loss` and `val/mae_db`. Saves the best epoch to `checkpoints/<run id>.npz`. |
| Evaluate | `scripts/evaluate.py` | Scores the checkpoint with the lowest validation error on the test split against a predict-the-mean baseline. Writes `reports/evaluation.json`. |

`tasks/channel-pipeline.toml` is a Workflow that runs the three steps in
order. Its `train` step trains three learning rates (`matrix`), and `evaluate`
picks the best of them.

## Run it

In Studio: open **Tasks**, pick **Synthesize, train, evaluate** and run it.
Or run each Study from the **Studies** page; the train Run shows its curves
there.

Without Studio:

```bash
uv run witwin workflow run channel-pipeline
```

or step by step:

```bash
uv run python -m witwin.studio.studies --project . --study link-gain --out .witwin/runs/synthesize
uv run python scripts/train.py --dataset datasets/link-gain
uv run python scripts/evaluate.py --dataset datasets/link-gain --split test
uv run pytest
```

## Change it

- More or different data: edit the variables and `samples` in
  `studies/link-gain.toml`, or bind more Scene fields (the Tx position, a
  building's position) to parameters and vary those.
- Another label: `delay_spread_ns` and `num_paths` are in the dataset already;
  set `TARGET` in `scripts/model.py`.
- Another model: replace `scripts/model.py`. For torch,
  `witwin.studio.datasets.open("datasets/link-gain").split("train").torch()`
  returns the samples as a torch `Dataset`.

## Notes

- `scripts/synthesize.py` calls `witwin.channel` directly, the library the
  Channel Plugin's Path Solver uses, with the frequency and bounce settings of
  the Scene's `Path Solver` component. It reads boxes (`PlatformGeometry`
  kind `box`, not rotated) and the ITU presets in its own table. The first
  sample compiles CUDA kernels and takes about 15 s.
- Dataset shards are stored with Git LFS (`datasets/**/shards/**`). If this
  folder is a git repository, install git-lfs before running the dataset Study.
- `.witwin/` is local runtime state (caches, Runs, Agent Sessions); git ignores
  it, and `checkpoints/`.
