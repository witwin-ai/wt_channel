"""Check and print the Channel setup of scenes/main.wtscene.

    python scripts/prepare_channel.py                 # print the setup
    python scripts/prepare_channel.py --tx=-5,3,4     # move the transmitter, save
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import witwin.studio as ws

PROJECT = Path(__file__).resolve().parents[1]


def find(scene: ws.Scene, component: str) -> ws.Entity:
    for entity in scene.objects.values():
        if component in entity.components:
            return entity
    raise SystemExit(f"no entity with a {component} component in the main scene")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tx", help="new transmitter position x,y,z in metres (y is up)")
    args = parser.parse_args(argv)

    project = ws.open_project(PROJECT)
    scene = project.scene()
    tx, rx = find(scene, "ChannelTx"), find(scene, "ChannelRx")
    radiomap, paths = find(scene, "RadioMap"), find(scene, "PathSolver")

    if args.tx:
        tx.transform.position = [float(v) for v in args.tx.split(",")]
        scene.save()
        print(f"saved: Tx at {tx.transform.position}")

    print(f"Project {project.name}, scene {scene.id}")
    print(f"  Tx            {tx.transform.position}, {int(tx.components['ChannelTx']['num_elements'])} elements")
    print(f"  Rx            {rx.transform.position}, {int(rx.components['ChannelRx']['num_elements'])} elements")
    print(f"  Tx-Rx         {math.dist(tx.transform.position, rx.transform.position):.2f} m")
    print(f"  RadioMap      {radiomap.components['RadioMap']['frequency'] / 1e9:.2f} GHz, "
          f"metric {radiomap.components['RadioMap']['metric']}")
    print(f"  Path Solver   {paths.components['PathSolver']['frequency'] / 1e9:.2f} GHz, "
          f"up to {int(paths.components['PathSolver']['max_bounces'])} bounces")
    print("Ready to solve in Studio (RadioMap: Solve, Path Solver: Solve Paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
