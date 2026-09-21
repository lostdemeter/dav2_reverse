#!/usr/bin/env python3
"""Run the webcam live using the learned (graduated) integer-head model.

This is a thin launcher over `geo_webcam.py --int-head`: it prints the
graduated widths from `adapt/runs/graduation.json`, then starts the
interactive GUI (M colormap / S save / X quit).

Usage:
    python learned_webcam.py                  # live GUI, learned model
    python learned_webcam.py --frames 5       # headless check, 5 frames
    python learned_webcam.py --cpu            # force CPU-only
Any extra args are forwarded to geo_webcam.py (see --help there).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    cfg_path = HERE / 'adapt' / 'runs' / 'graduation.json'
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))['incumbent']['config']
        print(f"learned model: {cfg}")
    else:
        print(f"no {cfg_path} — geo_webcam will use fallback widths")
    sys.argv = ['geo_webcam.py', '--int-head'] + sys.argv[1:]
    import geo_webcam
    geo_webcam.main()


if __name__ == '__main__':
    main()
