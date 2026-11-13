"""Tempo por execução, lido dos done.json de runs/<exp>/.

uv run python scripts/run_times.py --exp khipu
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="khipu")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    rows = []
    for d in sorted((Path(args.runs) / args.exp).glob("*/done.json")):
        j = json.loads(d.read_text())
        rows.append((d.parent.name, j["step"], j["wall_s"]))
    if not rows:
        raise SystemExit(f"nenhuma execução concluída em {Path(args.runs) / args.exp}")
    print(f"{'execução':45s} {'passos':>7s} {'tempo':>9s} {'ms/passo':>9s}")
    for name, step, wall in rows:
        print(f"{name:45s} {step:7d} {wall / 60:7.1f} m {1000 * wall / step:9.1f}")
    print(f"{'total':45s} {'':7s} {sum(r[2] for r in rows) / 60:7.1f} m")


if __name__ == "__main__":
    main()
