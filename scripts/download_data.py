"""Baixa MNIST e/ou CIFAR-10 para data/ (única etapa que usa a internet; os jobs só leem).

uv run python scripts/download_data.py --datasets mnist cifar10
"""

import argparse

import torchvision

DATASETS = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    for name in args.datasets:
        for train in (True, False):
            ds = DATASETS[name](root=args.root, train=train, download=True)
            print(f"{name} train={train}: {len(ds)} amostras, shape {tuple(ds.data.shape[1:])}")


if __name__ == "__main__":
    main()
