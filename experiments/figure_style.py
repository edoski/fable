"""Shared thesis figure styling and deterministic PDF export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).parents[1] / "outputs" / "figures"

_FAMILY_STYLES = {
    "lstm": ("#0072B2", "o"),
    "transformer": ("#D55E00", "s"),
    "transformer_lstm": ("#009E73", "^"),
}
_DISPLAY_NAMES = {
    "ethereum": "Ethereum",
    "polygon": "Polygon",
    "avalanche": "Avalanche",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "transformer_lstm": "Transformer–LSTM",
}
_RC = mpl.RcParams(
    {
        "axes.axisbelow": True,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlesize": 9,
        "figure.dpi": 144,
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)
mpl.rcParams.update(_RC)


def subplots(rows: int, columns: int, *, height: float) -> tuple[Figure, Any]:
    return plt.subplots(
        rows, columns, figsize=(6.4, height), squeeze=False, constrained_layout=True
    )


def family_style(family: str) -> tuple[str, str]:
    try:
        return _FAMILY_STYLES[family]
    except KeyError as error:
        raise ValueError(f"unsupported model family: {family}") from error


def display_name(value: str) -> str:
    return _DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def add_family_legend(figure: Figure, axis: Axes) -> None:
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside upper center", ncols=max(1, len(labels)), frameon=False
    )


def save_pdf(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="pdf",
        metadata={
            "Creator": "FABLE",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return path
