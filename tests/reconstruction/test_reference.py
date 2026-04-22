from __future__ import annotations

import json
from pathlib import Path

from spice.reconstruction.reference import load_reference_predictions


def test_load_reference_predictions_dedupes_rows_and_attaches_unique_classes(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "\n".join(
            [
                "chain,seconds,type_model,mae_block,mae_block_rounded,acc_block_rounded,mae_min_fee",
                "ethereum,36,LSTM,0.75,0.75,0.38,2909192",
                "ethereum,36,LSTM,0.75,0.75,0.38,2909192",
            ]
        ),
        encoding="utf-8",
    )
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "outputs": [
                    {
                        "output_type": "stream",
                        "text": [
                            "Number of unique minBlock values in training set: 4\n",
                            "MAE minBlock (raw): 0.75\n",
                            "MAE minBlock (rounded): 0.75\n",
                            "Accuracy minBlock (round): 0.38\n",
                            "MAE minBaseFee: 2909192\n",
                        ],
                    }
                ],
            }
        ]
    }
    (tmp_path / "plot.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    rows = load_reference_predictions(tmp_path)

    assert len(rows) == 1
    assert rows[0].unique_min_block_classes == 4
