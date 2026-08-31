from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from market_state_lab.models import PROBABILITY_COLUMNS, STATE_NAMES


def synthetic_regime_metrics(
    probabilities: pd.DataFrame,
    true_state: pd.Series,
    maximum_delay: int = 30,
) -> dict[str, Any]:
    aligned = pd.concat([probabilities[PROBABILITY_COLUMNS], true_state.rename("truth")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No aligned synthetic labels and probabilities")
    truth = aligned["truth"].astype(str)
    prediction = aligned[PROBABILITY_COLUMNS].idxmax(axis=1).str.replace("p_", "", regex=False)
    target = pd.DataFrame(
        {f"p_{name}": truth.eq(name).astype(float) for name in STATE_NAMES},
        index=aligned.index,
    )
    change_positions = np.flatnonzero(truth.ne(truth.shift()).to_numpy())[1:]
    delays: list[int] = []
    for position in change_positions:
        expected = truth.iloc[position]
        future = prediction.iloc[position : position + maximum_delay + 1]
        matches = np.flatnonzero(future.eq(expected).to_numpy())
        delays.append(int(matches[0]) if len(matches) else maximum_delay + 1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "brier": float(((aligned[PROBABILITY_COLUMNS] - target) ** 2).sum(axis=1).mean()),
        "mean_transition_delay_days": float(np.mean(delays)) if delays else np.nan,
        "observations": len(aligned),
    }
