"""Adaptive tariff campaign agent.

Only public environment fields and the supplied historical transition data are used.
The history ranks hypotheses; the pilot observations update their expected lift.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY = Path(__file__).resolve().parent / "data" / "change_tariff.csv"
PILOT_SIZE = 200
PILOT_COUNT = 17
NOISE_PER_CUSTOMER = 0.804


class Agent:
    def act(self, env) -> list[dict]:
        profile = env.customer_profile
        cells = (
            profile.groupby(["current_tariff", "arpu_segment"], observed=True)
            .agg(n=("ID_NUMBER", "size"), arpu=("predicted_arpu", "mean"))
            .reset_index()
        )
        cells = cells[cells.n >= 80].copy()
        if cells.empty:
            return self._fallback(env)

        # An empirical prior. The transition frequency is the conversion proxy
        # used in the public scoring rules; the mean ARPU change is clipped in
        # the same way as the published mock model.
        history = pd.read_csv(HISTORY)
        history = history[history.AVG_ARPU_PREV_3M >= 100].copy()
        history["arpu_segment"] = pd.cut(
            history.AVG_ARPU_PREV_3M,
            bins=[-np.inf, 1000, 5000, np.inf],
            labels=["LOW", "MID", "HIGH"],
        ).astype(str)
        history["change"] = (
            (history.AVG_ARPU_NEXT_3M - history.AVG_ARPU_PREV_3M)
            / history.AVG_ARPU_PREV_3M
        ).clip(-1, 3)
        grouped = (
            history.groupby(
                ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"],
                observed=True,
            )
            .agg(samples=("change", "size"), change=("change", "mean"))
            .reset_index()
        )
        grouped["source_total"] = grouped.groupby(
            ["tariff_plan_code_from", "arpu_segment"], observed=True
        ).samples.transform("sum")
        grouped["prior_push"] = (
            grouped.change * grouped.samples / grouped.source_total * 0.50
        )
        candidates = grouped.merge(
            cells,
            left_on=["tariff_plan_code_from", "arpu_segment"],
            right_on=["current_tariff", "arpu_segment"],
            how="inner",
        )
        candidates = candidates[
            candidates.tariff_plan_code_from != candidates.tariff_plan_code_to
        ].copy()
        # Small historical groups are useful hypotheses but deserve a weaker prior.
        candidates["prior_push"] *= candidates.samples / (candidates.samples + 8)
        candidates["potential"] = (
            candidates.prior_push.clip(lower=0) * candidates.arpu
            * candidates.n.clip(upper=5000)
        )
        candidates = candidates.sort_values("potential", ascending=False)

        # Spread exploration across source cells. Repeating the same cell with
        # many target tariffs spends contacts on the same people.
        chosen = []
        per_cell = {}
        per_target = {}
        for row in candidates.itertuples(index=False):
            cell = (row.current_tariff, row.arpu_segment)
            target = row.tariff_plan_code_to
            if per_cell.get(cell, 0) >= 2 or per_target.get(target, 0) >= 7:
                continue
            if row.potential <= 0:
                continue
            chosen.append(row)
            per_cell[cell] = per_cell.get(cell, 0) + 1
            per_target[target] = per_target.get(target, 0) + 1
            if len(chosen) >= min(PILOT_COUNT, env.pilots_left):
                break

        observed = []
        for row in chosen:
            if env.pilots_left <= 0 or env.remaining_contacts < 10:
                break
            size = min(PILOT_SIZE, row.n, env.remaining_contacts)
            try:
                result = env.run_pilot(
                    target_tariff=row.tariff_plan_code_to,
                    channel="push",
                    n_customers=size,
                    filter_arpu_segment=row.arpu_segment,
                    filter_current_tariff=row.current_tariff,
                )
            except (RuntimeError, ValueError):
                continue
            n = max(int(result["n_customers"]), 1)
            # Allow meaningful drift from history. The pilot noise is known
            # from the case description and determines the posterior weight.
            prior_variance = 0.12**2
            pilot_variance = NOISE_PER_CUSTOMER**2 / n
            pilot_weight = prior_variance / (prior_variance + pilot_variance)
            mean = (
                pilot_weight * float(result["observed_lift_ratio"])
                + (1 - pilot_weight) * float(row.prior_push)
            )
            uncertainty = math.sqrt(
                prior_variance * pilot_variance / (prior_variance + pilot_variance)
            )
            observed.append((row, mean, uncertainty))

        if not observed:
            return self._fallback(env)

        # One final tariff for each disjoint source cell. Compare channel gain
        # against channel cost and the budget consumed by other campaigns.
        best_by_cell = {}
        for row, mean, uncertainty in observed:
            cell = (row.current_tariff, row.arpu_segment)
            conservative = max(0.0, mean - 0.35 * uncertainty)
            value = conservative * row.arpu * min(row.n, 5000)
            if cell not in best_by_cell or value > best_by_cell[cell][0]:
                best_by_cell[cell] = (value, row, conservative)

        options = sorted(best_by_cell.values(), key=lambda x: x[0], reverse=True)
        campaigns = []
        budget = float(env.remaining_budget)
        contacts = int(env.remaining_contacts)
        channel_order = ["push", "sms", "digital_ads", "call"]
        for _, row, ratio_push in options:
            if len(campaigns) >= 10 or contacts <= 0 or ratio_push <= 0:
                break
            n = min(int(row.n), 5000, contacts)
            # Reserve a portion of budget for remaining cells. The value of
            # an option includes an opportunity charge on scarce budget.
            best = None
            remaining_slots = max(1, min(10 - len(campaigns), len(options) - len(campaigns)))
            soft_cap = max(budget / remaining_slots * 2.0, 4000.0)
            for channel in channel_order:
                info = env.channels[channel]
                cost = float(info["cost_per_contact"])
                multiplier = float(info["conversion_multiplier"])
                if cost * n > budget or cost * n > soft_cap and channel != "push":
                    continue
                expected = n * (row.arpu * ratio_push * multiplier / 0.50 - cost)
                utility = expected - 0.20 * cost * n
                if best is None or utility > best[0]:
                    best = (utility, channel, cost, expected)
            if best is None or best[3] <= 0:
                continue
            _, channel, cost, _ = best
            campaigns.append({
                "campaign_name": f"adaptive_{len(campaigns)+1}_{row.current_tariff}_{row.tariff_plan_code_to}",
                "filter_current_tariff": row.current_tariff,
                "filter_arpu_segment": row.arpu_segment,
                "target_tariff": row.tariff_plan_code_to,
                "channel": channel,
            })
            budget -= n * cost
            contacts -= n
        return campaigns or self._fallback(env)

    @staticmethod
    def _fallback(env) -> list[dict]:
        profile = env.customer_profile
        cells = profile.groupby(["current_tariff", "arpu_segment"], observed=True).size()
        if cells.empty:
            return []
        current, segment = cells.idxmax()
        target = next((t for t in env.tariffs.tariff_plan_code if t != current), current)
        return [{
            "campaign_name": "fallback_push",
            "filter_current_tariff": current,
            "filter_arpu_segment": segment,
            "target_tariff": target,
            "channel": "push",
        }]
