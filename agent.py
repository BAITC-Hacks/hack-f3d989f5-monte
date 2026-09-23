"""Adaptive tariff campaign agent.

Only public environment fields and the supplied historical transition data are used.
The history ranks hypotheses; the pilot observations update their expected lift.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY = Path(__file__).resolve().parent / "data" / "change_tariff.csv"
PILOT_SIZE = 200
PILOT_COUNT = 17
NOISE_PER_CUSTOMER = 0.804
LOG = logging.getLogger(__name__)


class Agent:
    # Allow 15 percentage points of transfer error in the historical push lift.
    # This broad prior lets a 200-person pilot supply most posterior precision;
    # it is an explicit uncertainty assumption, not a fitted population estimate.
    prior_sd = 0.15
    # Subtract one posterior standard deviation before scaling a campaign.
    # This is a risk buffer, not a calibrated guarantee after candidate selection.
    risk_shift = 1.0
    # Charge 20% extra cost during initial allocation for spending scarce budget.
    # Upgrades later compare their incremental gain with the actual remaining cost.
    budget_penalty = 0.20
    # Allow two equal shares of the remaining budget, not half of that budget.
    # The allocation also has a 4000 floor and a hard remaining-budget check.
    soft_cap_factor = 2.0
    # Five cells, at most two candidates each: ordinary selection spends at most
    # 5 * 2 * 200 * 4 = 8000 on SMS exploration, prioritizing high-value audiences.
    sms_pilot_top = 5
    # The weakest estimate in a group must be at least 90% of its strongest.
    # This limits estimated heterogeneity; it does not prove equal true effects.
    merge_ratio_gap = 0.10
    # Positive risk-adjusted lift is sufficient; no additional absolute cutoff.
    merge_min_ratio = 0.0
    # lift / sqrt(n) is the geometric mean of total lift and lift per contact:
    # a symmetric compromise between campaign slots and the contact constraint.
    option_density_weight = 0.50
    enable_upgrades = True
    # Expensive calls must fit initial budget reservations. Extrapolation from
    # push/SMS can overstate call uplift when conversion reaches its upper bound,
    # so spare-budget upgrades stay within the cheaper channels.
    allow_call_exemption = False
    allow_call_upgrade = False

    def act(self, env) -> list[dict]:
        try:
            return self._act(env)
        except Exception:
            LOG.exception("Agent failed; returning a safe fallback campaign")
            try:
                return self._fallback(env)
            except Exception:
                LOG.exception("Fallback campaign also failed")
                return []

    def _act(self, env) -> list[dict]:
        profile = env.customer_profile
        cells = (
            profile.groupby(["current_tariff", "arpu_segment"], observed=True)
            .agg(n=("ID_NUMBER", "size"), arpu=("predicted_arpu", "mean"))
            .reset_index()
        )
        dense_cells = cells[cells.n >= 80].copy()
        cells = dense_cells if not dense_cells.empty else cells[cells.n >= 10].copy()
        if cells.empty:
            return self._fallback(env)

        # Historical transition frequency is a heuristic conversion proxy, not
        # an identified response probability for this audience. Bound extreme
        # relative changes and let pilots correct the uncertain transfer.
        try:
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
            # Small historical groups deserve a weaker prior.
            candidates["prior_push"] *= candidates.samples / (candidates.samples + 8)
            candidates["potential"] = (
                candidates.prior_push.clip(lower=0) * candidates.arpu
                * candidates.n.clip(upper=5000)
            )
            candidates = candidates.sort_values("potential", ascending=False)
        except Exception as exc:
            LOG.warning("History unavailable (%s); choosing pilots from audience cells", exc)
            candidates = self._blind_candidates(env, cells)

        if candidates.empty or not (candidates.potential > 0).any():
            LOG.warning("No positive historical hypotheses; relaxing pilot selection")
            candidates = self._blind_candidates(env, cells)

        # Spread exploration across source cells. Repeating the same cell with
        # many target tariffs spends contacts on the same people.
        chosen = self._select_candidates(candidates, env.pilots_left, 2, 7)
        if not chosen:
            LOG.warning("No pilots selected; relaxing cell and target limits")
            chosen = self._select_candidates(candidates, env.pilots_left, PILOT_COUNT, PILOT_COUNT)
        if not chosen:
            LOG.warning("No pilots selected after relaxation; trying audience-only candidates")
            chosen = self._select_candidates(
                self._blind_candidates(env, cells), env.pilots_left, PILOT_COUNT, PILOT_COUNT
            )
        LOG.info("Selected %d pilot candidates", len(chosen))

        observed = []
        cell_values = sorted(
            {(row.current_tariff, row.arpu_segment): row.n * row.arpu for row in chosen}.items(),
            key=lambda item: item[1], reverse=True,
        )
        sms_cells = {cell for cell, _ in cell_values[:self.sms_pilot_top]}
        for row in chosen:
            if env.pilots_left <= 0 or env.remaining_contacts < 10:
                break
            size = min(PILOT_SIZE, row.n, env.remaining_contacts)
            sms_cost = float(env.channels["sms"]["cost_per_contact"]) * size
            channel = (
                "sms" if (row.current_tariff, row.arpu_segment) in sms_cells
                and sms_cost <= env.remaining_budget else "push"
            )
            try:
                result = env.run_pilot(
                    target_tariff=row.tariff_plan_code_to,
                    channel=channel,
                    n_customers=size,
                    filter_arpu_segment=row.arpu_segment,
                    filter_current_tariff=row.current_tariff,
                )
                n = max(int(result["n_customers"]), 1)
                # Pilot observations have the same absolute noise in every channel.
                # Convert the SMS result to push units before pooling with history.
                multiplier = float(env.channels[channel]["conversion_multiplier"])
                ratio_push = float(result["observed_lift_ratio"]) * 0.50 / multiplier
                prior_variance = self.prior_sd**2
                pilot_variance = (NOISE_PER_CUSTOMER * 0.50 / multiplier)**2 / n
                pilot_weight = prior_variance / (prior_variance + pilot_variance)
                mean = (
                    pilot_weight * ratio_push
                    + (1 - pilot_weight) * float(row.prior_push)
                )
                uncertainty = math.sqrt(
                    prior_variance * pilot_variance / (prior_variance + pilot_variance)
                )
                observed.append((row, mean, uncertainty))
            except Exception:
                LOG.exception("Pilot failed for %s -> %s; continuing", row.current_tariff,
                              row.tariff_plan_code_to)
                continue

        if not observed:
            return self._fallback(env)

        # One final tariff for each disjoint source cell. Compatible cells can
        # later share a campaign without losing their individual lift estimates.
        best_by_cell = {}
        for row, mean, uncertainty in observed:
            cell = (row.current_tariff, row.arpu_segment)
            conservative = max(0.0, mean - self.risk_shift * uncertainty)
            value = conservative * row.arpu * min(row.n, 5000)
            if cell not in best_by_cell or value > best_by_cell[cell][0]:
                best_by_cell[cell] = (value, row, conservative)

        options = self._group_options(best_by_cell)
        campaigns = []
        selected = []
        budget = float(env.remaining_budget)
        contacts = int(env.remaining_contacts)
        channel_order = ["push", "sms", "digital_ads", "call"]
        top_call_candidates = set(
            sorted(range(len(options)), key=lambda i: options[i]["push_lift"] / options[i]["n"],
                   reverse=True)[:2]
        )
        for index, option in enumerate(options):
            if len(campaigns) >= 10 or contacts <= 0:
                break
            n = option["n"]
            if n > contacts:
                # A partial multi-tariff filter would contact an arbitrary ID
                # prefix, so skip it and look for a smaller complete group.
                continue
            # Reserve a portion of budget for remaining cells. The value of
            # an option includes an opportunity charge on scarce budget.
            best = None
            remaining_slots = max(1, min(10 - len(campaigns), len(options) - index))
            soft_cap = max(budget / remaining_slots * self.soft_cap_factor, 4000.0)
            for channel in channel_order:
                info = env.channels[channel]
                cost = float(info["cost_per_contact"])
                multiplier = float(info["conversion_multiplier"])
                # A high-value group may justify a call even when the soft
                # reservation for later campaigns would otherwise exclude it.
                exempt_call = (
                    self.allow_call_exemption and channel == "call"
                    and index in top_call_candidates
                )
                if cost * n > budget or (
                    cost * n > soft_cap and channel != "push" and not exempt_call
                ):
                    continue
                expected = option["push_lift"] * multiplier / 0.50 - n * cost
                utility = expected - self.budget_penalty * cost * n
                if best is None or utility > best[0]:
                    best = (utility, channel, cost, expected)
            if best is None or best[3] <= 0:
                continue
            _, channel, cost, _ = best
            campaigns.append({
                "campaign_name": f"adaptive_{len(campaigns)+1}_{option['target']}",
                "filter_current_tariff": ";".join(option["sources"]),
                "filter_arpu_segment": option["segment"],
                "target_tariff": option["target"],
                "channel": channel,
            })
            selected.append(option)
            budget -= n * cost
            contacts -= n

        # The soft cap only reserves budget while building the plan. Once all
        # groups are chosen, spend the remainder on profitable channel upgrades.
        while budget > 0 and self.enable_upgrades:
            upgrade = None
            for i, option in enumerate(selected):
                old = env.channels[campaigns[i]["channel"]]
                for channel in channel_order:
                    if channel == "call" and not self.allow_call_upgrade:
                        continue
                    new = env.channels[channel]
                    extra = option["n"] * (
                        float(new["cost_per_contact"]) - float(old["cost_per_contact"])
                    )
                    if extra <= 0 or extra > budget:
                        continue
                    gain = (
                        option["push_lift"]
                        * (float(new["conversion_multiplier"])
                           - float(old["conversion_multiplier"])) / 0.50
                        - extra
                    )
                    if gain > 0 and (upgrade is None or gain > upgrade[0]):
                        upgrade = (gain, i, channel, extra)
            if upgrade is None:
                break
            _, i, channel, extra = upgrade
            campaigns[i]["channel"] = channel
            budget -= extra
        return campaigns or self._fallback(env)

    def _group_options(self, best_by_cell):
        buckets = {}
        for _, row, ratio in best_by_cell.values():
            if ratio > 0:
                buckets.setdefault((row.arpu_segment, row.tariff_plan_code_to), []).append(
                    (row, ratio)
                )

        options = []
        for (segment, target), entries in buckets.items():
            entries.sort(key=lambda item: item[1], reverse=True)
            group = []
            group_n = 0
            anchor = None
            for row, ratio in entries:
                n = min(int(row.n), 5000)
                if ratio < self.merge_min_ratio:
                    if group:
                        options.append(self._make_option(segment, target, group))
                        group, group_n, anchor = [], 0, None
                    options.append(self._make_option(segment, target, [(row, ratio, n)]))
                    continue
                if group and (ratio < anchor * (1 - self.merge_ratio_gap)
                              or group_n + n > 5000):
                    options.append(self._make_option(segment, target, group))
                    group, group_n, anchor = [], 0, None
                if not group:
                    anchor = ratio
                group.append((row, ratio, n))
                group_n += n
            if group:
                options.append(self._make_option(segment, target, group))
        return sorted(
            options,
            key=lambda option: option["push_lift"] / option["n"]**self.option_density_weight,
            reverse=True,
        )

    @staticmethod
    def _make_option(segment, target, group):
        n = sum(item[2] for item in group)
        push_lift = sum(item[2] * float(item[0].arpu) * item[1] for item in group)
        return {
            "segment": segment,
            "target": target,
            "sources": [item[0].current_tariff for item in group],
            "n": n,
            # Count-weighted estimate, retained for auditing grouped campaigns.
            "ratio_push": sum(item[2] * item[1] for item in group) / n,
            "push_lift": push_lift,
        }

    @staticmethod
    def _select_candidates(candidates, pilots_left, max_per_cell, max_per_target):
        if pilots_left <= 0:
            return []
        chosen = []
        per_cell = {}
        per_target = {}
        for row in candidates.itertuples(index=False):
            cell = (row.current_tariff, row.arpu_segment)
            target = row.tariff_plan_code_to
            if row.potential <= 0 or per_cell.get(cell, 0) >= max_per_cell:
                continue
            if per_target.get(target, 0) >= max_per_target:
                continue
            chosen.append(row)
            per_cell[cell] = per_cell.get(cell, 0) + 1
            per_target[target] = per_target.get(target, 0) + 1
            if len(chosen) >= min(PILOT_COUNT, pilots_left):
                break
        return chosen

    @staticmethod
    def _blind_candidates(env, cells):
        tariffs = env.tariffs[["tariff_plan_code", "price_tariff"]].sort_values("price_tariff")
        ranked = cells.assign(cell_value=cells.n * cells.arpu).sort_values(
            "cell_value", ascending=False
        )
        rows = []
        for cell in ranked.itertuples(index=False):
            targets = tariffs[tariffs.tariff_plan_code != cell.current_tariff]
            current_price = tariffs.loc[
                tariffs.tariff_plan_code == cell.current_tariff, "price_tariff"
            ]
            if not current_price.empty:
                higher = targets[targets.price_tariff > current_price.iloc[0]]
                if not higher.empty:
                    targets = higher
            for target in targets.head(2).itertuples(index=False):
                rows.append({
                    "tariff_plan_code_from": cell.current_tariff,
                    "arpu_segment": cell.arpu_segment,
                    "tariff_plan_code_to": target.tariff_plan_code,
                    "change": 0.0,
                    "samples": 0,
                    "source_total": 0,
                    "prior_push": 0.0,
                    "current_tariff": cell.current_tariff,
                    "n": cell.n,
                    "arpu": cell.arpu,
                    "potential": 0.01 * cell.n * cell.arpu,
                })
        return pd.DataFrame(rows, columns=[
            "tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to",
            "change", "samples", "source_total", "prior_push", "current_tariff",
            "n", "arpu", "potential",
        ])

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
