"""Local robustness checks; this file is not part of the submitted agent.

Run ``python stress_eval.py --runs 20`` for the reference and proposed policy.
Add ``--sensitivity`` for one-at-a-time changes to the SMS and merge settings.
The agent never receives the impact table. Both pilots and the official local
scorer use the same temporary scenario, restored after each context. Model
randomness is fixed independently of the pilot seed. These deliberately chosen
scenarios check distribution shifts; they do not predict the judging score.

Only the downshift scenario changes the published fallback for absent historical
transitions. Other scenarios retain its formula (with the scenario's median
conversion, as supplied by the public environment and scorer).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json

import numpy as np
import pandas as pd
from unittest.mock import patch

from agent import Agent
import environment
import local_eval
import mock_environment
from scoring_core import validate_strategy


MODEL_SEED = 20260923
SCENARIOS = ("ordinary", "downshift", "flip25", "heterogeneous", "high_noise")
BASELINE = {
    "prior_sd": 0.08,
    "risk_shift": 0.35,
    "budget_penalty": 0.20,
    "soft_cap_factor": 2.0,
    "merge_ratio_gap": 0.08,
    "merge_min_ratio": 0.0,
    "option_density_weight": 0.54,
    "sms_pilot_top": 5,
}
PROPOSED = {
    **BASELINE,
    "prior_sd": 0.15,
    "risk_shift": 1.0,
    "merge_ratio_gap": 0.10,
    "option_density_weight": 0.50,
}
POLICIES = {
    "baseline": BASELINE,
    "proposed": PROPOSED,
    "sms3": {**PROPOSED, "sms_pilot_top": 3},
    "sms7": {**PROPOSED, "sms_pilot_top": 7},
    "merge05": {**PROPOSED, "merge_ratio_gap": 0.05},
    "merge08": {**PROPOSED, "merge_ratio_gap": 0.08},
    "merge20": {**PROPOSED, "merge_ratio_gap": 0.20},
}


@contextmanager
def scenario_context(name: str, model_seed: int = MODEL_SEED):
    """Patch both imported bindings, never the agent or organizer files."""
    original_model = mock_environment._mock_impact_model
    original_fallback = mock_environment._mock_fallback

    def alternative_model(history):
        model = original_model(history).sort_values(
            ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]
        ).reset_index(drop=True)
        rng = np.random.default_rng(model_seed)
        if name == "downshift":
            model["arpu_change_pct"] = (0.7 * model.arpu_change_pct - 0.2).clip(-1, 3)
        elif name == "flip25":
            flipped = rng.choice(len(model), size=len(model) // 4, replace=False)
            model.loc[flipped, "arpu_change_pct"] *= -1
            model["arpu_change_pct"] = model.arpu_change_pct.clip(-1, 3)
        elif name == "heterogeneous":
            model["arpu_change_pct"] = (
                0.6 * model.arpu_change_pct + rng.normal(0, 0.35, len(model))
            ).clip(-1, 3)
            model["conversion_rate"] = (
                model.conversion_rate * rng.uniform(0.6, 1.4, len(model))
            ).clip(0, 1)
        return model

    def alternative_fallback(*args, **kwargs):
        change, conversion = original_fallback(*args, **kwargs)
        if name == "downshift":
            change = float(np.clip(0.7 * change - 0.2, -1, 3))
        return change, conversion

    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    with ExitStack() as stack:
        for module in (mock_environment, local_eval):
            stack.enter_context(patch.object(module, "_mock_impact_model", alternative_model))
            stack.enter_context(patch.object(module, "_mock_fallback", alternative_fallback))
        if name == "high_noise":
            stack.enter_context(patch.object(
                environment, "PER_CUSTOMER_STD", 1.5 * environment.PER_CUSTOMER_STD
            ))
        yield


class AuditedAgent:
    """Capture only public fields and the returned plan for limit checks."""

    def __init__(self, settings):
        self.agent = Agent()
        for name, value in settings.items():
            setattr(self.agent, name, value)
        self.env = None
        self.campaigns = None

    def act(self, env):
        self.env = env
        self.campaigns = self.agent.act(env)
        return self.campaigns

    def check(self, result):
        if result is None or self.campaigns is None:
            raise AssertionError("Agent returned no evaluable result")
        assert isinstance(self.campaigns, list), "Plan must be a list"
        assert 1 <= len(self.campaigns) <= 10, "Final campaign count outside 1..10"
        validate_strategy(pd.DataFrame(self.campaigns), self.env.tariffs)
        pilots = self.env.pilot_history
        assert 0 < len(pilots) <= 20, "Pilot count outside 1..20"
        assert all(10 <= item["n_customers"] <= 200 for item in pilots), "Invalid pilot size"
        assert self.env.remaining_budget >= 0, "Pilot budget exceeded"
        assert self.env.remaining_contacts >= 0, "Pilot contacts exceeded"
        assert result["n_pilots"] == len(pilots)
        assert result["n_campaigns"] == len(pilots) + len(self.campaigns), "Rejected campaigns"
        assert 0 <= result["total_cost"] <= 100_000, "Total budget exceeded"
        assert 0 <= result["total_contacts"] <= 15_000, "Total contacts exceeded"
        assert all(0 <= item["n_contacts"] <= 5000
                   for item in result["campaigns_detail"]), "Campaign contact limit exceeded"
        assert np.isfinite(result["net_arpu_gain"]), "Nonfinite score"


def evaluate_policy(policy, scenario, runs, model_seed):
    records = []
    with scenario_context(scenario, model_seed):
        for seed in range(runs):
            audited = AuditedAgent(POLICIES[policy])
            result = local_eval.evaluate_agent(audited, seed=seed, verbose=False)
            audited.check(result)
            record = {key: result[key] for key in (
                "net_arpu_gain", "total_cost", "total_contacts", "coverage_pct",
                "risk_score_pct", "budget_used_pct", "n_pilots",
            )}
            record.update(
                seed=seed,
                final_campaigns=len(audited.campaigns),
                capped_campaigns=sum(any(item[flag] for flag in (
                    "capped_at_campaign_limit", "capped_at_reach_budget", "capped_at_money_budget"
                )) for item in result["campaigns_detail"]),
            )
            records.append(record)
    frame = pd.DataFrame(records)
    net = frame.net_arpu_gain
    summary = {
        "policy": policy,
        "scenario": scenario,
        "runs": runs,
        "model_seed": model_seed,
        "median": float(net.median()),
        "min": float(net.min()),
        "max": float(net.max()),
        "mean": float(net.mean()),
        "std": float(net.std(ddof=0)),
        "positive": int((net > 0).sum()),
        "median_budget_pct": float(frame.budget_used_pct.median()),
        "median_coverage_pct": float(frame.coverage_pct.median()),
        "median_risk_pct": float(frame.risk_score_pct.median()),
        "max_risk_pct": float(frame.risk_score_pct.max()),
        "capped_campaigns": int(frame.capped_campaigns.sum()),
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES))
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--json", action="store_true", help="Also print all per-seed records as JSON")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    policies = args.policies or (list(POLICIES) if args.sensitivity else ["baseline", "proposed"])
    print(f"Fixed model seed: {args.model_seed}; pilot seeds: 0..{args.runs - 1}", flush=True)
    print("Limits checked on every run. Budget/coverage/risk columns are medians.", flush=True)
    print("policy     scenario          median        min       mean        std positive  budget coverage risk  maxrisk caps", flush=True)
    output = []
    for scenario in args.scenarios:
        for policy in policies:
            summary, records = evaluate_policy(policy, scenario, args.runs, args.model_seed)
            print(
                f"{policy:10} {scenario:14} {summary['median']:10,.0f} {summary['min']:10,.0f} "
                f"{summary['mean']:10,.0f} {summary['std']:10,.0f} "
                f"{summary['positive']:2}/{args.runs:<2} "
                f"{summary['median_budget_pct']:7.2f} {summary['median_coverage_pct']:7.2f} "
                f"{summary['median_risk_pct']:5.2f} {summary['max_risk_pct']:7.2f} "
                f"{summary['capped_campaigns']:4}", flush=True,
            )
            output.append({"summary": summary, "settings": POLICIES[policy], "runs": records})
    if args.json:
        print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
