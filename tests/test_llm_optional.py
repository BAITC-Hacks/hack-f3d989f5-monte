"""Optional LLM behavior, with a fake key and no external network requests."""

import io
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import agent
from mock_environment import make_mock_env


FAKE_API_KEY = "unit-test-placeholder-not-a-real-key"
HYPOTHESIS = {
    "from_tariff_hint": "tariff_4",
    "to_tariff_hint": "tariff_8",
    "reason": "The larger data bundle may suit subscribers using mobile internet.",
}


def response_bytes(hypotheses=None, *, text=None, status="completed", content=None):
    if text is None:
        text = json.dumps({"hypotheses": [HYPOTHESIS] if hypotheses is None else hypotheses})
    if content is None:
        content = [{"type": "output_text", "text": text}]
    return io.BytesIO(json.dumps({
        "status": status,
        "output": [{"type": "message", "content": content}],
    }).encode("utf-8"))


class OptionalLLMTests(unittest.TestCase):
    def setUp(self):
        self.api = agent.Agent()
        self.env = SimpleNamespace(tariffs=pd.DataFrame({
            "tariff_plan_code": ["tariff_4", "tariff_8", "tariff_9"],
        }))
        self.environ_patch = patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_API_KEY})
        self.environ_patch.start()
        self.addCleanup(self.environ_patch.stop)

        real_read_csv = pd.read_csv

        def read_csv(path, *args, **kwargs):
            if Path(path).name == "tariff_dictionary.csv":
                return pd.DataFrame({
                    "tariff_plan_code": ["tariff_4", "tariff_8", "tariff_9"],
                    "description": ["Voice bundle", "Larger data bundle", "Balanced bundle"],
                })
            if Path(path).name == "feature_dictionary.csv":
                return pd.DataFrame({
                    "feature": ["arpu_segment", "data_segment"],
                    "description": ["Revenue segment", "Mobile internet usage segment"],
                })
            return real_read_csv(path, *args, **kwargs)

        self.csv_patch = patch.object(agent.pd, "read_csv", side_effect=read_csv)
        self.csv_reader = self.csv_patch.start()
        self.addCleanup(self.csv_patch.stop)

    def test_missing_key_performs_no_dictionary_read_or_network_request(self):
        os.environ.pop("OPENAI_API_KEY", None)
        with patch.object(agent, "urlopen") as transport, patch.object(Path, "read_text") as read_text:
            self.assertEqual(self.api._llm_tariff_hints(self.env), set())
        transport.assert_not_called()
        self.csv_reader.assert_not_called()
        read_text.assert_not_called()

    def test_valid_response_uses_single_request_with_bounded_timeout(self):
        with patch.object(agent, "urlopen", return_value=response_bytes()) as transport:
            self.assertEqual(self.api._llm_tariff_hints(self.env), {("tariff_4", "tariff_8")})
        transport.assert_called_once()
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(transport.call_args.kwargs["timeout"], agent.LLM_TIMEOUT_SECONDS)
        payload = json.loads(request.data)
        self.assertIn("input", payload)
        self.assertNotIn("customer_profile", json.dumps(payload))
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertIs(payload["text"]["format"]["strict"], True)

    def test_missing_dictionary_is_silent_and_skips_network(self):
        with patch.object(agent.pd, "read_csv", side_effect=FileNotFoundError("dictionary missing")):
            with patch.object(agent, "urlopen") as transport, self.assertNoLogs(agent.LOG, level="DEBUG"):
                self.assertEqual(self.api._llm_tariff_hints(self.env), set())
        transport.assert_not_called()

    def test_invalid_responses_are_ignored_without_logging(self):
        invalid_responses = [
            {"text": "This is not JSON"},
            {"text": "```json\n{\"hypotheses\": []}\n```"},
            {"text": "[]"},
            {"text": "{}"},
            {"hypotheses": [{**HYPOTHESIS, "to_tariff_hint": "tariff_unknown"}]},
            {"hypotheses": [{**HYPOTHESIS, "to_tariff_hint": "tariff_4"}]},
            {"hypotheses": [HYPOTHESIS, {**HYPOTHESIS, "from_tariff_hint": "unknown"}]},
            {"hypotheses": [{"from_tariff_hint": "tariff_4", "to_tariff_hint": "tariff_8"}]},
            {"status": "incomplete"},
            {"content": [{"type": "refusal", "refusal": "Cannot provide hints"}]},
        ]
        for kwargs in invalid_responses:
            with self.subTest(response=kwargs), patch.object(
                agent, "urlopen", return_value=response_bytes(**kwargs)
            ) as transport, self.assertNoLogs(agent.LOG, level="DEBUG"):
                self.assertEqual(self.api._llm_tariff_hints(self.env), set())
                transport.assert_called_once()

    def test_network_error_is_silent_and_is_not_retried(self):
        with patch.object(agent, "urlopen", side_effect=OSError("simulated connection failure")) as transport:
            with self.assertNoLogs(agent.LOG, level="DEBUG"):
                self.assertEqual(self.api._llm_tariff_hints(self.env), set())
        transport.assert_called_once()

    def test_hard_deadline_does_not_wait_for_stuck_transport(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocked_transport(*args, **kwargs):
            entered.set()
            try:
                release.wait(timeout=5)
                return response_bytes()
            finally:
                finished.set()

        try:
            with patch.object(agent, "LLM_TIMEOUT_SECONDS", 0.02), patch.object(
                agent, "urlopen", side_effect=blocked_transport
            ) as transport:
                started = time.monotonic()
                self.assertEqual(self.api._llm_tariff_hints(self.env), set())
                elapsed = time.monotonic() - started
                self.assertTrue(entered.is_set())
                self.assertFalse(release.is_set())
                self.assertLess(elapsed, 1.0, "The main thread waited for a stuck transport")
                transport.assert_called_once()
        finally:
            release.set()
            self.assertTrue(finished.wait(timeout=1))

    def test_tiebreaker_only_reorders_exact_ties_and_preserves_numeric_values(self):
        candidates = pd.DataFrame({
            "tariff_plan_code_from": ["tariff_9", "tariff_4", "tariff_9", "tariff_4"],
            "tariff_plan_code_to": ["tariff_8", "tariff_8", "tariff_4", "tariff_8"],
            "potential": [100.00001, 100.0, 100.0, 99.0],
            "prior_push": [0.3, 0.2, 0.4, 0.1],
            "n": [30, 40, 50, 60],
        }, index=["highest", "hinted", "unhinted", "lower"])
        candidates = candidates.loc[["highest", "unhinted", "hinted", "lower"]]
        original = candidates.copy(deep=True)
        actual = self.api._apply_llm_tiebreaker(candidates, {("tariff_4", "tariff_8")})
        self.assertEqual(actual.index.tolist(), ["highest", "hinted", "unhinted", "lower"])
        pd.testing.assert_frame_equal(actual.sort_index(), original.sort_index())
        pd.testing.assert_frame_equal(candidates, original)
        pd.testing.assert_frame_equal(self.api._apply_llm_tiebreaker(candidates, set()), original)

    def test_single_llm_call_happens_before_all_real_pilots(self):
        env, _ = make_mock_env(seed=0)
        events = []
        real_pilot = env.run_pilot

        def transport(*args, **kwargs):
            events.append("llm")
            return response_bytes()

        def pilot(*args, **kwargs):
            events.append("pilot")
            return real_pilot(*args, **kwargs)

        with patch.object(agent, "urlopen", side_effect=transport) as request, patch.object(
            env, "run_pilot", side_effect=pilot
        ) as run_pilot:
            campaigns = self.api._act(env)
        self.assertEqual(events[0], "llm")
        self.assertEqual(events.count("llm"), 1)
        request.assert_called_once()
        self.assertGreater(run_pilot.call_count, 0)
        self.assertTrue(1 <= len(campaigns) <= 10)

    def test_failed_llm_preserves_exact_no_key_campaigns(self):
        without_key_env, _ = make_mock_env(seed=1)
        failed_call_env, _ = make_mock_env(seed=1)
        with patch.dict(os.environ):
            os.environ.pop("OPENAI_API_KEY", None)
            without_key = agent.Agent()._act(without_key_env)
        with patch.object(agent, "urlopen", side_effect=OSError("simulated outage")) as request:
            failed_call = agent.Agent()._act(failed_call_env)
        request.assert_called_once()
        self.assertEqual(without_key, failed_call)
        self.assertEqual(without_key_env.pilot_history, failed_call_env.pilot_history)
        self.assertEqual(without_key_env.remaining_budget, failed_call_env.remaining_budget)
        self.assertEqual(without_key_env.remaining_contacts, failed_call_env.remaining_contacts)


if __name__ == "__main__":
    unittest.main()
