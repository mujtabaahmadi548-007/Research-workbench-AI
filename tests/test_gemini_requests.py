"""Gemini request rules, fallback chains and the embedding request (no network: httpx.post is replaced).

Run:  .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="wb-test-")
os.environ.update({"WORKBENCH_DB": os.path.join(_TMP, "test.db"), "WORKBENCH_NO_FETCH": "1", "WORKBENCH_SITE_URL": ""})
for k in ("WORKBENCH_MODEL_DEEP", "WORKBENCH_MODEL_FILTER", "WORKBENCH_MODEL_EMBED"):
    os.environ.pop(k, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import news_intel as NI  # noqa: E402
import server as S  # noqa: E402

S.init_db()
S.GEMINI_RETRY_DELAYS = (0.0, 0.0)   # no real waiting in tests (production: 2 s, 5 s)
KEY = "test-key-not-real-0000000000000000"


class FakeResp:
    def __init__(self, status: int, data: dict):
        self.status_code, self._data = status, data
        self.text = json.dumps(data)
        self.headers = {}

    def json(self):
        return self._data


def ok_text(text: str) -> FakeResp:
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})


def err(status: int, message: str, state: str = "") -> FakeResp:
    return FakeResp(status, {"error": {"code": status, "message": message, "status": state}})


class Recorder:
    """Stands in for httpx.post: answers by model name and records every request body."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, url, json=None, **kw):
        model = re.search(r"/models/([^:]+):", url).group(1)
        self.calls.append((model, json))
        a = self.answers(model, json, len([c for c in self.calls if c[0] == model]))
        return a


class Base(unittest.TestCase):
    def setUp(self):
        self._post = S.httpx.post
        for k in ("intel_model_cooldown", "intel_missing_models", "intel_used"):
            S.set_setting(k, None)
        S._missing_models.clear()

    def tearDown(self):
        S.httpx.post = self._post

    def use(self, answers):
        rec = Recorder(answers)
        S.httpx.post = rec
        return rec


class Defaults(unittest.TestCase):
    def test_models_and_chains(self):
        self.assertEqual(S.intel_models(), {"filter": "gemini-3.5-flash-lite", "embed": "gemini-embedding-001", "deep": "gemini-4-hydrogen"})
        self.assertEqual(S.model_chain("gemini-3.5-flash-lite", "filter")[:3], ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"])
        self.assertEqual(S.model_chain("gemini-4-hydrogen", "deep")[:3], ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.5-flash-lite"])
        self.assertEqual(NI.CLUSTER_SIM, 0.885)


class Gemini4Payload(Base):
    def body_for(self, model, **kw):
        rec = self.use(lambda m, j, n: ok_text('{"ok": true}'))
        S._gemini_once("prompt", KEY, model, "system", **kw)
        return rec.calls[-1][1]

    def assert_gemini4_rules(self, body, level):
        gen = body["generationConfig"]
        for k in ("temperature", "topP", "topK"):
            self.assertNotIn(k, gen)
        self.assertEqual(gen["thinkingConfig"], {"thinkingLevel": level})
        self.assertIn(gen["thinkingConfig"]["thinkingLevel"], ("low", "medium", "high", "xhigh"))
        self.assertGreaterEqual(gen["maxOutputTokens"], 32768)
        text = json.dumps(body)
        self.assertNotIn("thinkingBudget", text)
        self.assertNotIn("minimal", text)
        self.assertNotIn("cachedContent", text)

    def test_hydrogen_rules(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assert_gemini4_rules(self.body_for("gemini-4-hydrogen", schema=schema, max_tokens=8192, temperature=0.4, thinking="minimal"), "low")
        self.assert_gemini4_rules(self.body_for("gemini-4-hydrogen", max_tokens=2048, temperature=0.4, thinking="medium"), "medium")
        self.assert_gemini4_rules(self.body_for("gemini-4-hydrogen", thinking=None), "low")   # never the slow default

    def test_gemini3_keeps_its_rules(self):
        gen = self.body_for("gemini-3.5-flash-lite", max_tokens=32768, temperature=None, thinking="low")["generationConfig"]
        self.assertEqual(gen["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertNotIn("temperature", gen)

    def test_proxy_cleans_page_bodies(self):
        body = {"cachedContent": "cachedContents/x", "contents": [],
                "generationConfig": {"temperature": 0.7, "topP": 0.9, "topK": 40, "maxOutputTokens": 4096,
                                     "thinkingConfig": {"thinkingLevel": "minimal", "thinkingBudget": 0}}}
        self.assert_gemini4_rules(S.gemini4_clean_body("gemini-4-hydrogen", body), "low")
        self.assertEqual(S.gemini4_clean_body("gemini-3.8-flash", body), body)   # other models untouched

    def test_server_tasks_set_explicit_levels(self):
        """Scheduled work (brief, breakdowns, per-story triage) asks for 'low'; review cards for 'medium'."""
        src = Path(S.__file__).read_text(encoding="utf-8")
        self.assertIn('call_gemini(prompt, key, model, SYSTEM_PROMPT, max_tokens=32768, thinking="low"', src)
        self.assertIn('kind="deep", thinking="low")', src)
        self.assertIn('gemini_structured(prompt, ConceptCard, key, model, thinking="medium")', src)


class FallbackChains(Base):
    def test_503_on_hydrogen_steps_down(self):
        rec = self.use(lambda m, j, n: err(503, "The model is overloaded.", "UNAVAILABLE") if m == "gemini-4-hydrogen" else ok_text('{"ok": 1}'))
        S.call_gemini("p", KEY, "gemini-4-hydrogen", "s", kind="deep", thinking="low")
        self.assertEqual(S.gemini_model_used("?"), "gemini-3.8-flash")
        self.assertEqual([c[0] for c in rec.calls], ["gemini-4-hydrogen"] * 3 + ["gemini-3.8-flash"])
        left = S._cooldowns()["gemini-4-hydrogen"] - __import__("time").time()
        self.assertTrue(600 <= left <= 900, left)                 # 10-15 minutes, not 6 hours
        self.assertFalse(S._model_missing("gemini-4-hydrogen"))
        # the 3.8-flash fallback got Gemini 3 rules, the hydrogen tries got Gemini 4 rules
        self.assertEqual(rec.calls[0][1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertGreaterEqual(rec.calls[0][1]["generationConfig"]["maxOutputTokens"], 32768)

    def test_503_on_flash_lite_scoring_logs_the_winner(self):
        with S.db() as c:
            S.upsert_news(c, {"id": "t1", "kind": "article", "source": "techcrunch", "source_name": "TechCrunch",
                              "source_type": "press", "title": "A lab releases an open-weight model with 26B parameters",
                              "url": "https://example.com/a", "summary": "It loops its middle layers.", "published": S.iso()})

        def answers(model, body, n):
            if model == "gemini-3.5-flash-lite":
                return err(503, "The model is overloaded.", "UNAVAILABLE")
            ids = re.findall(r"^id: (\S+) \|", body["contents"][0]["parts"][0]["text"], re.M)
            return ok_text(json.dumps([{"id": i, "is_ai_relevant": True, "category": "Model releases", "technical_impact": 40,
                                        "industry_impact": 10, "event_slug": "lab-releases-open-model", "one_line_why": "x"} for i in ids]))
        self.use(answers)
        report = {"errors": [], "calls": 0, "scored": 0, "dropped": 0, "pending": 0}
        S._intel_score(KEY, "gemini-3.5-flash-lite", report, force=True)
        self.assertEqual(report["scored"], 1)
        self.assertEqual(report["used"]["filter"], "gemini-3.1-flash-lite")
        used = S.models_used()["filter"]
        self.assertEqual((used["model"], used["configured"], used["fallback"]), ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", True))
        self.assertIn("used", S.intel_snapshot())                  # published in intel.json

    def test_429_and_timeouts_step_down_too(self):
        def answers(model, body, n):
            if model == "gemini-4-hydrogen":
                return err(429, "You exceeded your current quota.", "RESOURCE_EXHAUSTED")
            if model == "gemini-3.8-flash":
                raise S.httpx.ReadTimeout("slow")
            return ok_text('{"ok": 1}')
        self.use(answers)
        S.call_gemini("p", KEY, "gemini-4-hydrogen", "s", kind="deep")
        self.assertEqual(S.gemini_model_used("?"), "gemini-3.5-flash-lite")

    def test_403_402_skip_the_model_for_6_hours_but_a_bad_key_stops(self):
        self.use(lambda m, j, n: err(403, "Permission denied for this model.", "PERMISSION_DENIED") if m == "gemini-4-hydrogen"
                 else err(402, "Billing required.") if m == "gemini-3.8-flash" else ok_text('{"ok": 1}'))
        S.call_gemini("p", KEY, "gemini-4-hydrogen", "s", kind="deep")
        self.assertEqual(S.gemini_model_used("?"), "gemini-3.5-flash-lite")
        self.assertTrue(S._model_missing("gemini-4-hydrogen") and S._model_missing("gemini-3.8-flash"))
        S.set_setting("intel_missing_models", None); S._missing_models.clear()
        self.use(lambda m, j, n: err(403, "API key not valid. Please pass a valid API key.", "PERMISSION_DENIED"))
        with self.assertRaises(S.GeminiError) as cm:
            S.call_gemini("p", KEY, "gemini-4-hydrogen", "s", kind="deep")
        self.assertEqual(cm.exception.kind, "auth")

    def test_old_long_pause_is_cleared(self):
        S.set_setting("intel_pause_score", S.iso(S.now_utc() + S.timedelta(hours=16)))
        self.assertEqual(S._pause_left("score"), 0.0)
        self.assertIsNone(S.get_setting("intel_pause_score"))


class Embeddings(Base):
    def test_request_is_unchanged(self):
        body = S.embed_request_body(["hello"], "gemini-embedding-001")
        r = body["requests"][0]
        self.assertEqual((r["model"], r["taskType"], r["outputDimensionality"]), ("models/gemini-embedding-001", "SEMANTIC_SIMILARITY", 768))

    def test_vectors_are_l2_normalized_and_503_retries_same_model(self):
        vals = [float(i % 7 - 3) for i in range(768)]   # deliberately not unit length
        rec = self.use(lambda m, j, n: err(503, "busy", "UNAVAILABLE") if n == 1 else FakeResp(200, {"embeddings": [{"values": vals}, {"values": vals}]}))
        out = S.embed_batch(["a", "b"], KEY, "gemini-embedding-001")
        self.assertEqual([c[0] for c in rec.calls], ["gemini-embedding-001", "gemini-embedding-001"])   # never another model
        for v in out:
            self.assertEqual(len(v), 768)
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
