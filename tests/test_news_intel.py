"""Unit tests for news_intel.py (no network, no database).

Run:  .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import news_intel as NI  # noqa: E402


class Sanitizer(unittest.TestCase):
    def test_wordpress_footer(self):
        s = ('<p>Nvidia&#8217;s board voted to reinstate Jensen Huang.</p><p>The post <a href="x">Nvidia reinstates Huang</a> '
             'appeared first on <a href="y">SiliconANGLE</a>.</p>')
        self.assertEqual(NI.sanitize_snippet(s), "Nvidia’s board voted to reinstate Jensen Huang.")

    def test_guardian_promo_and_continue_reading(self):
        s = ("Senator says the affair was costly. Get our breaking news email, free app or daily news podcast "
             "The chipmaker's board met on Friday. Continue reading...")
        self.assertEqual(NI.sanitize_snippet(s), "Senator says the affair was costly. The chipmaker's board met on Friday.")

    def test_read_on_and_bracket_ellipsis(self):
        self.assertEqual(NI.sanitize_snippet("OpenAI shipped GPT-6.2. READ ON: our full review of the model"), "OpenAI shipped GPT-6.2.")
        self.assertEqual(NI.sanitize_snippet("A new agent framework lands today. It supports MCP [&#8230;]"),
                         "A new agent framework lands today.")

    def test_truncated_tail(self):
        out = NI.sanitize_snippet("The company said the model beats rivals on coding benchmarks. It also said it would inves…")
        self.assertEqual(out, "The company said the model beats rivals on coding benchmarks.")
        short = NI.sanitize_snippet("Researchers propose a looped transformer that reuses middle lay...")
        self.assertTrue(short.endswith("…") and "lay" not in short.split()[-1])

    def test_hn_feed_body(self):
        s = "Article URL: https://example.com/a Comments URL: https://news.ycombinator.com/item?id=1 Points: 120 # Comments: 44"
        self.assertEqual(NI.sanitize_snippet(s), "")

    def test_idempotent_and_entities(self):
        s = "Gemini &amp;amp; Claude &lt;b&gt;compared&lt;/b&gt;​ on   tasks."
        once = NI.sanitize_snippet(s)
        self.assertEqual(once, "Gemini & Claude compared on tasks.")
        self.assertEqual(NI.sanitize_snippet(once), once)

    def test_clip(self):
        self.assertEqual(NI.clip("short", 10), "short")
        c = NI.clip("one two three four five six seven", 20)
        self.assertTrue(c.endswith("…") and len(c) <= 21 and not c[:-1].endswith(" "))


class Tiers(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(NI.source_tier("lab", "openai"), "primary")
        self.assertEqual(NI.source_tier("hub", "hf_models"), "primary")
        self.assertEqual(NI.source_tier("repo", "github"), "primary")
        self.assertEqual(NI.source_tier("community", "hn"), "community")
        self.assertEqual(NI.source_tier("analysis", "chinai"), "community")
        self.assertEqual(NI.source_tier("press", "techmeme"), "press")
        self.assertEqual(NI.source_tier("lab", "alerts"), "primary")
        self.assertEqual(NI.source_tier("custom", "custom1", "https://arxiv.org/abs/2611.1"), "primary")
        self.assertEqual(NI.source_tier(None), "press")


class Vectors(unittest.TestCase):
    def test_normalize_and_cosine(self):
        rnd = random.Random(1)
        v = [rnd.uniform(-1, 1) for _ in range(768)]
        n = NI.l2_normalize(v)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in n)), 1.0, places=9)
        self.assertAlmostEqual(NI.cosine(n, n), 1.0, places=9)
        self.assertEqual(NI.l2_normalize([0.0, 0.0]), [0.0, 0.0])

    def test_pack_roundtrip(self):
        v = NI.l2_normalize([1.0, 2.0, 3.0])
        back = NI.unpack_vec(NI.pack_vec(v))
        for a, b in zip(v, back):
            self.assertAlmostEqual(a, b, places=6)

    def test_best_event_threshold(self):
        a = NI.l2_normalize([1, 0, 0])
        b = NI.l2_normalize([0.9, 0.1, 0])
        c = NI.l2_normalize([0, 1, 0])
        cents = {"e1": NI.centroid([a]), "e2": NI.centroid([c])}
        self.assertEqual(NI.best_event(b, cents)[0], "e1")
        self.assertIsNone(NI.best_event(NI.l2_normalize([1, 1, 1]), cents, 0.9)[0])


class Drift(unittest.TestCase):
    """Real embeddings of unrelated AI stories share a strong generic direction. The centroid of a few such stories
    approaches that direction, so every new story is 'similar to the event'; the mean similarity is not fooled."""

    def test_mean_similarity_resists_generic_blob(self):
        rnd = random.Random(7)
        dim = 768
        generic = NI.l2_normalize([rnd.gauss(0, 1) for _ in range(dim)])

        def story():   # unrelated AI story: generic part + its own topic
            own = NI.l2_normalize([rnd.gauss(0, 1) for _ in range(dim)])
            return NI.l2_normalize([0.86 * g + 0.51 * o for g, o in zip(generic, own)])

        members = [story() for _ in range(8)]
        new = story()
        pair = NI.cosine(members[0], members[1])
        self.assertLess(pair, 0.82)                       # two unrelated stories do not merge
        cent = NI.centroid(members)
        self.assertGreaterEqual(NI.cosine(new, cent), 0.82)   # ...but the centroid would swallow a new one
        s = [sum(col) for col in zip(*members)]
        self.assertLess(NI.mean_similarity(new, s, len(members)), 0.82)   # the mean does not
        # a real rephrasing of member 0 still joins
        rephrase = NI.l2_normalize([0.97 * a + 0.24 * b for a, b in zip(members[0], NI.l2_normalize([rnd.gauss(0, 1) for _ in range(dim)]))])
        self.assertGreaterEqual(NI.mean_similarity(rephrase, members[0], 1), 0.82)

    def test_embeddable(self):
        self.assertFalse(NI.embeddable("article", "The Toolsmith", ""))
        self.assertFalse(NI.embeddable("model", "Qwen/Qwen3.8-Flash", "A long description of the model and what it does well."))
        self.assertTrue(NI.embeddable("article", "Nvidia's board votes to reinstate Jensen Huang as CEO", ""))
        self.assertTrue(NI.embeddable("discussion", "Show HN: a tiny agent runtime", "It runs agents in microVMs with snapshots."))


class Scoring(unittest.TestCase):
    def test_formula(self):
        s, p = NI.display_score(40, 10, outlets=3, tier="primary")
        self.assertEqual(s, 40 + round(3.5) + 10 + 6)
        self.assertEqual(p["tier"], "Notable")
        s2, _ = NI.display_score(10, 48, outlets=9, tier="press", hn_points=150)
        self.assertEqual(s2, 48 + 4 + 20 + 3)
        self.assertEqual(NI.score_tier(75), "Major")
        self.assertEqual(NI.score_tier(74), "Notable")
        self.assertEqual(NI.score_tier(49), "Low")
        s3, _ = NI.display_score(8, 12, outlets=1, tier="primary", hn_points=500)   # primary bonus, never both
        self.assertEqual(s3, 12 + 3 + 6)

    def test_validate_scores(self):
        reply = [
            {"id": "a", "is_ai_relevant": True, "category": "Hardware & chips", "technical_impact": 31.6, "industry_impact": 80,
             "event_slug": "Nvidia Reinstates  Huang!", "one_line_why": "Leadership change at the largest AI chip vendor."},
            {"id": "b", "is_ai_relevant": "false", "category": "research", "technical_impact": 0, "industry_impact": 0,
             "event_slug": "x", "one_line_why": "n/a"},
            {"id": "zzz", "is_ai_relevant": True, "category": "Agents", "technical_impact": 1, "industry_impact": 1,
             "event_slug": "y", "one_line_why": "unknown id"},
            {"id": "c", "is_ai_relevant": True, "category": "Not a category", "technical_impact": 1, "industry_impact": 1,
             "event_slug": "z", "one_line_why": "bad enum"},
        ]
        out = NI.validate_scores(reply, ["a", "b", "c"])
        self.assertEqual(set(out), {"a", "b"})
        self.assertEqual(out["a"]["category"], "hardware")
        self.assertEqual(out["a"]["technical_impact"], 32)
        self.assertEqual(out["a"]["industry_impact"], 50)
        self.assertEqual(out["a"]["event_slug"], "nvidia-reinstates-huang")
        self.assertFalse(out["b"]["ai_relevant"])
        self.assertEqual(out["b"]["category"], "research")

    def test_prompt_contains_ids_and_known_events(self):
        p = NI.score_prompt([{"id": "n1", "title": "T &amp; U", "summary": "<p>S</p>", "source_name": "Wired", "tier": "press"}],
                            [("nvidia-ceo", "Jensen Huang is back")])
        self.assertIn("id: n1", p)
        self.assertIn("T & U", p)
        self.assertIn("nvidia-ceo: Jensen Huang is back", p)


class Brief(unittest.TestCase):
    def test_validate_brief(self):
        keys = {"E1": {"event_id": "ev:1", "item_id": "1", "source_name": "OpenAI"},
                "E2": {"event_id": "ev:2", "item_id": "2", "source_name": "Wired"}}
        reply = {"executive_summary": "A busy day.", "sections": [
            {"heading": "Industry & policy", "items": [{"title": "Huang returns", "summary": "Board reverses.", "source_name": "Wired", "event_id": "E2"}]},
            {"heading": "Top model releases", "items": [
                {"title": "GPT ships", "summary": "New model.", "source_name": "OpenAI", "event_id": "[E1]"},
                {"title": "Duplicate", "summary": "x", "source_name": "x", "event_id": "E1"},
                {"title": "Invented", "summary": "x", "source_name": "x", "event_id": "E9"}]},
            {"heading": "Not a heading", "items": [{"title": "x", "summary": "x", "source_name": "x", "event_id": "E2"}]}]}
        out = NI.validate_brief(reply, keys)
        self.assertEqual([s["heading"] for s in out["sections"]], ["Top model releases", "Industry & policy"])
        self.assertEqual(out["sections"][0]["items"][0]["event_id"], "ev:1")
        self.assertEqual(len(out["sections"][0]["items"]), 1)
        with self.assertRaises(ValueError):
            NI.validate_brief({"executive_summary": "x", "sections": []}, keys)


if __name__ == "__main__":
    unittest.main()
