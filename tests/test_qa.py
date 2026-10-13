"""python -m unittest discover tests   (stdlib only, no key, no network)"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze  # noqa: E402
import noise  # noqa: E402
import qa  # noqa: E402

CALLS = qa.load_transcripts()


class NoiseTest(unittest.TestCase):
    def test_measured_wer_matches_target_exactly(self):
        for c in CALLS:
            ref = " ".join(t["text"] for t in c["turns"])
            n = len(noise.words(ref))
            for w in [0.05, 0.10, 0.20, 0.30]:
                noisy, _ = noise.corrupt(c["turns"], w, seed=1, key=c["id"])
                got = noise.wer(ref, " ".join(t["text"] for t in noisy))
                self.assertAlmostEqual(got, round(w * n) / n, msg=f"{c['id']} @ {w}")

    def test_deterministic_and_seed_sensitive(self):
        turns = CALLS[0]["turns"]
        a, _ = noise.corrupt(turns, 0.2, seed=0, key="T01")
        b, _ = noise.corrupt(turns, 0.2, seed=0, key="T01")
        c, _ = noise.corrupt(turns, 0.2, seed=1, key="T01")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_speaker_swaps_scale_with_wer(self):
        turns = CALLS[0]["turns"]
        _, ops = noise.corrupt(turns, 0.30, seed=0, key="T01")
        self.assertEqual(sum(o["op"] == "speaker" for o in ops), round(0.5 * 0.30 * len(turns)))


class DataTest(unittest.TestCase):
    def test_gold_is_balanced(self):
        self.assertEqual(len(CALLS), 24)
        for crit in qa.CRITERIA:
            self.assertEqual(sum(1 - c["gold"][crit] for c in CALLS), 6, crit)

    def test_every_failed_criterion_has_evidence(self):
        for c in CALLS:
            for crit, ok in c["gold"].items():
                if not ok:
                    self.assertTrue(c["evidence"][crit], f"{c['id']} {crit}")


class GroundingTest(unittest.TestCase):
    turns = [{"speaker": "agent", "text": "Good morning, my name is Sanne."},
             {"speaker": "customer", "text": "I was charged twice."}]

    def test_verbatim_quote_passes_across_turns_and_labels(self):
        self.assertTrue(qa.grounded("my name is sanne", self.turns))
        self.assertTrue(qa.grounded("Agent: my name is Sanne. Customer: I was charged", self.turns))

    def test_paraphrase_and_empty_fail(self):
        self.assertFalse(qa.grounded("my name's Sanne", self.turns))
        self.assertFalse(qa.grounded("", self.turns))


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        body = {n: {"quote": "q", "comment": "c", "verdict": "yes"} for n in qa.CRITERIA}
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
                               content=[SimpleNamespace(type="text", text=json.dumps(body))])


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def judge(self, client, budget=1.0, offline=False):
        return qa.Judge(client=client, offline=offline, budget=budget,
                        cache_dir=self.tmp / "cache", usage_log=self.tmp / "usage.jsonl")

    def test_cache_hit_never_calls_or_bills_again(self):
        fake = FakeClient()
        j = self.judge(fake)
        j.score("p", rep=0)
        j.score("p", rep=0)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(len((self.tmp / "usage.jsonl").read_text().splitlines()), 1)
        self.assertIsNotNone(self.judge(None, offline=True).score("p", rep=0))  # offline replays the cache
        self.assertIsNone(self.judge(None, offline=True).score("other", rep=0))

    def test_budget_counts_prior_runs_and_stops_before_the_call(self):
        fake = FakeClient()
        self.judge(fake).score("p", rep=0)
        spent = self.judge(fake).spent
        with self.assertRaises(qa.BudgetExceeded):
            self.judge(fake, budget=spent + 1e-6).score("p", rep=1)
        self.assertEqual(fake.calls, 1)


class AnalyzeTest(unittest.TestCase):
    def row(self, cid, wer, seed, v):
        return {"id": cid, "wer": wer, "seed": seed, "verdict": {n: v for n in qa.CRITERIA}}

    def test_floor_is_subtracted(self):
        # Clean repeats disagree in 2 of 3 pairs; each noisy draw disagrees with 1 of 3 clean repeats.
        rows = [self.row("X", 0.0, 0, 1), self.row("X", 0.0, 1, 1), self.row("X", 0.0, 2, 0)]
        rows += [self.row("X", 0.30, s, 1) for s in qa.SEEDS]
        by = analyze.index(rows)
        self.assertAlmostEqual(analyze.flip_rate(by, ["X"], "opening", 0.0), 2 / 3)
        self.assertAlmostEqual(analyze.flip_rate(by, ["X"], "opening", 0.30), 1 / 3)
        point, _, _ = analyze.excess_ci(by, ["X"], "opening", 0.30, n_boot=10)
        self.assertAlmostEqual(point, 1 / 3 - 2 / 3)


if __name__ == "__main__":
    unittest.main()
