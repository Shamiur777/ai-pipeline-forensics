# AI Pipeline Forensics (find the step that broke it, and turn failures into eval cases)

[![tests](https://github.com/Shamiur777/ai-pipeline-forensics/actions/workflows/tests.yml/badge.svg)](https://github.com/Shamiur777/ai-pipeline-forensics/actions)

Two tools that feed one growing evaluation dataset. A tracing and failure-localisation layer finds *which step* of a multi-step AI pipeline caused a bad final output. A log miner scans production LLM logs and turns the interesting interactions into labelled eval cases.

All data in this repo is synthetic and all model behaviour is simulated (a scripted mock provider). Nothing here measures real model quality.

## The problem
When a multi-step AI pipeline returns a bad answer, the last step usually looks guilty, and the first step that looks odd is not always the real cause. Teams then fix the wrong step, and the failure never makes it into their test set. Separately, production logs hold the best test cases (errors, thumbs-down, user corrections, retries), but nobody has time to curate them by hand.

## How it works

```
 pipeline run -> spans (JSONL) -> per-step validators -> earliest bad step
                                          |
                                          +-> counterfactual replay: put a known-good output
                                              at step k, re-run, does the FINAL output pass?
                                          |
                                          +-> ranked blame report  -> eval dataset (JSONL)
                                                                           ^
 production logs (JSONL) -> signals + TF-IDF rarity -> dedupe -> stratify -+
                            (incremental, state file)
```

- `forensics/tracing.py`: decorator and context manager that record nested spans (name, input, output, latency, error, metadata, parent) to JSONL. Standard library only. Exceptions are recorded and re-raised.
- `forensics/pipeline.py`: example support-ticket assistant (classify, retrieve policy, draft reply, safety check) with a scripted mock model and five switchable faults, so every failure is reproducible.
- `forensics/validators.py`: pluggable checks per step: `Schema`, `Contains`, `Length`, `Custom`, and an optional `LLMJudge` that goes through the `Provider` interface.
- `forensics/localize.py`:
  - **Earliest bad step:** the first step whose validators fail.
  - **Counterfactual replay:** for each step, substitute a known-good output, re-run the downstream steps, and test whether the final output passes. The root cause is the earliest step whose substitution fixes the final output. A failing step that does not fix it is a *false alarm*. A later failing step that does fix it is a *downstream victim*.
  - **Blame report:** ranked, with the validator evidence and the replay verdict for each step.
- `forensics/dataset.py`: append-only JSONL store in the golden shape `{id, input, expected, tags, weight}`, deduped by normalised input hash. Failures get their `expected` from the fixing counterfactual, or are marked `needs_review` with a reason.
- `forensics/miner.py`: the log miner (below). `forensics/synth.py` generates seeded logs with planted failures so the miner can be scored.

### The tricky case
With faults `retrieve_verbose` (right policy plus noise, harmless) and `draft_hallucination` injected together, the earliest suspicious step is `retrieve` (it fails a length check), but replacing it does not help. Replacing `draft` does. The demo prints this:

```
Earliest suspicious step: retrieve
Root cause (counterfactual): draft  [confidence: high]
  classify   -> final still bad
  retrieve   -> final still bad
  draft      -> final becomes GOOD
  safety     -> final becomes GOOD
```

### Log miner
Signals: errors, thumbs-down, user edits, retries, latency outliers (robust z-score per model), refusals (regex), and very long or very short responses. Rarity comes from TF-IDF vectors (pure Python) and leader clustering; clusters no larger than max(3, 2% of the batch) get a `rare_cluster` signal and a score boost. Per (cluster, failure type) it keeps up to 3 diverse representatives (farthest-point selection). Then: dedupe against the dataset and within the run, and stratify with a round-robin over failure types under a per-run budget (default 60) and a per-type cap (40% of budget when 3 or more types compete). If the user edited the answer, the edit becomes `expected`; otherwise the case is `needs_review` with the triggering signal as the reason. Each case records provenance (log id, run number, signals, score, cluster size). A state file stores seen log ids, so each run only processes new logs and appends a growth record.

## Results
Everything below is measured on synthetic data and a simulated pipeline. Reproduce with:

```bash
python -m forensics.evaluate
```

**Failure localisation** (7 fault scenarios x 8 sample tickets = 56 failing runs, simulated model):

| Method | Correct root cause | On the 24 runs with a harmless decoy |
|---|---|---|
| Earliest bad step | 40 of 56 | 8 of 24 |
| Counterfactual replay | 56 of 56 | 24 of 24 |

This is the expected outcome for a clean, single-root-cause simulation where I wrote both the faults and the validators. It shows the method separates cause from decoy; it does not predict accuracy on a real pipeline. The earliest-bad failures are exactly the decoy runs.

**Log miner** (400 synthetic log lines per run, seeds 7, 8, 9, default settings; about 9% of lines are planted failures across 9 modes, plus unplanted noise: 2% random thumbs-down and 3% mildly slow calls):

| Run | Planted | Flagged | Dups removed | Added | Precision of added | Planted covered (recall) | Dataset size |
|---|---|---|---|---|---|---|---|
| 1 | 35 | 50 | 11 | 39 | 0.718 | 31 (0.886) | 39 |
| 2 | 39 | 55 | 27 | 28 | 0.679 | 35 (0.897) | 67 |
| 3 | 39 | 55 | 40 | 15 | 0.600 | 33 (0.846) | 82 |

- Recall counts a planted failure as covered when its normalised input is in the dataset *and* the stored case came from a planted log. Precision is the share of added cases that came from planted logs; the unplanted noise (users downvoting fine answers) lowers it by design, and those cases are not necessarily useless.
- Dedupe removes more each run (11, 27, 40 of 50, 55, 55 flagged) because the generator reuses a small set of prompts. Growth slows as the dataset saturates, from +39 to +28 to +15.
- Largest failure type in the final 82-case dataset: 22% (latency outliers 18, thumbs-down 17, retry 12, length anomaly 11, user edit 9, rare cluster 6, refusal 6, error 3). 9 cases have an `expected` (user edits), 73 are `needs_review`.
- The suite includes a regression test that fails if recall drops below 0.75 or precision below 0.5 per run.

The generator is templated and easy. The real world will be harder; I have not run this on real logs.

## Run it
Windows, from the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                          # 33 tests, no API key needed
python -m forensics demo                     # end-to-end offline demo, writes ./demo_out/
python -m forensics analyze demo_out/trace.jsonl --dataset demo_out/eval_dataset.jsonl
python -m forensics mine demo_out/prod_logs_1.jsonl --dataset data/eval.jsonl --state data/state.json
python -m forensics.evaluate                 # reproduce the Results tables
```

`AnthropicProvider` (optional, reads `ANTHROPIC_API_KEY`, model from `ANTHROPIC_MODEL`, default `claude-sonnet-5-5`) is included for the `LLMJudge` validator or your own pipeline. It has only been tested with an injected fake client, never against the live API.

## Design decisions
- **Counterfactual over heuristics.** Validators say where something looks wrong; replay says where it actually matters. The report shows both, so a decoy is labelled `false_alarm` instead of silently ignored.
- **Root cause = earliest step whose fix repairs the final output.** Later steps trivially repair the output when replaced, so "earliest" is what separates cause from victim.
- **Known-good outputs are an input.** Here they come from a fault-free run; in practice they would come from a golden record or a human correction.
- **Golden-compatible output with an extra `meta` field.** Tags are `key:value` strings; consumers that read only the five core fields can ignore `meta`.
- **Deterministic everywhere.** Seeded generators, scripted mock model, counter-based span ids.

## Limits and next steps
- The pipeline, faults, validators and logs are all written by me, so the numbers show the method works on a toy, not that it generalises.
- Counterfactual replay needs a re-runnable pipeline and known-good outputs. Two independent faults that must both be fixed are reported as `unresolved` (earliest bad step, low confidence), not decomposed.
- Replay re-runs deterministic steps; real LLM steps are stochastic and would need repeated replays.
- Tracing is sync only (no async or distributed spans).
- Dedupe is one case per normalised input, so a prompt that failed in two different ways yields one case.
- Rarity is computed per batch, not across history. Clustering is leader-based and order dependent.
- Refusal and long/short detection are simple heuristics. No PII scrubbing of logs.
- Next: async tracing, rarity across runs, combined-fault replay, and running the miner on real logs.
