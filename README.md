# Track: Practo (Healthcare)

## Dataset design choices

`dataset.py` generates 60 appointment records with `random.Random(SEED)` and
`SEED = 42`.

- **Category weights:** General Medicine 35, Cardiology 15, Dermatology 20,
  Pediatrics 15, Orthopedics 15.
- **Status weights:** Scheduled 30, Completed 35, Cancelled 15, No-Show 10,
  Rescheduled 10.
- **Fee ranges (INR):** General Medicine 300–800, Cardiology 800–2,000,
  Dermatology 500–1,500, Pediatrics 400–1,200, and Orthopedics 700–1,800.
- **Fee-range reasoning:** Specialty-specific ranges model higher fees for
  specialist consultations while keeping all fees within the required INR
  300–2,000 range.
- **Follow-up rate:** 28.3% of the generated records have `follow_up_required`
  set to `True` (within the 10%–30% target).

## Escalation score

The escalation score is `0.6 * follow_up_required + 0.4 * (days_since_created / 30)`, and the threshold is **0.8960**. This threshold corresponds to follow-up-required appointments at or above the 80th percentile of `days_since_created` in the generated data (P80 = 22.2 days); the values are calculated from the seeded appointment dataset and recorded in [`transcripts/escalation_threshold.txt`](./transcripts/escalation_threshold.txt).

## How to run

From the project directory, create and activate a Python 3.11 virtual
environment, install the pinned packages, and enable the offline CrewAI mock:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export CREWAI_DISABLE_TELEMETRY=true
```

The project uses `MockLLM` with model name `MOCK_LLM`; it does not require an
OpenAI API key. To generate the dataset, build the Chroma index, and run the
agent demo:

```bash
python dataset.py
python -m rag.index
```

Policy retrieval uses the measured `kb_fixed` threshold below by default. To
override it, set `RAG_SIMILARITY_THRESHOLD` to a value calibrated for the
selected collection:

```bash
python calibrate.py
export RAG_SIMILARITY_THRESHOLD=0.33285
python -m agents.crew
```

`chromadb==0.6.3` uses the positional `posthog.capture()` API, so
`requirements.txt` pins `posthog==2.5.0` for compatibility with Chroma. The
indexing and retrieval clients also set `anonymized_telemetry=False`; the
`CREWAI_DISABLE_TELEMETRY` setting above separately disables CrewAI telemetry.

## Threshold calibration

`calibrate.py` prints the top-1 similarities for three in-scope and two
out-of-scope queries, separately for `kb_fixed` and `kb_sentence`, along with
the gap between the clusters. The captured run in
[`transcripts/calibration.txt`](./transcripts/calibration.txt) reported:

| Collection | In-scope similarities | Out-of-scope similarities | Cluster gap |
|---|---|---|---:|
| `kb_fixed` | 0.7183, 0.4704, 0.6516 | 0.0426, 0.1953 | 0.2751 |
| `kb_sentence` | 0.6721, 0.5236, 0.7310 | 0.0014, 0.1627 | 0.3609 |

The production collection is `kb_fixed`. Its minimum in-scope score is 0.4704
and its maximum out-of-scope score is 0.1953, so the configured
`RAG_SIMILARITY_THRESHOLD` is `0.33285`, the midpoint of those measured bounds.
This keeps the threshold strictly between the observed clusters. The code uses
that measured value by default; `RAG_SIMILARITY_THRESHOLD` may override it.

## Chunking comparison

The captured evaluation in [`transcripts/rag_eval.txt`](./transcripts/rag_eval.txt)
reports `kb_fixed` macro precision **0.463** and recall **1.000**, versus
`kb_sentence` macro precision **0.320** and recall **1.000**. Both strategies
retrieve every hand-labelled relevant document, while fixed-overlap chunking
returns a smaller share of irrelevant chunks. We recommend `kb_fixed` because
its measured precision is higher without a recall tradeoff, and it is the
production collection.

## Governance

Only the Lookup agent receives `check_appointment_status`, while Retrieval
receives policy lookup and Composer receives no tools. The crew construction
assertion rejects any wiring that grants appointment access to another agent,
following the least-autonomy principle that each agent receives only the
capabilities required for its role; the rejection is demonstrated in
[`transcripts/least_autonomy.txt`](./transcripts/least_autonomy.txt).

The risk level is **High** because the system handles patient appointment and
clinical-policy data, which the project specification places in the High
category for medical data.

- **Token budget:** 1,000 estimated tokens per query, using
  `len(query.split()) * 1.3`. Over-budget queries raise `BudgetExceeded`; the
  API returns HTTP 413 for `/ask` and sends an error response for a WebSocket
  turn. Run `python -m governance.policy` to see the rejection demo.
- **Query cache:** grounded-generation results are held in memory using
  `" ".join(query.lower().split())` as the key. Run
  `python -m governance.cache` to see the generation counter on a cache miss
  and a repeated normalized query.
- **Guardrail test hook:** setting `GUARDRAIL_TEST_INJECT_UNGROUNDED=1` appends
  the deliberately unsupported Sunday-consultation claim before groundedness
  validation. It is used only by the guardrail and review-path demos; unset it
  for ordinary requests.

## Evaluation limitations

The MOCK_LLM judge is deterministic and rule-based, not a model that understands
meaning. Accuracy and completeness use configured topic keywords, grounding is
the fraction of unique answer content words also found in the context, and
safety checks for the project's phone-number and prompt-injection flags. Ratios
are mapped to scores from 1 to 5; these lexical checks can miss paraphrases,
semantic errors, and unsupported claims that happen to share words, and must
not be treated as clinical validation.

The latest 15-query run in [`transcripts/eval_15.txt`](./transcripts/eval_15.txt)
scored Accuracy **3.47/5**, Grounding **3.40/5**, Completeness **3.67/5**, and
Safety **5.00/5**. These averages describe only the deterministic lexical
heuristics and are not clinical-quality metrics.

## Transcript index

- `transcripts/dataset.txt` — generated appointment distribution summary;
  `python dataset.py > transcripts/dataset.txt`.
- `transcripts/index.txt` — indexed chunk counts for both collections;
  `python -m rag.index > transcripts/index.txt`.
- `transcripts/calibration.txt` — measured similarity clusters;
  `python calibrate.py > transcripts/calibration.txt`.
- `transcripts/grounded_generation.py` — runnable thresholded-retrieval demo;
  `python transcripts/grounded_generation.py` generates the adjacent output.
- `transcripts/grounded_generation.txt` — six grounded-generation cases;
  `python transcripts/grounded_generation.py > transcripts/grounded_generation.txt`.
- `transcripts/rag_eval.txt` — per-query precision/recall and macro averages;
  `python -m rag.eval_rag > transcripts/rag_eval.txt`.
- `transcripts/escalation_threshold.py` — runnable data-derived threshold
  calculation; run it with `python transcripts/escalation_threshold.py`.
- `transcripts/escalation_threshold.txt` — P80 and escalation threshold output;
  `python transcripts/escalation_threshold.py > transcripts/escalation_threshold.txt`.
- `transcripts/guardrail_demo.py` — runnable real-path guardrail demo; run it
  with `python transcripts/guardrail_demo.py`.
- `transcripts/guardrail_demo.txt` — phone masking, injection refusal, and
  ungrounded fallback; `python transcripts/guardrail_demo.py > transcripts/guardrail_demo.txt`.
- `transcripts/crew_run.txt` — tool routing and validated crew responses;
  `python -m agents.crew > transcripts/crew_run.txt`.
- `transcripts/review_demo.txt` — standalone AutoGen approval and revision;
  `python -m review.autogen_review > transcripts/review_demo.txt`.
- `transcripts/review_path_demo.py` — runnable review test through the crew;
  run it with `python transcripts/review_path_demo.py`.
- `transcripts/review_path_demo.txt` — production-path approval and revision;
  `python transcripts/review_path_demo.py > transcripts/review_path_demo.txt`.
- `transcripts/least_autonomy_demo.py` — runnable least-autonomy check; run it
  with `python transcripts/least_autonomy_demo.py`.
- `transcripts/least_autonomy.txt` — valid wiring and blocked Composer access;
  `python transcripts/least_autonomy_demo.py > transcripts/least_autonomy.txt`.
- `transcripts/budget_demo.py` — runnable API budget check; run it with
  `python transcripts/budget_demo.py`.
- `transcripts/budget_demo.txt` — normal and oversized API outcomes;
  `python transcripts/budget_demo.py > transcripts/budget_demo.txt`.
- `transcripts/cache_demo.py` — runnable normalized-cache check; run it with
  `python transcripts/cache_demo.py`.
- `transcripts/cache_demo.txt` — cache call counts and timings;
  `python transcripts/cache_demo.py > transcripts/cache_demo.txt`.
- `transcripts/api_demo.py` — runnable HTTP and WebSocket demo; run it with
  `python transcripts/api_demo.py` while the documented Uvicorn server runs.
- `transcripts/api_demo.txt` — endpoint responses and disconnect recovery;
  `python transcripts/api_demo.py > transcripts/api_demo.txt`.
- `transcripts/server.log` — Uvicorn process output;
  `uvicorn api.main:app --port 8000 > transcripts/server.log 2>&1`.
- `transcripts/log_check.py` — runnable JSONL log validator; run it with
  `python transcripts/log_check.py`.
- `transcripts/log_check.txt` — valid-line count, sample log lines, and PII
  result; `python transcripts/log_check.py > transcripts/log_check.txt`.
- `transcripts/eval_15.txt` — 15-query judge rows and four averages;
  `python -m eval.judge > transcripts/eval_15.txt`.
- `transcripts/memory_demo.py` — runnable, rule-based session-memory demo; run
  it with `python transcripts/memory_demo.py`.
- `transcripts/memory_transcript.txt` — three turns and isolated-session
  result; `python transcripts/memory_demo.py` writes this file.
