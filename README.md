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
  the normalized, lower-cased query plus the
  `GUARDRAIL_TEST_INJECT_UNGROUNDED` state as the key. This prevents a demo-hook
  response from being reused for an ordinary request. The captured
  [`transcripts/cache_demo.txt`](./transcripts/cache_demo.txt) shows a normalized
  cache hit and a separate hook-enabled generation.
- **Guardrail test hook:** setting `GUARDRAIL_TEST_INJECT_UNGROUNDED=1` appends
  the deliberately unsupported Sunday-consultation claim before groundedness
  validation. It is for demonstrations only, is included in the cache key,
  and should be unset for ordinary requests.

## Guardrails

The real request path applies PII masking, prompt-injection detection, and the
token-budget check before entering the cache and crew. After the crew runs,
the response is checked for groundedness and query/context relevance; an
unsupported draft is replaced with the clinic-policy fallback before review.
Only grounded drafts proceed to the AutoGen review stage. The
[`transcripts/guardrail_demo.txt`](./transcripts/guardrail_demo.txt) exercises
phone masking, injection refusal, and the groundedness fallback;
[`transcripts/review_path_demo.txt`](./transcripts/review_path_demo.txt) shows a
grounded draft being reviewed and an ungrounded draft being refused before
review.

## Appointment answers

Appointment status and consultation fee are taken deterministically from the
`check_appointment_status` tool result, not from the Composer's LLM output.
The appointment record is included in the grounding evidence, and the answer
reports the record ID, status, and fee. See
[`transcripts/crew_run.txt`](./transcripts/crew_run.txt) and the phone-masking
appointment example in [`transcripts/guardrail_demo.txt`](./transcripts/guardrail_demo.txt).

## Log masking

The API logger masks phone numbers in the raw request text before writing JSON
Lines. [`transcripts/log_phone_check.txt`](./transcripts/log_phone_check.txt)
was generated through the real `/ask` route with a phone-number query; it
confirms `[PHONE_MASKED]` was logged and the raw digits were absent.

## Evaluation limitations

The MOCK_LLM judge is deterministic and rule-based, not a model that understands
meaning. Accuracy and completeness use configured topic keywords, grounding is
the fraction of unique answer content words also found in the context, and
safety checks for the project's phone-number and prompt-injection flags. Ratios
are mapped to scores from 1 to 5; these lexical checks can miss paraphrases,
semantic errors, and unsupported claims that happen to share words, and must
not be treated as clinical validation.

The latest 15-query run in [`transcripts/eval_15.txt`](./transcripts/eval_15.txt)
scored Accuracy **4.87/5**, Grounding **4.27/5**, Completeness **4.87/5**, and
Safety **5.00/5**. The two in-scope queries below 5/5 Accuracy are the
cancellation/rescheduling question and the Cardiology-fee question. The
rule-based keyword judge expects `cancel`, while the policy uses forms such as
`cancelled` and `cancellations`; it does not stem inflected words. For the fee
query, the policy writes the amount as `₹1,200`, while the judge's ASCII
tokenizer does not normalize that formatted amount to the expected token
`1200`. These are lexical-scoring limitations, not missing policy records; the
expected topics and keyword sets were not changed. The three out-of-scope
queries have correct abstention Accuracy, but their Grounding scores are low
because the lexical judge compares the fallback text with irrelevant or empty
retrieved context. None of these scores is clinical validation.

## Transcript index

Every file under `transcripts/` is listed below. Demo `.py` files are runnable
sources; each captured `.txt` entry names the command that produced its output.

- `transcripts/api_demo.py` — HTTP/WebSocket demo source; run with
  `python transcripts/api_demo.py` while Uvicorn is running.
- `transcripts/api_demo.txt` — `/ask`, `/add-document`, WebSocket reply, and
  abrupt-disconnect recovery; `python transcripts/api_demo.py > transcripts/api_demo.txt`.
- `transcripts/budget_demo.py` — API budget demo source; run with
  `python transcripts/budget_demo.py`.
- `transcripts/budget_demo.txt` — normal 200 and oversized 413 responses;
  `python transcripts/budget_demo.py > transcripts/budget_demo.txt`.
- `transcripts/cache_demo.py` — normalized cache and test-hook separation
  source; run with `python transcripts/cache_demo.py`.
- `transcripts/cache_demo.txt` — cache hit, call counts, and hook isolation;
  `python transcripts/cache_demo.py > transcripts/cache_demo.txt`.
- `transcripts/calibration.txt` — measured RAG similarity scores;
  `python calibrate.py > transcripts/calibration.txt`.
- `transcripts/crew_run.txt` — policy and appointment tool routing with
  validated responses; `python -m agents.crew > transcripts/crew_run.txt`.
- `transcripts/dataset.txt` — seeded dataset distribution;
  `python dataset.py > transcripts/dataset.txt`.
- `transcripts/escalation_threshold.py` — runnable threshold calculation;
  run with `python transcripts/escalation_threshold.py`.
- `transcripts/escalation_threshold.txt` — P80 and calculated threshold;
  `python transcripts/escalation_threshold.py > transcripts/escalation_threshold.txt`.
- `transcripts/eval_15.txt` — 15 evaluation cases and four averages;
  `python -m eval.judge > transcripts/eval_15.txt`.
- `transcripts/grounded_generation.py` — runnable retrieval demo source;
  run with `python transcripts/grounded_generation.py`.
- `transcripts/grounded_generation.txt` — five in-scope and one out-of-scope
  retrieval result; `python transcripts/grounded_generation.py > transcripts/grounded_generation.txt`.
- `transcripts/guardrail_demo.py` — real request-path guardrail demo source;
  run with `python transcripts/guardrail_demo.py`.
- `transcripts/guardrail_demo.txt` — PII masking, injection refusal, grounded
  refusal; `python transcripts/guardrail_demo.py > transcripts/guardrail_demo.txt`.
- `transcripts/index.txt` — indexed chunk counts;
  `python -m rag.index > transcripts/index.txt`.
- `transcripts/least_autonomy_demo.py` — runnable tool-access guard source;
  run with `python transcripts/least_autonomy_demo.py`.
- `transcripts/least_autonomy.txt` — valid Lookup access and blocked Composer
  access; `python transcripts/least_autonomy_demo.py > transcripts/least_autonomy.txt`.
- `transcripts/log_check.py` — JSONL log validator source; run with
  `python transcripts/log_check.py`.
- `transcripts/log_check.txt` — JSONL line count, sample entries, and raw-PII
  check; `python transcripts/log_check.py > transcripts/log_check.txt`.
- `transcripts/log_phone_check.py` — TestClient phone-mask verifier source;
  run with `python transcripts/log_phone_check.py`.
- `transcripts/log_phone_check.txt` — real `/ask` request and masked log
  evidence; `python transcripts/log_phone_check.py > transcripts/log_phone_check.txt`.
- `transcripts/memory_demo.py` — rule-based memory demo source; run with
  `python transcripts/memory_demo.py`.
- `transcripts/memory_transcript.txt` — three-turn memory and session
  isolation; written by `python transcripts/memory_demo.py`.
- `transcripts/rag_eval.txt` — per-query and macro precision/recall;
  `python -m rag.eval_rag > transcripts/rag_eval.txt`.
- `transcripts/review_demo.txt` — standalone AutoGen approval and revision;
  `python -m review.autogen_review > transcripts/review_demo.txt`.
- `transcripts/review_path_demo.py` — production review-path demo source;
  run with `python transcripts/review_path_demo.py`.
- `transcripts/review_path_demo.txt` — grounded approval and guarded refusal;
  `python transcripts/review_path_demo.py > transcripts/review_path_demo.txt`.
- `transcripts/server.log` — clean Uvicorn startup, HTTP/WebSocket requests,
  and graceful shutdown; `uvicorn api.main:app --port 8000 > transcripts/server.log 2>&1`.
