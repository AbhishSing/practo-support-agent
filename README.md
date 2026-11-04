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

Before running policy retrieval, measure the similarity values as described
below and set `RAG_SIMILARITY_THRESHOLD` to the threshold you select:

```bash
python calibrate.py
export RAG_SIMILARITY_THRESHOLD="<your calibrated threshold>"
python -m agents.crew
```

## Threshold calibration

The project has not yet recorded calibration measurements or a selected
threshold. The current workspace has no `chroma_db` index and lacks the
`chromadb`, `sentence_transformers`, and `numpy` packages, so similarity values
cannot be reported without installing dependencies and building the index.
`calibrate.py` prints the top-1 similarities for three in-scope and two
out-of-scope queries, separately for `kb_fixed` and `kb_sentence`, along with
the gap between the clusters. Choose and record the threshold only after
reviewing those measured values; no threshold is assumed here.

## Chunking comparison

Macro precision and recall for `kb_fixed` and `kb_sentence` have not yet been
measured in this workspace; run `python -m rag.eval_rag` after indexing to
produce the per-query arithmetic and collection averages. **Recommendation:**
do not declare a winning collection until those results are available. Compare
both macro precision and recall on the hand-labelled queries, then prefer the
strategy that best meets the project's retrieval needs.

## Governance

- **Least autonomy:** only the Lookup agent receives
  `check_appointment_status`; crew construction asserts that the other agents
  do not have this tool.
- **Risk level: High.** The system handles patient appointment and
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

## Evaluation limitations

The MOCK_LLM judge is deterministic and rule-based, not a model that understands
meaning. Accuracy and completeness use configured topic keywords, grounding is
the fraction of unique answer content words also found in the context, and
safety checks for the project's phone-number and prompt-injection flags. Ratios
are mapped to scores from 1 to 5; these lexical checks can miss paraphrases,
semantic errors, and unsupported claims that happen to share words, and must
not be treated as clinical validation.

## Transcript index

- `transcripts/memory_demo.py` and `transcripts/memory_transcript.txt`:
  demonstrate three-turn session memory and isolation between two sessions.
- `transcripts/guardrail_demo.py`: demonstrates phone masking, injection
  detection, and a rejected ungrounded answer.
- `transcripts/review_demo.txt`: records AutoGen's deterministic approval of a
  supported draft and revision of a draft containing an unsupported Sunday
  consultation claim.
