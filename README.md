# ESG / Safety Incident Intelligence Dashboard

Turns free-text safety and ESG incident narratives into management-ready insight: **category**, **severity**, **risk score**, a **one-line summary**, **recurring patterns** and **recommended preventive actions**, with the evidence behind each call.

**Context:** Greenko-style renewable-energy sites (solar blocks, wind gearboxes, substations, battery containers, contractor camps), where EHS teams get many short narratives and need to see what matters and what keeps recurring.

## Run locally
```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload        # open http://localhost:8000  (API docs: /docs)
python -m backend.pipeline              # re-run evaluation, writes docs/evaluation.json
```

## Deploy to Railway
1. Push this folder to a GitHub repo (`git init && git add . && git commit -m "ESG dashboard"`, then push).
2. Railway > New Project > Deploy from GitHub repo > select it. Nixpacks detects Python from `requirements.txt`; `.python-version` pins 3.11; the `Procfile` starts uvicorn on `$PORT`.
3. Service > Settings > Networking > **Generate Domain**. Open it: the dashboard is served at `/`, API at `/api/*`.
4. Check `https://<your-domain>/api/health` returns `{"status":"ok"}`, then add that link and a screenshot to your submission. No secrets or API keys are needed.

## How it works
| Step | Method |
|---|---|
| Preprocessing | lowercase, strip punctuation, TF-IDF (1-2 grams, stop-words removed) |
| Category (safety / environment / esg_compliance) | Blend of TF-IDF + logistic regression (40%) and an EHS keyword-lexicon prior (60%) |
| Severity (low / medium / high) | Blend of logistic regression (20%) and rule signals: critical keywords (fire, fracture, hospital...) force high; near-miss, energy/height/vehicle hazards add; "contained", "spill kit", "minor" subtract |
| Risk score 0-100 | Expected severity from the probabilities (20/55/90), +5 if no response action is recorded |
| Summary | Extractive, template-based: severity + category + asset + hazard themes + response recorded |
| Patterns / trends | 12 hazard themes matched per incident; counts, share, high-severity count; a theme with 3+ incidents is flagged **recurring**; top keywords per category |
| Preventive actions | Theme-to-corrective-action library, ranked by frequency and severity |
| Retrieval | TF-IDF cosine similarity returns the 3 most similar past incidents |
| Explainability | Top contributing terms, matched rule hits and severity signals returned with every prediction |

**API:** `GET /api/dashboard`, `GET /api/incidents`, `POST /api/analyze {"text": ...}` (also `/api/run`), `POST /api/batch`, `POST /api/upload` (CSV with a `narrative` column), `GET /api/evaluation`, `GET /api/health`.

## Data and assumptions
- `data/safety_esg_incidents.csv`: the 6 bundled incidents (used as the gold check).
- `data/safety_esg_incidents_extended.csv`: 48 incidents **I wrote and labelled** (16 per category) to make training and validation possible. Labels reflect my reading of typical EHS practice; they are not real company records.
- Assumption: severity is judged on potential and actual consequence in the narrative; "near miss" with energy or vehicle hazards is high even without injury (consistent with S004).
- The data has no dates, so "trends" means recurring-pattern frequency, not time series. If a `date` column is added, monthly trends are the natural next step.

## Evaluation (5-fold stratified CV, 54 incidents)
| Task | Majority | Rules only | ML only | **Hybrid (used)** |
|---|---|---|---|---|
| Category accuracy / macro-F1 | 0.315 / 0.244 | 0.981 / 0.981 | 0.611 / 0.615 | **0.963 / 0.962** |
| Severity accuracy / macro-F1 | 0.352 / 0.174 | 0.556 / 0.520 | 0.259 / 0.232 | **0.556 / 0.520** |

Gold check (train on the 48 extended rows, test on the 6 bundled): category 1.00, severity 0.50 (n=6, so very noisy).
Full output and confusion matrices: `docs/evaluation.json` and `/api/evaluation`.

**Reading these honestly**
- ML alone is weak: about 50 rows is too few, and it does worse than the majority baseline on severity. That is why rules carry most of the weight.
- Category numbers are optimistic: I wrote both the lexicons and the extended data, so rules fit the data's vocabulary. Expect lower on real narratives.
- The hybrid weights (0.4 / 0.2) were chosen after looking at this same CV, so they are mildly tuned to it.
- **Severity is the weak point.** The model rarely predicts *medium* (recall 4 of 17); borderline cases collapse to low or high. Severity is also subjective, so a real label set needs EHS reviewers.

## Limitations and failure cases
- Negation and nuance ("no fire", "not injured") are only partly handled; keyword hits can mislead.
- Summaries are extractive templates, not generative; they cannot add facts beyond the narrative.
- Themes and actions are a hand-built library; novel hazards fall back to "route to EHS reviewer".
- Dashboard figures on the bundled corpus use a model trained on that same corpus, so they are descriptive, not a held-out estimate.

## Production-readiness gaps
Real labelled history (hundreds+ of incidents) and inter-rater agreement; sentence-embedding or fine-tuned classifier plus an LLM summariser (API key via environment variable, never in code) with human review; dates, site, asset and contractor fields for true trend analysis; auth, persistence (Postgres), logging and monitoring; model versioning and drift checks; Hindi / regional-language narratives.
