# ESG Safety Incident Intelligence Dashboard

Turns free-text safety and ESG incident narratives into management-ready insights: category, severity, risk score, a one-line summary, recurring patterns, and recommended preventive actions, complete with the evidence behind each call.

## Live Deployment Link
* **Live Application URL:** [https://esg-safety-intelligence-incident.onrender.com](https://esg-safety-intelligence-incident.onrender.com)
* **Health Check Endpoint:** `https://esg-safety-intelligence-incident.onrender.com/api/health`

---

## Context
Greenko-style renewable-energy sites (solar blocks, wind gearboxes, substations, battery containers, contractor camps), where EHS teams receive numerous short narratives and need immediate visibility into critical issues and recurring trends.

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload        # open http://localhost:8000 (API docs: /docs)
python -m backend.pipeline              # re-run evaluation, writes docs/evaluation.json


## Deploy to Render (Recommended)
1. Push your project folder to a GitHub repository (`git init && git add . && git commit -m "ESG dashboard"`, then push).
2. Go to **Render**, click **New + > Web Service**, and connect your GitHub repository.
3. Configure your service settings:
   * **Environment:** Python 3
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
4. Select the **Free** tier and click **Create Web Service**.
5. Once live, open your Render domain: the dashboard is served at `/`, and API endpoints are available at `/api/*`.
6. Verify that `https://<your-domain>/api/health` returns `{"status":"ok"}`, then include your live link and a screenshot in your final submission. No secrets or API keys are required.

---

## How It Works

| Step | Method |
| :--- | :--- |
| **Preprocessing** | Lowercase, strip punctuation, TF-IDF (1–2 grams, stop-words removed). |
| **Category** (`safety` / `environment` / `esg_compliance`) | Blend of TF-IDF + logistic regression (40%) and an EHS keyword-lexicon prior (60%). |
| **Severity** (`low` / `medium` / `high`) | Blend of logistic regression (20%) and rule signals: critical keywords (*fire, fracture, hospital...*) force high; near-miss, energy/height/vehicle hazards add risk; *"contained"*, *"spill kit"*, *"minor"* subtract. |
| **Risk Score** (0–100) | Expected severity from probabilities (20/55/90), plus +5 if no response action is recorded. |
| **Summary** | Extractive, template-based: severity + category + asset + hazard themes + response recorded. |
| **Patterns / Trends** | 12 hazard themes matched per incident; counts, share, high-severity count. A theme with 3+ incidents is flagged as recurring; top keywords per category are tracked. |
| **Preventive Actions** | Theme-to-corrective-action library, ranked by frequency and severity. |
| **Retrieval** | TF-IDF cosine similarity returns the 3 most similar past incidents. |
| **Explainability** | Top contributing terms, matched rule hits, and severity signals returned with every prediction. |

---

## API Endpoints
* `GET /api/dashboard`
* `GET /api/incidents`
* `POST /api/analyze` (expects `{"text": ...}`)
* `POST /api/batch`
* `POST /api/upload` (CSV with a narrative column)
* `GET /api/evaluation`
* `GET /api/health`

---

## Data and Assumptions
* `data/safety_esg_incidents.csv`: The 6 bundled incidents (used as the gold check).
* `data/safety_esg_incidents_extended.csv`: 48 incidents written and labelled (16 per category) to enable training and validation. Labels reflect typical EHS practice rather than real company records.
* **Assumption:** Severity is judged on potential and actual consequence in the narrative; a "near miss" with energy or vehicle hazards is rated high even without an injury (consistent with S004).
* The dataset contains no dates, so "trends" refers to recurring-pattern frequency rather than time series. Adding a date column enables natural monthly trend analysis.

---

## Evaluation (5-Fold Stratified CV, 54 Incidents)

| Task | Majority | Rules Only | ML Only | Hybrid (Used) |
| :--- | :--- | :--- | :--- | :--- |
| **Category Accuracy / Macro-F1** | 0.315 / 0.244 | 0.981 / 0.981 | 0.611 / 0.615 | **0.963 / 0.962** |
| **Severity Accuracy / Macro-F1** | 0.352 / 0.174 | 0.556 / 0.520 | 0.259 / 0.232 | **0.556 / 0.520** |

* **Gold Check** (train on 48 extended rows, test on 6 bundled): Category accuracy = 1.00, Severity accuracy = 0.50 ($n=6$, highly noisy). Full outputs and confusion matrices are available in `docs/evaluation.json` and via `/api/evaluation`.

### Critical Takeaways
* **ML alone is weak:** ~50 rows is insufficient data, performing worse than the majority baseline on severity. Consequently, rules carry the majority of the weight.
* **Category numbers are optimistic:** Lexicons and extended data were written together, causing rules to closely fit the vocabulary. Real-world performance on unseen narratives will be lower.
* **Tuning:** Hybrid weights (`0.4 / 0.2`) were selected after reviewing this CV, making them mildly tuned to the dataset.
* **Severity is challenging:** The model rarely predicts "medium" (recall 4 of 17); borderline cases collapse to low or high. Severity is inherently subjective, requiring validation by EHS reviewers.

---

## Limitations and Failure Cases
* **Negation and Nuance:** Phrases like "no fire" or "not injured" are only partly handled; keyword hits can occasionally mislead.
* **Extractive Summaries:** Summaries use templates rather than generative models, limiting their ability to introduce facts outside the narrative.
* **Hand-Built Libraries:** Themes and actions rely on a predefined library; novel hazards fall back to "route to EHS reviewer."
* **Descriptive Metrics:** Dashboard figures on the bundled corpus use a model trained on that same corpus, reflecting descriptive analysis rather than a held-out estimate.

---

## Production-Readiness Gaps
* Real labelled historical data (hundreds+ of incidents) and inter-rater agreement studies.
* Sentence-embedding or fine-tuned classifiers paired with an LLM summariser (API key via environment variables, never hardcoded) alongside human review.
* Inclusion of date, site, asset, and contractor fields for authentic trend analysis.
* Authentication, persistence (PostgreSQL), logging, and monitoring infrastructure.
* Model versioning and data drift checks.
* Support for Hindi and regional-language narratives.
