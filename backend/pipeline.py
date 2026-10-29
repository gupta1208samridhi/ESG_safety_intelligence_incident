"""ESG / Safety incident intelligence pipeline.

Hybrid design: TF-IDF + logistic regression (learns from data) blended with
domain rules (lexicons, severity signals) so it works on tiny data and is explainable.
"""
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATS = ["safety", "environment", "esg_compliance"]
SEVS = ["low", "medium", "high"]
W_CAT, W_SEV = 0.4, 0.2  # hybrid weight on the ML model per task; rules get 1 - w (ML is weak on ~50 rows, esp. severity)

# ---------------------------------------------------------------- domain rules
CAT_LEX = {
    "safety": r"injur|slip|fell|fall|ppe|helmet|harness|near miss|vehicle|forklift|crane|shock|electric|arc flash|fire|smoke|lockout|height|ladder|guardrail|trip|first aid|unauthori[sz]ed|heat",
    "environment": r"oil|spill|seepage|leak|drip|diesel|water|effluent|emission|contaminat|soil|discharge|bird|dust|noise|runoff|silt|vegetation|chemical|sf6|electrolyte|valve",
    "esg_compliance": r"waste|segregat|audit|permit|licen[cs]e|policy|grievance|labou?r|wage|certificate|record|register|manifest|induction|overdue|expired|statutory|csr|clearance|consent|documentation|report|ledger",
}
HARD_HIGH = r"\bfatal|\bdied\b|\bdeath|explosion|electrocut|\bfire\b(?! extinguisher)|collapse|fracture|hospital|underage"
SEV_W = {  # presence-based signal weights (each counted once)
    r"near miss": 3,
    r"shock|arc flash|unauthori[sz]ed|lockout|height|harness|forklift|vehicle|crane|swung|slid|sf6|electrolyte|hazardous": 2,
    r"injur|fell|slip|leak|spill|seepage|contaminat|manifest|expired|grievance|unpaid|delayed|oil|diesel|battery|overflow": 1,
    r"contained|spill kit|bund|absorbent|wiped|first aid|no injury|minor|verbal warning|housekeeping|recovered|closed|removed|cordoned": -1,
    r"records|register|display|board|ledger|recycling|incomplete|not logged": -1,
}
SEV_BUCKETS = [(0, [0.70, 0.25, 0.05]), (2, [0.20, 0.60, 0.20]), (99, [0.05, 0.25, 0.70])]

THEMES = {
    "housekeeping_slips": (r"slip|trip|wet|muddy|walkway|housekeeping|loose cable", "Housekeeping / slip-trip hazard",
        ["Daily housekeeping walkdown with sign-off", "Anti-slip surfacing and cable management on walkways"]),
    "ppe": (r"ppe|helmet|harness|glove", "PPE non-compliance",
        ["PPE check at site gate and before each permit", "Stop-work and re-induction for repeat PPE violations"]),
    "vehicle_lifting": (r"vehicle|forklift|reversing|crane|tanker|banksman|tag line", "Vehicle / lifting interface",
        ["Segregate pedestrian and vehicle routes; enforce banksman", "Lift plans with tag lines and exclusion zones"]),
    "work_at_height": (r"ladder|height|fell|guardrail|harness", "Work at height",
        ["Height-work permit with anchor-point verification", "Install permanent guardrails at open edges"]),
    "electrical": (r"electric|shock|arc|breaker|lockout|tagout|live |switchyard|cord", "Electrical safety / isolation",
        ["Enforce LOTO with supervisor verification", "Inspect and tag portable electrical equipment monthly"]),
    "fire_battery": (r"fire|smoke|battery|electrolyte", "Fire / battery hazard",
        ["Thermal and smoke monitoring on battery containers", "Quarterly emergency evacuation drills"]),
    "oil_chemical": (r"oil|spill|seepage|diesel|leak|drip|chemical|sf6|electrolyte", "Oil / chemical release",
        ["Secondary containment and drip trays at all oil points", "Spill kits at each bay plus quarterly spill drill"]),
    "water_use": (r"water|valve|overflow|borewell|runoff|silt", "Water use / runoff",
        ["Flow meters and monthly water budget per site", "Preventive maintenance of valves and tanker fill controls"]),
    "waste_mgmt": (r"waste|segregat|recycl|manifest|drum", "Waste management",
        ["Colour-coded bins with weekly contractor audit", "Reconcile hazardous-waste manifests monthly"]),
    "contractor_mgmt": (r"contractor|labou?r|induction|camp", "Contractor management",
        ["Contractor pre-qualification and induction gate-check", "Include ESG KPIs in contractor contracts"]),
    "documentation": (r"record|register|certificate|licen[cs]e|permit|report|logged|documentation|overdue|expired|ledger|consent|manifest", "Records / statutory documentation",
        ["Compliance calendar with owner and 30-day expiry alerts", "Quarterly internal audit of statutory records"]),
    "community_biodiversity": (r"village|bird|noise|dust|grazing|vegetation|community", "Community / biodiversity impact",
        ["Community grievance log with 7-day closure target", "Pre-work biodiversity and boundary checks"]),
}
ASSETS = r"solar block|wind gearbox|gearbox|cable trench|substation|transformer|battery container|switchyard|switchgear room|control room|turbine|nacelle|inverter|tracker|stores|workshop|contractor camp|storage area|access road|site office|generator|walkway"
RESPONSE = r"spill kit|contained|first aid|stopped|warning|evacuat|cordoned|closed|removed|applied|hospital|ventilated|wiped|absorbent|hazmat|reported|recovered"
REPEAT = r"repeated|repeat|again|recurr|second time"


def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def rule_cat(c):
    s = np.array([len(re.findall(p, c)) for p in CAT_LEX.values()], float) + 0.5
    return s / s.sum()


def rule_sev(c):
    m = re.search(HARD_HIGH, c)
    if m:
        return np.array([0.0, 0.05, 0.95]), [f"critical keyword '{m.group(0).strip()}'"]
    score, sig = 0, []
    for p, w in SEV_W.items():
        m = re.search(p, c)
        if m:
            score += w
            sig.append(f"{m.group(0)} ({w:+d})")
    for limit, proba in SEV_BUCKETS:
        if score <= limit:
            return np.array(proba), sig
    return np.array(SEV_BUCKETS[-1][1]), sig


def _align(p, classes, order):
    cols = {c: i for i, c in enumerate(classes)}
    return np.column_stack([p[:, cols[o]] if o in cols else np.zeros(len(p)) for o in order])


def load_data():
    frames = [pd.read_csv(DATA / f) for f in ("safety_esg_incidents.csv", "safety_esg_incidents_extended.csv") if (DATA / f).exists()]
    return pd.concat(frames, ignore_index=True).drop_duplicates("narrative").reset_index(drop=True)


# ------------------------------------------------------------------- model
class Triage:
    def fit(self, df):
        self.df = df.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.X = self.vec.fit_transform(self.df.narrative.map(clean))
        self.cat = LogisticRegression(max_iter=2000, C=10).fit(self.X, self.df.target_category)
        self.sev = LogisticRegression(max_iter=2000, C=10).fit(self.X, self.df.target_severity)
        return self

    def probs(self, texts, mode="hybrid"):
        cl = [clean(t) for t in texts]
        X = self.vec.transform(cl)
        pc = _align(self.cat.predict_proba(X), self.cat.classes_, CATS)
        ps = _align(self.sev.predict_proba(X), self.sev.classes_, SEVS)
        rc = np.array([rule_cat(c) for c in cl])
        rs = np.array([rule_sev(c)[0] for c in cl])
        wc, ws = {"ml": (1.0, 1.0), "rules": (0.0, 0.0), "hybrid": (W_CAT, W_SEV)}[mode]
        return wc * pc + (1 - wc) * rc, ws * ps + (1 - ws) * rs, X, cl

    def labels(self, texts, mode="hybrid"):
        pc, ps, _, _ = self.probs(texts, mode)
        return [CATS[i] for i in pc.argmax(1)], [SEVS[i] for i in ps.argmax(1)]

    def _terms(self, x, clf, label):
        contrib = x.multiply(clf.coef_[list(clf.classes_).index(label)]).toarray().ravel()
        names = self.vec.get_feature_names_out()
        return [names[i] for i in np.argsort(-contrib)[:4] if contrib[i] > 0]

    def analyze(self, texts, exclude_identical=False):
        pc, ps, X, cl = self.probs(texts)
        sims = (X @ self.X.T).toarray()  # tf-idf rows are L2-normalised -> cosine
        out = []
        for i, text in enumerate(texts):
            cat, sev = CATS[pc[i].argmax()], SEVS[ps[i].argmax()]
            c = cl[i]
            themes = [k for k, (p, _, _) in THEMES.items() if re.search(p, c)]
            asset = (re.search(ASSETS, c) or [None])[0]
            resp = sorted(set(re.findall(RESPONSE, c)))
            expected = float(ps[i] @ np.array([20, 55, 90]))
            risk = int(min(100, round(expected + (0 if resp else 5))))
            order = np.argsort(-sims[i])
            similar = [
                {"incident_id": self.df.incident_id[j], "narrative": self.df.narrative[j], "category": self.df.target_category[j],
                 "severity": self.df.target_severity[j], "similarity": round(float(sims[i][j]), 2)}
                for j in order if not (exclude_identical and sims[i][j] > 0.999) and sims[i][j] > 0.15
            ][:3]
            label = lambda k: THEMES[k][1]
            summary = (
                f"{sev.upper()}-severity {cat.replace('_', ' ')} incident"
                + (f" at/near {asset}" if asset else "")
                + (f" involving {', '.join(label(k).lower() for k in themes[:2])}" if themes else "")
                + f". Response recorded: {', '.join(resp) if resp else 'none - verify controls'}."
            )
            out.append({
                "narrative": str(text), "category": cat, "severity": sev, "risk_score": risk,
                "category_probs": {k: round(float(v), 2) for k, v in zip(CATS, pc[i])},
                "severity_probs": {k: round(float(v), 2) for k, v in zip(SEVS, ps[i])},
                "summary": summary, "asset": asset, "response": resp,
                "themes": [{"id": k, "label": label(k)} for k in themes],
                "actions": [a for k in themes[:3] for a in THEMES[k][2][:2]],
                "repeat_marker": bool(re.search(REPEAT, c)),
                "evidence": {
                    "category_terms": self._terms(X[i], self.cat, cat),
                    "category_rule_hits": sorted(set(re.findall(CAT_LEX[cat], c)))[:5],
                    "severity_terms": self._terms(X[i], self.sev, sev),
                    "severity_signals": rule_sev(c)[1],
                },
                "similar_incidents": similar,
            })
        return out


@lru_cache(maxsize=1)
def get_model():
    return Triage().fit(load_data())


# ------------------------------------------------------- dashboard aggregation
def summarize(rows):
    """Aggregate analysed incidents into dashboard outputs: KPIs, patterns, priority actions."""
    n = len(rows)
    cats, sevs = Counter(r["category"] for r in rows), Counter(r["severity"] for r in rows)
    matrix = {c: {s: sum(1 for r in rows if r["category"] == c and r["severity"] == s) for s in SEVS} for c in CATS}
    th = defaultdict(lambda: {"count": 0, "high": 0, "cats": Counter()})
    for r in rows:
        for t in r["themes"]:
            d = th[t["id"]]
            d["count"] += 1
            d["high"] += r["severity"] == "high"
            d["cats"][r["category"]] += 1
    patterns = sorted(
        [{"id": k, "label": THEMES[k][1], "count": v["count"], "high": v["high"], "share_pct": round(100 * v["count"] / max(n, 1)),
          "top_category": v["cats"].most_common(1)[0][0], "recurring": v["count"] >= 3, "actions": THEMES[k][2]}
         for k, v in th.items()], key=lambda x: (-x["count"], -x["high"]))
    keywords = {}
    for c in CATS:
        texts = [clean(r["narrative"]) for r in rows if r["category"] == c]
        try:
            cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)
            m = cv.fit_transform(texts).sum(0).A1
            keywords[c] = [cv.get_feature_names_out()[i] for i in np.argsort(-m)[:6]]
        except ValueError:
            keywords[c] = []
    top = sorted(rows, key=lambda r: -r["risk_score"])[:5]
    priority = sorted(patterns, key=lambda p: -(p["count"] + 2 * p["high"]))[:4]
    return {
        "kpis": {"total": n, "high_severity": sevs["high"], "high_pct": round(100 * sevs["high"] / max(n, 1)),
                 "avg_risk": round(float(np.mean([r["risk_score"] for r in rows])), 1) if n else 0,
                 "recurring_patterns": sum(p["recurring"] for p in patterns)},
        "by_category": {c: cats[c] for c in CATS}, "by_severity": {s: sevs[s] for s in SEVS},
        "category_severity_matrix": matrix, "patterns": patterns, "keywords_by_category": keywords,
        "top_risks": [{k: r[k] for k in ("narrative", "category", "severity", "risk_score", "summary")} | {"incident_id": r.get("incident_id")} for r in top],
        "priority_actions": [{"theme": p["label"], "incidents": p["count"], "actions": p["actions"]} for p in priority],
    }


def corpus_dashboard():
    m = get_model()
    df = m.df
    rows = m.analyze(df.narrative.tolist(), exclude_identical=True)
    for r, iid in zip(rows, df.incident_id):
        r["incident_id"] = iid
    return summarize(rows), rows


# ------------------------------------------------------------------ evaluation
def _scores(y, p, labels):
    return {"accuracy": round(accuracy_score(y, p), 3), "macro_f1": round(f1_score(y, p, average="macro", labels=labels, zero_division=0), 3)}


@lru_cache(maxsize=1)
def evaluate(seed=7):
    df = load_data()
    res = {"n_incidents": len(df), "protocol": "5-fold stratified CV on bundled + extended data; plus gold check (train on extended only, test on the 6 bundled incidents)"}
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, col, labels in (("category", "target_category", CATS), ("severity", "target_severity", SEVS)):
        y = df[col].values
        preds = {k: np.empty(len(df), dtype=object) for k in ("majority", "rules", "ml", "hybrid")}
        for tr, te in skf.split(df, y):
            m = Triage().fit(df.iloc[tr])
            preds["majority"][te] = pd.Series(y[tr]).mode()[0]
            for mode in ("rules", "ml", "hybrid"):
                preds[mode][te] = m.labels(df.narrative.iloc[te].tolist(), mode)[0 if name == "category" else 1]
        res[name] = {k: _scores(y, p, labels) for k, p in preds.items()}
        res[name]["labels"] = labels
        res[name]["confusion_matrix_hybrid"] = confusion_matrix(y, preds["hybrid"], labels=labels).tolist()
    gold = df[df.incident_id.str.startswith("S")]
    m = Triage().fit(df[~df.incident_id.str.startswith("S")])
    gc, gs = m.labels(gold.narrative.tolist(), "hybrid")
    res["gold_check_bundled_6"] = {"category_accuracy": round(accuracy_score(gold.target_category, gc), 3),
                                    "severity_accuracy": round(accuracy_score(gold.target_severity, gs), 3), "n": len(gold)}
    return res


if __name__ == "__main__":
    r = evaluate()
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "evaluation.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
