# Project Abstract

## ShiftGuard: A Human-in-the-Loop Framework for Distribution Shift Detection, Attribution, and Adaptive Retraining in Non-Stationary Forex Markets

### Team Members

- **Sohan Mahesh** — mahesh.so@northeastern.edu
- **Anusha Ravi Kumar** — ravikumar.anu@northeastern.edu
- **Dishaben Manubhai Patel** — patel.dishabe@northeastern.edu

---

### Problem Statement

Distribution shift is one of the most consequential and least solved challenges in applied machine learning. Forex markets present an unusually rich environment to study this problem: they operate continuously across global sessions, are driven by sovereign-level forces such as central bank policy and geopolitical events, and experience shifts at every scale — from catastrophic disruptions like COVID-19 and Brexit, to routine but impactful changes caused by scheduled macro releases (CPI reports, Non-Farm Payroll), interest rate differentials, and news flow. Critically, these shifts differ in timing, magnitude, and feature signatures, yet almost no existing system treats them differently.

### Motivation and Related Work

The relevant literature spans three areas that have developed largely in parallel:

1. **Concept Drift Detection** — Methods such as ADWIN, DDM, KS, and MMD (Gama et al., 2014; Lu et al., 2019) can identify when a data distribution has changed, but treat all shifts uniformly and provide no explanation of what changed or why.
2. **Domain Adaptation** — Approaches like domain-adversarial training (Ganin et al., 2016) focus on closing the gap between source and target distributions, but do so automatically without distinguishing shift types.
3. **Human-in-the-Loop ML** — Work by Monarch (2021) and Amershi et al. (2014) demonstrates that selective human feedback can meaningfully improve model performance in ambiguous cases, but this idea has rarely been applied to financial time series.

ShiftGuard is motivated by the intuition that combining event-aware detection, interpretable attribution, and optional human oversight into a single pipeline is both technically feasible and practically valuable in a way that none of these approaches achieves alone.

### Proposed Approach

ShiftGuard is an end-to-end ML pipeline with three core components:

1. **Dual-Mode Detection Engine** — Separates scheduled event-driven shifts (e.g., macro data releases) from unexpected anomalies (e.g., geopolitical crises) using statistical tests (KS, MMD, ADWIN, DDM) combined with an economic event calendar.
2. **SHAP-Based Attribution Layer** — Traces each detected shift back to the specific feature groups (price-based, volatility, momentum, macro indicators) that changed, providing interpretable explanations for why the shift occurred.
3. **Human-in-the-Loop Dashboard** — A Streamlit interface where users can review shift alerts, inspect feature attributions, confirm or reject detections, and trigger selective model retraining on approved shifts.

### Significance

The significance of this work extends beyond forex. Any domain where distribution shift is frequent, varied in nature, and consequential — whether in healthcare, climate modeling, or supply chain forecasting — faces the same fundamental problem. ShiftGuard demonstrates that a more structured, interpretable, and human-assisted approach to handling distribution shift is both achievable and worth building.

---

### Scope of Work

The project will deliver:

1. **Data pipeline** — Collection and preprocessing of forex OHLCV and macroeconomic indicator data (CPI, NFP, interest rates) across major currency pairs.
2. **Shift detection module** — Statistical detectors (KS, MMD, ADWIN, DDM) integrated with an economic event calendar for dual-mode (scheduled vs. anomalous) shift detection.
3. **Attribution module** — SHAP-based feature attribution that identifies which feature groups drove each detected shift.
4. **Human-in-the-loop dashboard** — Streamlit interface for reviewing alerts, inspecting attributions, and triggering selective model retraining.
5. **Evaluation** — Benchmarking shift detection recall and precision against historical labeled events (e.g., COVID-19, Brexit, major macro releases).

---

### Split of Work

| Team Member | Responsibilities |
|---|---|
| Sohan Mahesh | Shift detection module (statistical tests + event-aware logic), evaluation framework |
| Anusha Ravi Kumar | SHAP attribution layer, Streamlit dashboard, retraining integration |
| Dishaben Manubhai Patel | Data pipeline (collection, feature engineering, preprocessing), EDA |

All members will contribute to integration, testing, and the final writeup.

---

### References

- Gama, J., et al. (2014). A survey on concept drift adaptation.
- Lu, J., et al. (2019). Learning under concept drift: A review.
- Ganin, Y., et al. (2016). Domain-adversarial training of neural networks.
- Monarch, R. (2021). *Human-in-the-Loop Machine Learning*.
- Amershi, S., et al. (2014). Power to the people: The role of humans in interactive machine learning.
