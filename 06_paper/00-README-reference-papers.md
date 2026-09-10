# Reference papers — proactive-feasibility-scheduler

31 PDFs, 33 MB. Downloaded 18 Aug 2026 for the manuscript
*"Ranking Degeneracy: A Learned Wait-Time Model Schedules No Better Than Sorting by Requested Size"*
(`phases_22_30/phase_28_manuscript/manuscript.tex`).

Sources: author-hosted copies (Dror Feitelson's HUJI page, MIT CSAIL), arXiv, and
IEEE Xplore via the VIT Chennai MyLOFT institutional subscription.

---

## Part 1 — The manuscript's 9 bibliography entries (all obtained)

| `\bibitem` key | File |
|---|---|
| `Feitelson1997` | `Feitelson-etal-1997-Theory-and-Practice-in-Parallel-Job-Scheduling.pdf` **+** `Feitelson-2002-Workload-Modeling-for-Performance-Evaluation.pdf` |
| `SLURM2003` | `Jette-Yoo-Grondona-2003-SLURM-Simple-Linux-Utility-for-Resource-Management-JSSPP.pdf` |
| `Jain1984` | `Jain-Chiu-Hawe-1984-Quantitative-Measure-of-Fairness-and-Discrimination-DEC-TR.pdf` |
| `Feitelson2000` | `Feitelson-2001-Metrics-for-Parallel-Job-Scheduling-and-Their-Convergence-JSSPP.pdf` |
| `Feitelson2011` | `Feitelson-Tsafrir-Krakov-2014-Experience-with-Using-the-Parallel-Workloads-Archive-JPDC.pdf` |
| `Mualem2001` | `Mualem-Feitelson-2001-Utilization-Predictability-and-User-Runtime-Estimates-SP2-Backfilling-IEEE-TPDS.pdf` |
| `Tsafrir2005` | `Tsafrir-Etsion-Feitelson-2005-Modeling-User-Runtime-Estimates-JSSPP.pdf` |
| `ML-Scheduling1` | `Mao-Alizadeh-Menache-Kandula-2016-Resource-Management-with-Deep-Reinforcement-Learning-HotNets.pdf` |
| `ML-Scheduling2` | `Mao-etal-2019-Learning-Scheduling-Algorithms-for-Data-Processing-Clusters-SIGCOMM.pdf` |

**Two citation errors to fix in `manuscript.tex` while you have these open:**

1. `Jain1984` is cited as **DEC-TR 506**. The fairness-index report is **DEC-TR-301**
   (Jain, Chiu & Hawe, 1984). Check against the downloaded PDF and correct.
2. `Feitelson2011` is dated 2011 and cited only as the Parallel Workloads Archive URL.
   The citable paper is Feitelson, Tsafrir & Krakov, *"Experience with using the
   Parallel Workloads Archive"*, **JPDC 74(10):2967–2982, 2014**. Use that.

---

## Part 2 — Backfilling and user runtime estimates

The literature your §7 estimate-error finding argues with.

| File | Why it matters to this project |
|---|---|
| `Tsafrir-Etsion-Feitelson-2007-Backfilling-Using-System-Generated-Predictions-Rather-Than-User-Runtime-Estimates-IEEE-TPDS.pdf` | **The single most important related work.** It is the direct predecessor of your idea — replacing user estimates with learned predictions. Your guide will ask how you differ; read this one first. |
| `Nissimov-Feitelson-2007-Probabilistic-Backfilling-JSSPP.pdf` | Distribution-valued rather than point predictions — the alternative your quantile/uncertainty study explores. |
| `Feitelson-2005-Root-Causes-Backfilling-Case-Study-TPDS.pdf` | Why backfilling results are so sensitive to evaluation choices; supports your fidelity caveat. |
| `Shmueli-Feitelson-Backfilling-with-Lookahead.pdf` | Packing-oriented backfill variant. |
| `Shmueli-Feitelson-2005-Adaptive-Backfilling-IPDPS.pdf` | Adaptive backfill. |
| `Talby-Feitelson-1999-Slack-Based-Backfilling-IPPS.pdf` | Priority + slack backfill; relevant to your fairness-budget dial. |
| `IEEE-2010-Using-Historical-Data-to-Predict-Application-Runtimes-on-Backfilling-Parallel-Systems.pdf` | History-based runtime prediction feeding backfill. |
| `IEEE-2023-Deviation-Backfilling-Robust-Backfilling-Scheme.pdf` | Recent robust backfill scheme. |
| `IEEE-2014-Guarantee-Strict-Fairness-and-Utilize-Prediction-Better-in-Parallel-Job-Scheduling.pdf` | Fairness + prediction together — your fairness/tail trade-off. |
| `IEEE-2025-Improving-Runtime-Prediction-Based-on-Application-Type-of-Jobs.pdf` | Per-application-type runtime prediction — **a concrete non-degenerate per-job feature.** |

---

## Part 3 — Queue wait-time prediction (your exact problem)

**Read these before the meeting.** They are the closest prior work to what you built,
and they are where your non-degeneracy condition has the most bite: check whether
each one's feature set contains a genuine per-job attribute, or whether it is
degenerate in the same way yours is.

| File |
|---|
| `IEEE-2019-Queue-Waiting-Time-Prediction-for-Large-scale-HPC-System.pdf` |
| `IEEE-2024-Hierarchical-Deep-Learning-for-Predicting-Job-Queue-Times-in-HPC.pdf` |
| `IEEE-2024-Combining-ML-and-Metaheuristics-for-Predicting-Waiting-Time-of-HPC-Jobs.pdf` |
| `Barsanti-Sodan-2006-Adaptive-Job-Scheduling-via-Predictive-Job-Resource-Allocation-JSSPP.pdf` |

---

## Part 4 — Workload modelling and evaluation methodology

Supports your "the synthetic generator is not a real workload" limitation.

| File |
|---|
| `Lublin-Feitelson-Workload-on-Parallel-Supercomputers-Rigid-Jobs.pdf` (the standard workload model — a principled replacement for `randint(1,8)` / `randint(5,20)`) |
| `Downey-Feitelson-1999-Elusive-Goal-of-Workload-Characterization.pdf` |
| `Feitelson-Shmueli-2009-Conservative-Workload-Modeling-Daily-Cycles-MASCOTS.pdf` (diurnal cycles your generator omits) |
| `IEEE-2024-Cross-System-Analysis-of-Job-Characterization-and-Scheduling.pdf` (cross-machine variation — directly supports your LANL-vs-SDSC divergence) |
| `IEEE-2022-Metrics-for-Packing-Efficiency-and-Fairness-of-HPC-Cluster-Batch-Job-Scheduling.pdf` |

---

## Part 5 — ML for cluster scheduling

| File |
|---|
| `IEEE-2023-Machine-Learning-Feature-Based-Job-Scheduling-for-Distributed-ML-Clusters-ToN.pdf` |
| `Hovestadt-etal-2003-HPC-Resource-Management-Queueing-vs-Planning-JSSPP.pdf` |
| (plus the two Mao papers in Part 1) |

---

## Not obtained — and why

| Wanted | Status |
|---|---|
| Nurmi, Brevik & Wolski, **QBETS: Queue Bounds Estimation from Time Series**, JSSPP 2007 (LNCS 4942) | SpringerLink shows *"Log in via an institution"* — VIT's Springer subscription does not cover this LNCS volume. **Get it via inter-library loan, or email the authors.** This is the classic queue-wait-bound paper and is worth citing. |
| Gaussier, Glesser, Reis & Trystram, **Improving backfilling by using machine learning to predict running times**, SC 2015 | ACM Digital Library is **not** in VIT's MyLOFT subscription list. A free author copy is on HAL (French open archive) — search "hal Gaussier improving backfilling machine learning". |
| Elsevier / ScienceDirect papers (FGCS, JPDC) — several highly relevant hits found, e.g. *"A Machine Learning Approach for an HPC Use Case: the Jobs Queuing Time Prediction"* (FGCS 2023, PII `S0167739X23000274`), *"Topology-aware GPU job scheduling with deep reinforcement learning"* (JPDC, PII `S0743731525001054`) | ScienceDirect returned an **anti-scraping block** against the MyLOFT proxy IP (52.66.29.202). I stopped rather than work around it — pushing further risks getting the whole campus proxy IP-blocked by Elsevier. **Download these manually**: open MyLOFT → ScienceDirect → paste the PII into the search box → click *Download PDF* yourself. Two clicks each. |

---

## Provenance

- Author-hosted / open: `cs.huji.ac.il/~feit` (Feitelson's own publication page, which
  hosts the JSSPP proceedings), `arxiv.org`, `people.csail.mit.edu/alizadeh`.
- Institutional: IEEE Xplore, accessed through MyLOFT with
  *"Access provided by: VIT University - Chennai Campus"* confirmed on-page.
- All 31 files were verified to begin with a valid `%PDF-` header.
- Duplicates from a retried download are parked in `_to_delete/` — delete that folder.
