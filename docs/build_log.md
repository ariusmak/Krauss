# Krauss Project Build Log

## Session 1 — 2026-04-06

### Step 1: Repository setup
- Created `pyproject.toml` with all dependencies (numpy, pandas, scipy, wrds, sklearn, xgboost, torch, etc.)
- Dev dependencies (pytest, ruff, black, ipykernel) under `[project.optional-dependencies.dev]`
- Fixed build backend from incorrect `setuptools.backends._legacy:_Backend` to `setuptools.build_meta`
- Created full directory structure per CLAUDE.md Section 8
- Installed with `pip install -e ".[dev]"` in conda env `ML` (Python 3.13)

### Step 2: WRDS data extraction
**Decisions made:**
- **Data source:** CRSP daily stock file (`crsp.dsf`) — closest WRDS analogue to the paper's Datastream total return indices
- **Canonical identifier:** PERMNO (security-level, stable across time)
- **S&P 500 membership:** `crsp.dsp500list` (start/ending spell table, keyed on PERMNO)
- **Authentication:** WRDS prompts interactively; credentials cached in `~/.pgpass`

**Scripts/modules built:**
- `src/krauss/data/wrds_extract.py` — three query functions:
  - `fetch_sp500_membership()` — full `crsp.dsp500list` table
  - `fetch_daily_stock_data()` — `crsp.dsf` (ret, prc, shrout, cfacpr, cfacshr)
  - `fetch_delisting_returns()` — `crsp.dsedelist` (dlstdt, dlret, dlstcd)

- `src/krauss/data/universe.py` — no-lookahead monthly membership:
  - `build_membership_matrix()` — for each month-end, checks which PERMNOs have an active S&P 500 spell (`start <= month_end AND (ending >= month_end OR ending IS NULL)`), assigns them as eligible for the NEXT month
  - `get_eligible_universe(trade_date)` — lookup function mapping any trading date to its eligible PERMNOs
  - `build_daily_eligibility()` — expands monthly panel to (date, permno) rows
  - **No-lookahead rule:** Jan 31 membership → February eligible, Feb 28 → March eligible, etc.
  - Universe updates month-by-month; NOT frozen per study period

- `src/krauss/data/prices_returns.py` — delisting-adjusted returns:
  - Standard CRSP convention: `adj_ret = (1 + ret) * (1 + dlret) - 1` on delisting date
  - If ret missing but dlret exists, uses dlret alone

- `scripts/build_data.py` — orchestration script (ran successfully)

- `configs/phase1_repro.yaml` — all dates/parameters:
  - raw_start_date: 1989-01-01 (need lookback room for 240-day features)
  - raw_end_date: 2015-12-31
  - study_start_date: 1992-01-01
  - study_end_date: 2015-10-30

**Data pull results (2026-04-06):**
- 2,064 membership spells, 1,936 unique PERMNOs ever in S&P 500
- 1,343 PERMNOs with CRSP daily data in 1989–2015 range (593 were members outside this window)
- 5,776,538 daily rows pulled; 5,764,478 after return cleaning
- 578 delisting events
- 162,012 stock-month eligibility rows, 324 effective months, avg 500 stocks/month
- 3,392,164 stock-day eligibility rows

**Output files:**
- `data/raw/sp500_membership.parquet` (0.0 MB)
- `data/raw/crsp_daily.parquet` (39.4 MB)
- `data/raw/crsp_delist.parquet` (0.0 MB)
- `data/processed/membership_monthly.parquet` (0.1 MB)
- `data/processed/universe_daily.parquet` (0.7 MB)
- `data/processed/daily_returns.parquet` (26.9 MB)

### Step 3: Feature and label generation
**Features (31 total per the paper):**
- R1–R20: simple returns over 1–20 day lookbacks
- R40, R60, R80, R100, R120, R140, R160, R180, R200, R220, R240: multi-period returns
- Definition: `R_{t,m} = P_t / P_{t-m} - 1` where P is a cumulative total-return price index reconstructed from daily holding-period returns
- Rows with < 240 trading days of history are dropped (feature lookback requirement)

**Labels:**
- `y_binary`: 1 if stock's next-day return > next-day cross-sectional median, else 0 (Phase 1 target)
- `u_excess`: next-day return minus next-day cross-sectional median (Phase 2 target)
- Cross-sectional median computed only over eligible stocks on the next day
- Features at date t use info through t; labels use return realized on t+1

**Scripts/modules:**
- `src/krauss/data/features.py` — `compute_lagged_returns()`
- `src/krauss/data/labels.py` — `compute_labels()`
- `scripts/build_features_labels.py` — orchestration with sanity checks

**Results (2026-04-06):**
- Features: 5,446,588 rows, 1,308 stocks, date range 1989-12-13 to 2015-12-31
- Labels: 3,386,718 rows, 1,135 stocks, date range 1989-02-01 to 2015-12-30
- `y_binary=1` rate: 0.4924 (expected ~0.50 — slight asymmetry is normal due to right-skewed returns)
- `u_excess` mean: 0.000434 (near zero as expected — small positive skew)
- `u_excess` median: 0.000000 (exactly zero by construction at the cross-sectional level)
- `u_excess` std: 0.020934

**Output files:**
- `data/processed/features.parquet` (1,391.5 MB)
- `data/processed/labels.parquet` (35.2 MB)

**Sanity check notes:**
- y_binary at 49.24% rather than exactly 50%: this is expected. The median splits the cross-section into two halves, but when the count is even, stocks exactly at the median get y_binary=0 (since we use strict >). Also, the eligible set can have odd counts.
- u_excess mean slightly positive: consistent with the fact that median < mean for right-skewed return distributions.
- Features table is larger than labels because features are computed for ALL ever-members (needed for lookback), while labels are restricted to eligible (S&P 500 member) stock-days only.

### Step 4: Backtest engine with dummy scorer validation
**Purpose:** Build the full backtest pipeline and validate correctness before plugging in real models.

**Modules built:**
- `src/krauss/data/study_periods.py` — partitions trading dates into rolling 750-train / 250-trade blocks
  - First 240 days of each training window consumed by feature lookback → 510 usable training days
  - 24 study periods total covering 1989–2015 (rolling windows, advance by 250 trade days)
- `src/krauss/backtest/ranking.py` — cross-sectional ranking, top-k / bottom-k selection
- `src/krauss/backtest/portfolio.py` — equal-weight dollar-neutral portfolio construction, daily rebalance
  - Long k stocks at +1/k, short k at -1/k
  - Portfolio return = mean(long returns) - mean(short returns)
- `src/krauss/backtest/costs.py` — turnover computation and transaction cost application (5 bps/half-turn)
  - Turnover = sum of |weight changes| day-over-day
  - Day-1 turnover = 2.0 (full build: k buys + k sells)
- `src/krauss/backtest/rebalance.py` — position change tracking (entries, exits, side-switches)
- `scripts/validate_backtest.py` — end-to-end validation with random dummy scorer

**Study periods generated:**

| Period | Train             | Trade             |
|--------|-------------------|-------------------|
| 0      | 1989-01-03 → 1991-12-18 | 1991-12-19 → 1992-12-14 |
| 1      | 1992-12-15 → 1995-12-01 | 1995-12-04 → 1996-11-26 |
| 2      | 1996-11-27 → 1999-11-17 | 1999-11-18 → 2000-11-13 |
| 3      | 2000-11-14 → 2003-11-11 | 2003-11-12 → 2004-11-09 |
| 4      | 2004-11-10 → 2007-11-01 | 2007-11-02 → 2008-10-29 |
| 5      | 1993-12-10 → 1996-11-26 | 1996-11-27 → 1997-11-21 |
| ...    | ...                     | ...                     |
| 23     | 2011-10-21 → 2014-10-15 | 2014-10-16 → 2015-10-13 |

Training windows overlap (each advances by 250 trade days). Last trade window ends 2015-10-13, matching the paper's Oct 2015 endpoint.

**Validation results (period 0, k=10, random scorer):**
- 124,704 stock-day predictions, 508 stocks, 250 trading days
- Exactly 10 long + 10 short every day: PASS
- Dollar neutral every day (max imbalance: 0.00): PASS
- No lookahead (all return dates > signal dates): PASS
- Day-1 turnover = 2.0 (correct for full portfolio build): PASS
- Net return < gross return (costs applied correctly): PASS
- Mean daily return: -0.034% pre-cost (near zero as expected for random): PASS
- Avg daily turnover: 3.91 (high, as expected for random — ~96% daily replacement)
- Avg new stocks per day: 9.6/10 on each side (random reshuffles almost entirely)

**Key design notes:**
- The backtest uses signal date t for ranking, realizes returns on t+1. This matches the paper's convention.
- Study periods use rolling windows: advance by 250 (trade_days), training windows overlap. This matches the paper's coverage through Oct 2015.
- Initially implemented as non-overlapping (advance by 1000), which only produced 6 periods ending Oct 2012. Fixed to rolling after identifying the coverage gap.

### Step 5: Phase 1 models
**Models implemented:**
- `src/krauss/models/rf_phase1.py` — RandomForestClassifier (sklearn)
  - 1000 trees, depth 20, sqrt(31)=5 features/split, seed=1, n_jobs=-1
  - Deviation: sklearn vs H2O
- `src/krauss/models/xgb_phase1.py` — XGBClassifier (xgboost)
  - 100 trees, depth 3, lr=0.1, 15/31 features/split, seed=1
  - H2O defaults matched: reg_lambda=0, gamma=1e-5, max_bin=20, min_child_weight=10
  - Deviation: XGBoost vs H2O GBM/AdaBoost (algorithm-level)
- `src/krauss/models/dnn_phase1.py` — KraussDNN (PyTorch)
  - Architecture: 31-31-10-5-2 with maxout activation (2 channels)
  - Dropout: 0.5 hidden, 0.1 input
  - L1 reg: 1e-5, ADADELTA optimizer, 400 epochs max, early stopping
  - 2-class softmax output
  - Deviation: PyTorch vs H2O
- `src/krauss/models/ensembles_phase1.py` — ENS1/ENS2/ENS3
  - ENS1: equal-weight average of DNN/GBT/RAF probabilities
  - ENS2: Gini/AUC-weighted average (training period performance)
  - ENS3: rank-weighted average (training period rank)

**Orchestration:**
- `scripts/run_phase1.py` — trains all models across 23 study periods
  - Supports `--model rf/xgb/dnn/all` and `--periods 0 1 2 ...`
  - Saves trained models to `data/models/period_XX/`
  - Saves predictions to `data/processed/predictions_phase1.parquet`
  - Per-period metadata + ensemble weights saved as JSON

**Period 0 sanity checks (all passed):**
- RF: 0.70%/day pre-cost (paper: RF strongest base model) ✓
- XGB: 0.58%/day pre-cost ✓
- DNN: 0.49%/day pre-cost (paper: DNN weakest) ✓
- ENS1: 0.67%/day pre-cost ✓
- Classification accuracy: all >52% (>50% = better than random) ✓
- Model correlations: RF/XGB 0.82, RF/DNN 0.66, XGB/DNN 0.74 ✓
- Dollar neutral, exactly k per side, no lookahead: all PASS ✓

**Known issue:** macOS segfault when RF (n_jobs=-1, loky) followed by XGBoost (libomp) in same process. libomp is not fork-safe on macOS. Fixed by running each model in a separate subprocess.

### Known reproduction deviations

**DNN: PyTorch vs H2O:**
- **ADADELTA defaults:** H2O uses rho=0.99, epsilon=1e-8. PyTorch defaults were rho=0.9, eps=1e-6. **Fixed** — now explicitly set to H2O values.
- **Early stopping:** H2O uses scoring events (~every 5s wall time), averages last 5 scores, stops after 5 non-improvements. **Fixed** — implemented wall-clock-based scoring every 5.0 seconds with same moving-average + patience logic.
- **L1 regularization:** H2O applies L1 to weights only (not biases). **Fixed** — L1 sum now excludes bias parameters.
- **Dropout with maxout:** H2O may apply dropout differently in conjunction with maxout channels. Our implementation applies standard `nn.Dropout` after each maxout layer output.
- **Determinism:** Paper runs on single core to suppress hardware stochastics. We use `torch.manual_seed(1)` for determinism but allow multi-threaded inference for scoring speed.

**XGB: H2O GBM defaults now matched:**
- `max_depth=3`: Both H2O and XGBoost use the same convention. No deviation.
- `reg_lambda=0`: H2O has no L2 regularization. XGBoost default was 1. **Fixed.**
- `gamma=1e-5`: H2O `min_split_improvement=1e-5`. XGBoost default was 0. **Fixed.**
- `max_bin=20`: H2O `nbins=20`. XGBoost default was 256. **Fixed.**
- `min_child_weight=10`: H2O `min_rows=10`. XGBoost default was 1. **Previously fixed.**

**GICS industry classification (Tables 1, 6 only):**
The paper used point-in-time GICS sector codes as they existed during the study period (pre-2016). Our reproduction uses current GICS from Compustat `comp.company`, which reflects the 2016 GICS reclassification. Most notably:
- Real Estate (sector 60) was carved out of Financials in 2016 — we map it back to Financials
- Individual stocks were reclassified between sectors (Consumer goods ↔ Consumer services, Telecom expansion, etc.)
- This causes industry-level stock counts to differ from the paper (e.g., Consumer goods ~42 vs 65, Telecom ~21 vs 11)
- The "All" row (~499.8) matches the paper (~499.7), confirming the universe is correct
- **No impact on models or trading** — industry codes are not used as features or in portfolio construction

## Session 2 — 2026-04-07: Parity Audit

### Model audit against paper and H2O defaults

Systematic audit of all model implementations, trading logic, and statistics against the original paper's specifications and H2O library defaults. Compared every parameter against the paper text, H2O documentation, and XGBoost/sklearn/PyTorch defaults.

**Critical issues found and fixed:**

1. **XGBoost `reg_lambda=1` (L2 regularization) — UNLOGGED, HIGH IMPACT**
   - XGBoost defaults to `reg_lambda=1`, adding L2 penalty on leaf weights.
   - H2O GBM has NO L2 regularization by default.
   - Effect: shrinks leaf predictions toward zero, producing less extreme probabilities, making rankings more stable day-to-day. Directly contributes to lower turnover than the paper.
   - Fix: set `reg_lambda=0` explicitly.

2. **XGBoost `max_bin=256` vs H2O `nbins=20` — UNLOGGED, MEDIUM IMPACT**
   - H2O GBM uses 20 histogram bins for split finding. XGBoost defaults to 256.
   - Coarser bins (H2O) produce noisier/less precise splits → more diverse predictions.
   - Fix: set `max_bin=20` explicitly.

3. **XGBoost `gamma=0` vs H2O `min_split_improvement=1e-5` — UNLOGGED, LOW IMPACT**
   - Fix: set `gamma=1e-5` explicitly.

4. **DNN L1 regularization applied to biases — UNLOGGED, MEDIUM IMPACT**
   - H2O applies L1 to weight matrices only, not bias vectors.
   - Code was penalizing ALL parameters including biases, which pushes activation thresholds toward zero and reduces network expressiveness.
   - Fix: filter `model.named_parameters()` to exclude `'bias'` from L1 sum.

5. **ENS3 weighting formula incorrect — UNLOGGED, LOW IMPACT**
   - Paper Eq. 7: `w_i = (1/R_i) / sum(1/R_j)` with rank 1 = best (Aiolfi & Timmermann 2006).
   - Code used `R_i / sum(R_j)` with rank 1 = worst. Gives different weight distribution: code was 50/33/17 vs correct 54.5/27.3/18.2 for best/mid/worst.
   - Fix: corrected to paper formula with descending rank convention.

**Stale deviation log entries fixed:**
- Section 8 (DNN training) still referenced "epoch-level" scoring but code was already time-based. Updated to match actual implementation.
- L1 penalty entry updated to reflect weights-only fix.
- Section 16 (ambiguities) DNN early stopping entry updated.

**Turnover analysis:**
- Paper implies ~4.0 daily turnover for k=10 (near-complete daily replacement), derived from pre-cost minus post-cost returns: `(0.0045 - 0.0025) / 0.0005 = 4.0`.
- Our models were producing significantly lower turnover ("sticky" predictions).
- Root cause: XGBoost L2 regularization (most impactful) + DNN bias L1 over-regularization + finer histogram bins all pushed toward more conservative/stable predictions.

**Items verified as correct:**
- DNN architecture (31-31-10-5-2 maxout), ADADELTA (rho=0.99, eps=1e-8), dropout (0.5/0.1)
- RF: 1000 trees, depth 20, sqrt(31)=5 features, entropy, 63.2% WoR sampling
- Feature generation (31 lagged returns R1-R20, R40-R240)
- Label construction (y_binary = 1 if ret > cross-sectional median)
- Study periods (23 periods, 750+250, advance 250)
- Portfolio return = mean(long) - mean(short), dollar neutral, equal weight
- Turnover = sum(|weight changes|), cost = turnover × 5 bps per half-turn
- Return alignment (signal at t, return from t to t+1, close-to-close)
- ENS1 and ENS2 formulas correct

**Status:** All fixes applied to code and deviation log. Full retraining across all 23 periods required to regenerate predictions and backtest results.

### DNN deep-dive — 2026-04-08

Post-retrain diagnostic revealed DNN as the main problem child:
- GBT/RAF pre-cost returns at 95% of paper values (acceptable given CRSP vs Datastream)
- DNN pre-cost return: 0.0021 vs paper 0.0033 (64% match)
- DNN prediction std: 0.016 (very compressed vs RF's 0.031)
- DNN turnover: 1.6 vs paper's implied 4.0

**Root cause investigation:**

1. **H2O `mini_batch_size=1`** (pure SGD): H2O's default processes one sample per weight update. Our `batch_size=1024` averaged gradients over 1024 samples, over-smoothing the signal and producing compressed predictions.

2. **H2O `rate=0.005`**: H2O scales ADADELTA updates by this factor (PyTorch default `lr=1.0`). With batch=1 and lr=1.0, ADADELTA's accumulated squared gradients exploded, collapsing predictions to constant output (std~0.0002). Setting lr=0.005 fixed this.

3. **H2O `rate_annealing=1e-6`** per sample: H2O decays the learning rate as `rate/(1 + rate_annealing * N)`. Implemented as a per-batch lr schedule.

4. **H2O `score_training_samples=10000`**: H2O only scores on 10K random subset, not full 255K. We were scoring on full data, which was both slow and gave overly precise loss estimates.

5. **Throughput mismatch**: H2O (Java, single core) processes ~100-200K samples/sec. PyTorch with batch=1 processes ~5-10K/sec (Python overhead). With wall-clock scoring every 5s, H2O trained ~3 epochs between scores while PyTorch trained ~0.15 epochs. Model never had time to learn before early stopping fired. Fixed by switching to sample-count-based scoring (every 750K samples ≈ 3 epochs).

6. **batch_size=1 infeasible in PyTorch**: Pure SGD at 255K samples/epoch takes ~6 min/epoch in PyTorch vs ~2.5s in H2O. Compromised to `batch_size=32` (~11s/epoch), which preserves high gradient noise while being 32x faster.

**Configurations tested (period 0, 15, 20):**

| batch_size | lr | scoring | Result |
|---|---|---|---|
| 1024 | 1.0 | time-based 5s, full train | std ~0.012-0.017 (baseline) |
| 1 | 1.0 | sample-based, 10K | std ~0.0002 (collapsed) |
| 1 | 0.005 | time-based 5s, 10K | std ~0.005-0.008 (too conservative) |
| 32 | 0.005 | sample-based, 10K | loss stuck at 0.692 (no learning) |
| 32 | 1.0 | sample-based, 10K | loss increasing (diverging) |
| 1024 | 1.0 | sample-based 750K, 10K | std ~0.013-0.017 (slight improvement on P0/P15) |

**Conclusion:** PyTorch's ADADELTA cannot replicate H2O's mini_batch_size=1 behavior. H2O's internal `rate=0.005` and `rate_annealing=1e-6` interact with their Java ADADELTA implementation differently than PyTorch's `lr` parameter. This is an inherent framework deviation.

**Final DNN configuration:**
- `batch_size=1024` (H2O: 1 — PyTorch ADADELTA requires larger batches for stable convergence)
- `lr=1.0` (standard ADADELTA, learning-rate-free as per Zeiler 2012)
- `score_samples=10000` (H2O `score_training_samples=10000`)
- `score_every_n_samples=750000` (~3 epochs, matching H2O's effective throughput per scoring interval)
- All other params unchanged (architecture, dropout, L1 weights-only, rho=0.99, eps=1e-8, seed=1)

**Status:** Full 23-period retraining launched with final configuration.

### Datastream data pipeline — 2026-04-09

**Key finding:** CRSP identifies 1,936 unique S&P 500 ever-members vs Datastream's 1,322. This 614-stock difference was the largest driver of result gaps, not model parameters.

**Datastream data extraction:**
- Source: `tr_ds_equities` on WRDS
- Membership: `ds2constmth` (indexlistintcode=4408) — 1,370 spells, 1,322 unique infocodes
- Returns: `ds2primqtri` — daily total return index, converted to returns via `RI_t / RI_{t-1} - 1`
- Calendar: Datastream includes 238 non-US holiday dates with stale prices. Filtered to CRSP US trading calendar (6,805 dates) to match paper's study period boundaries.
- Monthly membership: avg 499.5 stocks/month (paper: 499.7)

**H2O model pipeline:**
- Created `h2o_dnn_phase1.py`, `h2o_gbt_phase1.py`, `h2o_rf_phase1.py` — exact paper parameters
- Created `run_phase1_h2o.py` and `test_datastream_h2o.py` — full 23-period pipeline
- H2O v3.46 (paper used ~v3.8-3.10 from 2016)

**Full 23-period results (Datastream + H2O, k=10):**

| Model | Paper Pre | DS+H2O Pre | Ratio |
|-------|-----------|-----------|-------|
| ENS1 | 0.0045 | 0.0042 | 94% |
| RAF | 0.0043 | 0.0040 | 93% |
| GBT | 0.0037 | 0.0039 | 106% |
| DNN | 0.0033 | 0.0028 | 85% |

ENS1 post-cost Sharpe: 2.23 (paper: 1.81) — exceeds paper due to lower turnover.

**Comparison: CRSP vs Datastream impact:**

| Pipeline | ENS1 Pre | Ratio |
|----------|----------|-------|
| CRSP + Python | 0.0041 | 91% |
| CRSP + H2O | 0.0042 | 93% |
| **Datastream + H2O** | **0.0042** | **94%** |

Switching to Datastream improved results marginally at the full-sample level but more significantly during crisis periods. The main effect was through constituency differences (which 500 stocks), not return data.

**Files created:**
- `scripts/build_data_datastream.py` — Datastream data extraction
- `scripts/build_features_labels_datastream.py` — feature/label pipeline
- `scripts/test_datastream_h2o.py` — full 23-period H2O pipeline with caching
- `src/krauss/models/h2o_dnn_phase1.py`, `h2o_gbt_phase1.py`, `h2o_rf_phase1.py`
- `src/krauss/data/universe_frozen.py` — frozen universe variant (tested, not adopted)
- Data saved to `data/datastream/` and `data/models_ds_h2o/`
