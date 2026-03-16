# Earnings Call Sentiment -> Alpha Signal Pipeline

**NLP pipeline that pulls real SEC 10-Q filings, extracts sentiment signals with FinBERT, and discovers a contrarian alpha signal — all tested with ML cross-validation and walk-forward backtesting.**

Built as an end-to-end demonstration of NLP/ML pipelines for quantitative equity research.

---

## Pipeline Overview

```
SEC EDGAR API ──> 10-Q Download ──> MD&A Extraction ──> FinBERT Sentiment ──> Feature Engineering ──> Signal Testing ──> ML Model (CV) ──> Walk-Forward Backtest ──> Portfolio Construction
  (Real filings)    (HTML parsing)   (Regex + BeautifulSoup)  (Sentence-level)    (6 alpha signals)    (IC, OLS, terciles)  (RF, GB, TimeSeriesSplit)    (Contrarian L/S)
```

## What This Project Does

### 1. Data Ingestion from SEC EDGAR
Pulls real 10-Q quarterly filings for 15 S&P 500 companies across 5 sectors directly from the SEC EDGAR API:
- **Tech**: AAPL, MSFT, GOOGL, META, AMZN, NVDA
- **Financials**: JPM, GS, BAC
- **Healthcare**: UNH, JNJ
- **Energy**: XOM, CVX
- **Consumer**: WMT, PG

Maps tickers to CIK numbers via the official EDGAR lookup, fetches the 3 most recent 10-Q filings per company (~45 filings total), and extracts the **MD&A (Management's Discussion & Analysis)** section using HTML parsing with BeautifulSoup and regex pattern matching.

### 2. FinBERT Sentiment Model & Validation
Loads [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), a BERT model (109M parameters) fine-tuned on ~50K financial sentences. Before running on real filings, validates the model on 8 hand-picked financial sentences that test edge cases:
- Comparative phrases ("declined less than expected")
- Ambiguous actions ("$5B share buyback" — positive or neutral?)
- Hedged language ("cautiously optimistic")
- Context-dependent spending ("capex increased to support AI buildout")

### 3. Sentiment Extraction from Real Filings
Splits each filing's MD&A text into sentences using NLTK's `sent_tokenize`, then scores each sentence with FinBERT to get `P(positive)`, `P(negative)`, `P(neutral)`. Computes **net sentiment** = P(positive) - P(negative) per sentence.

### 4. Feature Engineering — 6 Alpha Signals
Aggregates sentence-level sentiment into filing-level features:

| Feature | Definition | Intuition |
|---|---|---|
| `sentiment_mean` | Mean net sentiment across sentences | Overall tone of the filing |
| `sentiment_std` | Std dev of sentence sentiments | Uncertainty/mixed messaging |
| `sentiment_skew` | Skewness of sentence sentiments | Hidden negative sentences in positive text? |
| `positive_ratio` | Fraction of positive sentences | Simple positive signal |
| `negative_ratio` | Fraction of negative sentences | Risk/distress signal |
| `sentiment_shift` | Change in mean sentiment vs. prior filing | QoQ tone change — improvement or deterioration |

Also checks feature correlations to assess multicollinearity before modeling.

### 5. Stock Return Data Merge
Downloads historical prices via `yfinance` and computes forward returns starting from the **next trading day after** each filing date (no lookahead bias):
- **5-day forward return** (1-week horizon)
- **20-day forward return** (1-month horizon)
- **Market-adjusted excess returns** (subtract SPY return over same window)

### 6. Signal Testing
Tests whether sentiment features predict forward returns using three standard methods from quantitative equity research:
- **Information Coefficient (IC)**: Spearman rank correlation between signal and forward return
- **Cross-sectional OLS regression**: Standardized coefficients with t-statistics
- **Tercile analysis**: Sort stocks by sentiment, compare average returns across top/middle/bottom groups

### 7. ML Models with Cross-Validation
Trains **Random Forest** and **Gradient Boosting** regressors with:
- **`TimeSeriesSplit`** (3 folds): Train on past, predict on future — no temporal data leakage
- **`GridSearchCV`**: Systematic hyperparameter search (tree depth, min samples, n_estimators)
- **Walk-forward model comparison**: Out-of-sample IC for OLS vs. RF vs. GB

Shallow trees (max_depth 2-4) and high min_samples_leaf (3-8) to prevent overfitting on ~45 observations.

### 8. Walk-Forward Backtest
Computes cross-sectional IC within each earnings season to test signal stability over time.

### 9. Portfolio Construction — Contrarian Long-Short Strategy
Based on the negative IC finding, constructs a **contrarian** long-short portfolio:
- **Long**: Bottom tercile by sentiment (most negative/cautious filings = oversold)
- **Short**: Top tercile by sentiment (most positive/optimistic filings = overpriced)

Measures per-quarter returns, hit rate, and annualized Sharpe ratio.

## Key Finding: It's a Contrarian Signal

The most important (and surprising) result: **positive filing sentiment predicts *lower* short-term returns.** Overly positive management tone is a sell signal — the market has already priced in the good news, and rosy language may indicate overconfidence. This is consistent with academic research (Loughran & McDonald, 2011).

**Why this makes sense:**
1. By the time a 10-Q is filed (40 days after quarter-end), the information is largely priced in
2. Abnormally positive financial language predicts negative future returns — management may be "spinning" bad news
3. Negative sentiment = oversold opportunity if the market overreacted

## Tech Stack

| Component | Tool |
|---|---|
| Sentiment Model | `ProsusAI/finbert` (HuggingFace Transformers, 109M params) |
| Data — Filings | SEC EDGAR API (direct HTTP, no wrapper library needed) |
| Data — Returns | `yfinance` |
| Text Parsing | BeautifulSoup, NLTK (`sent_tokenize`), regex |
| ML Models | scikit-learn (RandomForest, GradientBoosting, GridSearchCV, TimeSeriesSplit) |
| Statistics | statsmodels (OLS), SciPy (Spearman IC) |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deep Learning | PyTorch (FinBERT backend) |

## Project Structure

```
earnings-sentiment-alpha/
├── earnings_sentiment_alpha.ipynb   # Full pipeline notebook (runs end-to-end)
├── README.md
├── requirements.txt
├── .gitignore
└── data/                            # Generated at runtime
    ├── parsed_filings.json          # Cached EDGAR filings
    ├── sentence_sentiments.csv      # FinBERT scores per sentence
    └── merged_signals.csv           # Features + forward returns
```

**Generated visualizations** (saved as PNGs at runtime):
- `sentiment_distribution.png` — sentence-level sentiment histogram, label counts, per-ticker averages
- `feature_correlations.png` — alpha signal correlation heatmap
- `tercile_returns.png` — average forward returns by sentiment tercile
- `signal_vs_returns.png` — scatter plots of sentiment vs. forward returns
- `feature_importance.png` — Random Forest and Gradient Boosting feature importances
- `portfolio_performance.png` — cumulative and per-quarter contrarian L/S returns

## Setup

```bash
# Clone
git clone https://github.com/hrootscraft/earnings-sentiment-alpha.git
cd earnings-sentiment-alpha

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the notebook
jupyter notebook earnings_sentiment_alpha.ipynb
```

**Note:** First run downloads FinBERT (~400MB) and fetches ~45 SEC filings from EDGAR. Total runtime: ~5-10 minutes on CPU.

## Methodology Notes

- **No lookahead bias**: Forward returns are measured starting the *next trading day* after the filing date. Signals use only publicly available information.
- **Market-adjusted returns**: Stock returns are adjusted by subtracting SPY returns over the same window, isolating stock-specific alpha from market beta.
- **TimeSeriesSplit cross-validation**: ML models are trained on past data and evaluated on future data — prevents temporal data leakage that random KFold would cause.
- **Cross-sectional analysis**: Tests whether *relative* sentiment (stock A vs. stock B) predicts *relative* returns, the standard approach in quantitative equity research.
- **Walk-forward validation**: IC is measured within each earnings season, not just in aggregate — the gold standard for financial signal research.
- **FinBERT over GPT-4/LLMs**: Deterministic (no temperature sampling), fast (~1,350 sentences/min on CPU), free (no API costs), and domain-tuned for financial language.
- **10-Q over earnings transcripts**: Free, standardized, and legally mandated. Transcripts require paid APIs (e.g., S&P Capital IQ).

## Production Roadmap

| Step | Description |
|---|---|
| **Scale data** | Automate EDGAR ingestion for 500+ tickers, 5+ years of filings |
| **Add 10-K filings** | Annual reports have richer MD&A sections |
| **Earnings transcripts** | Add management vs. Q&A sentiment (requires paid API) |
| **Risk-adjust returns** | Use Fama-French factors for proper alpha measurement |
| **Signal combination** | Combine with fundamental factors (value, momentum) via ridge regression |
| **Production pipeline** | Airflow DAG, signal database, portfolio optimizer integration |
| **Monitoring** | Track IC decay, turnover, factor exposure drift |

