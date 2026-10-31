# SGX AI-Powered Sentiment & Stock Selection Assistant

NUS FT5007 capstone, Liang Xin (A0221401E). Version 3.5, 24 October 2026.

The system reads company news, price indicators, macro conditions and options
activity for 15 SGX-listed stocks every trading day (Tasks 1–4), turns them
into an explained evidence reading of what the news adds on top of the price
(Task 5: weighted rules, two ML models, a contradiction check and an AI
summary), forecasts the next three trading sessions with calibrated models
(Task 6) and writes two documents (Task 7) on one report engine (11 patterns,
attention, withholding rule, compliance checks R1–R7):

- the **Daily Signal Report** for investors, in five numbered sections:
  1 today in 30 seconds (the day's market read: where the moves were, whether
  company news, the sector or the market moved with them, the backdrop, what
  today's news adds, the next three days, what it means) and all 15 stocks
  ranked; 2 the last five trading days by sector and cards for the stocks
  where today's news adds something; 3 the next three trading days: normal
  swing, calmer stocks, Task 6's lean and its record per stock; 4 the system
  behind the report; 5 what three months of testing show, limitations,
  glossary. How the report was produced is saved next to it
  (`T7_reviewer_notes_<date>.md`), not printed in it.
- the **Research Report** for reviewers: reliability, calibration, the
  evidence layer, research findings, methodology and the incident log.

What the Daily Signal Report says about the next three sessions, and how often
each statement has held up (research R14, out-of-fold walk-forward
predictions, 31 dates from 27 Jul to 15 Sep 2026, 465 stock-days; an event is
a three-session move of more than 1%):

| Statement | Record |
| --- | --- |
| Normal three-session swing, ±(20-day volatility × √3) | realised change inside it 73.8% of the time (about 68% expected) |
| "Calmer than most": the five stocks a day with the highest chance of no big move | moved more than 1% in 41.9% of cases vs 56.5% for the other ten; lower rate on 22 of 31 dates (2 tied) |
| Up-or-down lean | not reliable: live calls right 40% vs 54% for always "no big move"; calls marked ✓ right 46%, ✗ 33%, but the gain is in "no big move" calls; ✓ rise-or-fall calls right 22% (R15) |
| Flag ⚑ (worth a closer look) | not a forecast of size: over the next 3 sessions flagged stocks moved 1.18% on average, the rest 1.26% (R15, 18 report dates) |

None of this is a trading strategy, and the report never gives advice.

## Credential-free review (no database, no API keys)

```bash
docker compose build
docker compose run --rm capstone python -m sgx_capstone doctor           # config, prompts, model preflight
docker compose run --rm capstone python -m sgx_capstone smoke            # offline tests
docker compose run --rm capstone python -m sgx_capstone sample           # Daily Signal Report for 21 Sep 2026
docker compose run --rm capstone python -m sgx_capstone sample --date 2026-09-17   # stored-records version
docker compose run --rm capstone python -m sgx_capstone figures          # report figures + sources.json
docker compose run --rm capstone python -m sgx_capstone outlook-evidence # recompute R14
docker compose run --rm capstone python -m sgx_capstone followup         # recompute R15
docker compose run --rm capstone python -m sgx_capstone research-report  # HTML research review
```

`sample` rebuilds 21 Sep 2026 offline from the saved copy of the live
database (`tests/fixtures/live_snapshot_2026-09-21.json`, saved by
`signal-report --save-snapshot`), including the AI text written on the live
run; every figure, forecast and chance is as stored that day.
`sample --date 2026-09-17` builds the report only from stored project records
(feature rows, rule-layer readings, stored returns and the stored forecast
record); there, 'calmer than most' comes from the bundled model, which was
retrained after that date, and the banner says so. Outputs land in
`outputs/sample_report/`.

Without Docker: create a Python 3.12 environment, `pip install -r
requirements.lock`, then run the same `python -m sgx_capstone ...` commands
from this folder. CPU PyTorch (`torch==2.6.0+cpu` from the PyTorch index) is
needed only for FinBERT during live ingestion.

## Live pipeline (existing PostgreSQL database)

Copy `.env.example` to `.env` and fill it locally. `SUPABASE_URL` must point
to the project's populated database. `GROQ_API_KEY` is needed for the Task 5
writer and the Task 7 AI text (without it Task 7 prints its template text).
No credentials are bundled.

```bash
docker compose run --rm capstone python -m sgx_capstone doctor --online
docker compose run --rm capstone python -m sgx_capstone run-all             # the whole day after the close (below)
docker compose run --rm capstone python -m sgx_capstone ingest              # Tasks 1-4, today only
docker compose run --rm capstone python -m sgx_capstone daily               # Tasks 5 + 6 for the latest complete date
docker compose run --rm capstone python -m sgx_capstone signal-report --date 2026-09-21
docker compose run --rm capstone python -m sgx_capstone check-dates --start 2026-08-17   # every stored date
docker compose run --rm capstone python -m sgx_capstone report --date 2026-09-21   # Research Report (Word)
```

`ingest` refuses any date other than today in Singapore or the latest closed
SGX session before the next one opens, because current feeds cannot
reconstruct a past day. `signal-report --date` and
`report --date` rebuild a stored date; an outcome is used only if it had
matured by that date.

## Daily automation

`run-all` takes the latest closed SGX session (and stops if its report is
already published), runs Tasks 1–4, Tasks 5 and 6 and the Daily Signal
Report, and publishes it to `reports/<date>/` (Word, PDF, reviewer notes)
with an index at `reports/README.md`; e-mail is optional.
`.github/workflows/daily_report.yml` runs it at 18:30 Singapore time on
weekdays, with a backup at 21:30 because GitHub can start a scheduled run
late, using the credentials stored as repository secrets. Setup and
reviewer access: `AUTOMATION.md`.

## Google Colab

`Run_T7_in_Colab.ipynb` (delivered with the package) uploads this folder,
installs it, reads the keys from Colab secrets or a hidden prompt, and runs
the live 21 Sep report and the multi-date check.

## What changed in 3.5

- **`republish`**: rebuilds every stored date's Daily Signal Report in the
  current format (AI writer on, PDF when LibreOffice is installed), checks
  each one and publishes it to `reports/<date>/` with a fresh index, so the
  archive a reviewer browses never mixes layouts. Run it after any layout
  change (Colab notebook Step 7a; Step 7b pushes code and archive together).
  Covered by a test with the database mocked.
- The report's pattern reference points to the final report's Appendix C
  (appendices now follow the order in which the report first uses them).
- Figures use the report's names for the outlook: *normal 3-day swing* (the
  code's typical range) and *calmer than most* (quieter-than-most ranking);
  the architecture figure counts 43 tests and shows `republish`.
- The R15 test now runs on a copy of the frozen inputs. Before, running the
  tests while `check-dates` or `republish` was writing `outputs/date_check/`
  recomputed `outputs/research/R15_report_followup.csv` from a partial set,
  and a report built at that moment printed "nan" in its testing section.

## What changed in 3.4

- **The reader's vocabulary.** An investor does not know the project's task
  numbers, so the Daily Signal Report no longer prints "Task 5" or "Task 6":
  cards say *News & signals reading*, *Rules behind the reading*, *AI summary
  of the evidence* and *Forecast lean*; section 4 describes four plain stages
  (collect the day's facts; read the news and signals; forecast the next 3
  days; check and write the report) and notes once which project Tasks they
  are. The AI prompt uses the same names, and code maps any "Task 5/6" in AI
  text to them (`narrative.plain_terms`, applied like the hyphen fix). A test
  asserts that no "Task 5-7" reaches the reader.
- The 21 Sep report rebuilt from the saved database copy is no longer labelled
  "SAMPLE" (it is the real report of that day); the stored-records
  reconstruction of 17 Sep keeps its label.
- Figures: the R14 chart's legend no longer overlaps a bar label, the
  calibration chart's point labels sit above their bubbles, and the
  architecture figure shows the AI writer, checks N1-N13 and the daily run.
- The report's pattern reference points to the final report's Appendix G.

## What changed in 3.3

A live rerun of the 21 Sep report on 24 Oct 2026 (the first with the 3.2 card
facts) showed two new failure modes of the AI writer: it wrote chances out in
words ("a fifty-percent chance"), which evades the no-digit rule, and said the
price had "already" followed the news on a card whose stored fact says not
yet. Two checks were added: **N13** (no figure written out in words) and
**N12** (a card may not contradict its stored price fact). The prompt states
both rules, saved AI text is **re-checked whenever the sample is rebuilt**
(a block that fails a newer check falls back to the code's text, named in the
output), the AI label no longer says "kept the wording" when lines were
reworded, and the reviewer notes say N1-N13. Snapshot refreshed from that
run; 42 offline tests.

## What changed in 3.2

- **R15, what followed the report** (`python -m sgx_capstone followup`, offline):
  from the saved inputs of all 20 checked reports (`evidence/date_check_inputs/`)
  and the stored outcomes, flagged stocks did not move more over the next three
  sessions (1.18% vs 1.26%), and the ✓ gain comes from "no big move" calls
  (58.4% vs 52.9%); ✓ rise-or-fall calls were right 21.7% (46 calls). The
  report now says so: the lean is "not reliable, even with ✓", a ✓ on a
  direction is called unknown on the card, and section 5 adds "Does ⚑ mean a
  bigger move?". Covered by a test.
- The AI writer no longer receives the pattern description or the phrase
  "treat as noise" for cards, which it had mixed into the five-day line.
- A flag on a model-only pattern says the model leans one way instead of
  "holds a firm view".
- Research Report: the R14 sentence uses the renamed fields (lower on 22 of 31
  dates, 2 tied); this crashed `report` in 3.1.
- Sample snapshot for 21 Sep refreshed from the 23 Oct live run; section 5
  compacted so the report stays at six pages.

## What changed in 3.1

`check-dates` was run on all 20 stored dates (17 Aug to 22 Sep 2026): 18 passed
every check. It found, and 3.1 fixes: two quoted headlines that read as advice
or a price target ("Should You Sell?", "Will DBS Reach S$100 by ...") are no
longer quoted; 'nan' printed on the early dates before any live forecast had
matured (the report now says the record is too short); ⚑ means ELEVATED only,
so it matches its definition (0-4 stocks a day instead of up to 7), with the
reason named per stock. The market read shortens long lists and spells counts
in words; the AI label says whether the AI reworded the code-written draft.

## What changed in 3.0

- **Cards show where today's news adds something**: stocks flagged ⚑ first,
  then the strongest positive or negative readings; the biggest movers only
  fill remaining places. Each card adds *Today: news vs price* (the pattern
  engine in plain words, with the pattern's name) and *Last 5 days in
  context* (company news against the sector peers and the 15-stock average,
  and the conclusion drawn from them). Task 5's AI summary is printed in
  full, advice sentences removed; the political rule is flagged once.
- **Market read** on page 1: six lines built by rules from Tasks 1–6 (what
  happened and where, whether company news explains the biggest moves, the
  backdrop, what today's news adds, the next three days, what it means). The
  AI rewrites it; rule N11 keeps each line to its facts.
- **Calmer than most** comes from the forecast probabilities stored that day
  when available (no hindsight; the same numbers as the chance column).
- Section 3 leads with what has held up (normal swing, calmer stocks); the
  up-or-down lean is shown with its record and the ✓ filter. The plain-word
  "Price vs news" replaces pattern names in the stock table; the internal
  contradiction note appears only when a reading is strongly mixed;
  borderline readings that depend on the unreliable political rules say so.
- Sections are numbered and referenced by number, not page.
- The reviewer page left the report: `T7_reviewer_notes_<date>.md`.
- New commands `run-all` and `check-dates`, the GitHub Actions schedule and
  `AUTOMATION.md`. Default sample: 21 Sep 2026.

## What changed in 2.5

- **Daily Signal Report rebuilt to four investor pages + one reviewer page**
  (supervisor and user review): each page answers one question; a tested
  verdict (✓ reliable, ! promising, ○ describes only, ✗ not reliable) sits
  next to each kind of statement on page 1 and in a scorecard on page 4.
- The chart that drew the typical range and the quieter-than-most ranking
  together was removed: the two measures disagree for some stocks, which
  readers could not follow. Page 3 is a table sorted by normal 3-day swing,
  with the swing also in dollars on S$1,000, a separate column for the
  model's calmer five, and one line per calmer stock saying whether its
  recent swings agree.
- Terms: "normal 3-day swing" (typical range), "calmer than most"
  (quieter-than-most ranking). Stock cards follow the structure of the first
  report version: what happened, what the news and signals said, does it fit,
  worth checking.
- Layout: A4, header band, status chips, page footer; one chart (page 2).
  `outputs/sample_report/` now holds `week_<date>.png` instead of the period
  and outlook charts.
- The sample test checks the new headings, that no advice, target or certainty
  pattern (R1-R3) appears anywhere in the report, and that each verdict is
  printed. The 21 offline tests pass.

## What changed in 2.4

- **Plainer words for beginners** (supervisor feedback): "likely calmer"
  instead of "quieter than most", "usual 3-day move" instead of "typical
  range", rates as natural frequencies ("about 4 in 10"), a ⚑ flag instead of
  ROUTINE/NOTABLE/ELEVATED, pattern names only in the reviewer appendix, and a
  glossary (RSI, VWAP, Bollinger band, MACD, ETF options, VIX).
- **The original brief's dashboard features, in the report**: page 2 adds a
  stocks-by-days chart (each day's evidence as colour, the day's price change,
  ★ on days with more company-event news than usual: the sentiment trend and
  catalyst heatmap); page 3 ranks all 15 stocks by today's evidence score
  (ties by number of news items, as in the original design) with the main
  signal for and against, whether news and price-chart signals agree, news
  coverage with a low-coverage flag, the usual 3-day move, and a sector table.
  Market-wide signals are stated once above the table. The page says the
  ranking is a description, not a pick list (R4). Nothing new is computed:
  every column comes from Task 5, Task 6 or prices already used.
- **Time frame**: `signal-report --sessions N` (and `sample --sessions 1|3|5`)
  sets the period.
- One evidence reading per trading day in the period (a reading dated on a
  weekend is not counted).
- The 21 offline tests pass.

## What changed in 2.3

- **Stock cards name the period** ("over the five sessions 14–18 Sep") and
  the appendix lists which stored evidence readings were counted.
- **Stock cards list the Task 5 rules that fired**, in plain words
  ("Signals pointing up: RSI below 30 (oversold) ..."). The live report reads
  them from `llm_analyses_audit` (the deterministic rule output already used by
  the Research Report); it was fetching Task 5's warning level and not printing
  it, and now says whether the internal checks found the signals consistent or
  mixed. The offline sample re-runs the production rule layer on the stored
  feature rows (`research/r8_rule_detail.py`) and uses a stock only if the
  re-run reproduces the stored rl_v8 net score (30 of 30 stock-days do;
  `outputs/research/R8_rl_v8_rule_detail.csv`).
- **Defect documented, not hidden**: `political_direction` and
  `avg_sentiment` (task6_features) average the `sentiment_score` column, which
  stores FinBERT's confidence (0.5–1.0), not a signed tone. The rule
  "Positive geopolitical backdrop" therefore fires on every reading (R7:
  100%), and Task 5 summaries could call the political backdrop supportive on
  a day of negative political news. The pair it forms with the political-volume
  rule adds the same amount to every stock, so no ranking changes. Fixing the
  features needs a refit of Task 5 and Task 6 and is future work; until then
  the Daily Signal Report leaves both political rules off its stock cards and
  removes Task 5 summary sentences about political news.
- The 21 offline tests pass; the sample test now also checks the period
  wording, the reproduced rule scores and that the political rule is never shown.

## What changed in 2.2

- **Daily Signal Report wording only; no number, rule or threshold changed.**
  A three-minute "How to read this report" guide opens page 1 (supervisor
  review, 14 October: a first-time reader should know which pages to read
  first). The stock cards say "evidence score" instead of "net rule score";
  the trust table describes the evidence reading's record in words, and the
  rank-correlation statistic moved to the reviewer appendix, which gains an
  "Evidence vs next-day returns (R4)" row. The 21 offline tests pass
  unchanged.

## What changed in 2.1

- **Daily Signal Report** (`sgx_capstone/daily_report.py`) replaces the
  Investor Brief. Same Task 7 engine; the period is the last five sessions
  (`t7.period_sessions`); facts and outlook on separate pages; possible
  reasons with the evidence and its sources; limitations in the report;
  statistics in a reviewer appendix. Commands `sample` and `signal-report`
  replace `specimen` and `brief`.
- **Outlook explanation** is template only. A language-model version with
  checks O1–O7 was built and tested; the checks cannot confirm that a number is
  attached to the right thing, so numbers and effect directions are templated.
  A test documents that limit.
- **R14** uses one event definition everywhere: a three-session move of more
  than 1%.
- **Compliance**: if the deterministic text also fails R1–R7, the stock's
  report is blocked rather than published (tested).
- **No hindsight in the sample**: the quoted forecast record uses only
  forecasts matured by the report date (tested).
- **Research Report**: the weekly-accuracy caption describes the live outcome
  mix (declines, rise calls and how many were right) instead of an untested
  market-regime explanation.
- **Fix**: Task 6 serving filled a missing input with 0, which invents a
  signal (RSI 0, VIX 0). It now uses the training mean and refuses a forecast
  with more than three missing inputs.

## Project layout

| Path | Purpose |
| --- | --- |
| `sgx_capstone/__main__.py` | Commands: `doctor`, `smoke`, `ingest`, `daily`, `report`, `signal-report`, `sample`, `figures`, `outlook-evidence`, `followup`, `research-report`, `run-all`, `check-dates` |
| `sgx_capstone/legacy/` | Task 1–6 serving code selected from the notebooks |
| `src/task7_report.py`, `src/task7_generate.py` | Task 7 engine: report input, 11 patterns, attention, price context, reliability, compliance R1–R7, renderer |
| `sgx_capstone/outlook.py` | Typical range, quieter-than-most ranking, group Shapley explanation |
| `sgx_capstone/daily_report.py` | Daily Signal Report (live and offline sample) |
| `sgx_capstone/report_builder.py` | Research Report (Word) |
| `sgx_capstone/figures.py` | Report figures from research tables |
| `sgx_capstone/research/` | Research scripts R1–R14 and the R8 rule detail (read-only; never write a serving model) |
| `outputs/research/`, `evidence/` | Retained research tables and frozen evidence |
| `config.yaml`, `settings/`, `prompts/` | Behaviour, feeds and LLM prompts, hashed into a manifest on every live run |
| `models/` | Pinned serving artefacts (trusted project files; never load a pickle from elsewhere) |
| `tests/` | Offline tests on real feature rows |
| `READER_TEST.md`, `VALIDATION.md` | Reader study protocol; what was and was not verified |

## Evidence limits

- 57 complete trading dates in one quarter of 2026; R14 covers 31 of them.
  Results describe that period.
- After honest calibration the three-session classifier is no better than
  always answering "no significant move" at any horizon tested; twelve
  alternatives in R11 failed criteria fixed in advance.
- The quieter-than-most rule is exploratory and frozen; its acceptance bar
  needs at least 15 new dates.
- Production calibration folds can split a trading day; the date-purged tool
  in `research/purged_calibration.py` exists but is not promoted.
- Article timestamps have not been audited against each prediction moment.
- `political_direction` and `avg_sentiment` are mean FinBERT confidences, not
  signed tones; a refit with signed features is future work.
- The reader study in `READER_TEST.md` has not been run.
- The Docker image has not been built on a container host from this
  environment; see `VALIDATION.md`.
