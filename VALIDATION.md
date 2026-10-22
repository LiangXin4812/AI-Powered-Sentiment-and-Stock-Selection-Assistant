# Verification status (version 3.1, 22 October 2026)

## Executed

- **Clean Python 3.12 environment from `requirements.lock`**: all pinned
  packages installed. CPU PyTorch is used only by FinBERT during live
  ingestion and is not imported by any offline command.
- **Offline test suite (`smoke`)**: 40 tests passed. Besides the model,
  config, baseline, calibration-fold, outlook and R14 checks of earlier
  versions they cover: the default sample (21 Sep 2026, saved copy of the live
  database) with its cards chosen by today's news readings, 'calmer than most'
  equal to the five highest stored chances of no big move, Task 5's summary in
  full without advice sentences, no reviewer page and the reviewer notes file;
  the stored-records sample (17 Sep) with the Task 5 rules reproducing the
  stored rl_v8 score for all 15 stocks; the AI narrative layer (mocked model):
  the six-line market read built by rules, checks N1–N11 each replacing only
  the failing block, one retry with feedback, saved text for an older layout
  not reused; and the automation pieces (both samples pass every report check,
  publishing writes the day folder and index, the workflow runs at 18:30
  Singapore time on weekdays and holds no credential).
- **Live path from Google Colab against the project database** (20 Oct 2026):
  `doctor --online` and `signal-report --date 2026-09-21 --save-snapshot` ran
  end to end six times while the AI narrative layer was hardened; on the last
  run every AI block passed its checks at the first attempt. The saved inputs
  are `tests/fixtures/live_snapshot_2026-09-21.json`.
- **Multi-date check** (`check-dates`, Colab against the project database,
  22 Oct 2026): the report was built for all 20 stored dates from 17 Aug to
  22 Sep 2026; 18 passed every check. The two failures were quoted third-party
  headlines matching the advice and price-target patterns (24 Aug, 2 Sep);
  the six earliest dates (17-24 Aug) also printed 'nan' for a forecast record that did not yet
  exist. Both are fixed in 3.1 and covered by a regression test.
- **Sample reports** for 21 Sep and 17 Sep 2026 converted to PDF with
  LibreOffice and inspected page by page: six A4 pages each, sections 1–5 in
  order, section 3 starting on its own page.
- **run-all** exercised with the four pipeline steps mocked: a non-trading day
  exits without work; a trading day calls ingest, daily and signal-report in
  order and publishes the Word report, PDF and reviewer notes to
  `reports/<date>/` with the index `reports/README.md`.

## Not executed or not established

| Item | Limit |
| --- | --- |
| Docker image build | Docker Hub was not reachable here; the container file set was verified in earlier versions |
| New-layout AI text on a live run | Colab notebook step 4 (`signal-report --date 2026-09-21 --save-snapshot`) refreshes the snapshot with it |
| Multi-date check after the 3.1 fixes | Colab notebook step 5 reruns it |
| Scheduled run on GitHub Actions | Needs the repository and its secrets; see `AUTOMATION.md` |
| Forward confirmation of R14 | Exploratory rule, frozen 16 Oct 2026; needs at least 15 new dates |
| New Task 6 directional skill | Not established; no model promoted |
| Reader comprehension | Protocol in `READER_TEST.md`; no participants yet |
| Article timestamps | Not audited against each prediction moment |

## Local submission run

1. `docker compose build`, then `doctor`, `smoke`, `sample`, `figures`.
2. With credentials: `doctor --online`, `signal-report --date 2026-09-21`,
   `check-dates --start 2026-08-17` and `report --date 2026-09-21`.
3. Open the Daily Signal Report; check the market read against the tables,
   that the cards show the stocks with a news reading, and that no direction
   is stated as certain and no advice is given.
