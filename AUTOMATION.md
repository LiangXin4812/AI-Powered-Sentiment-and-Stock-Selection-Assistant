# Daily automation

Every SGX trading day after the close, one command runs the whole system and
publishes the investor report:

```bash
python -m sgx_capstone run-all
```

1. **Report date.** The latest SGX session that has closed, read from the
   daily price bar (weekends and public holidays have none, so no holiday
   table to maintain). If that session's report is already in `reports/`, it
   stops and does nothing. A run that starts late, even after midnight, still
   reports on that session: until the next session opens its prices are final
   and any news since the close is known before the next open.
2. **Tasks 1–4** (`ingest`): news with FinBERT, technical indicators, macro
   backdrop, options positioning, written to the project database.
3. **Tasks 5 and 6** (`daily`): evidence readings with the Task 5 AI summary;
   the champion forecast with calibrated probabilities. The run stops if any
   of the 15 stocks is missing, so a half-built report is never published.
4. **Task 7** (`signal-report`): the Daily Signal Report, with the AI-written
   market read and card lines checked by rules N1–N13 (template text if the AI
   is unavailable).
5. **Publish**: `reports/<date>/` gets the Word report, a PDF (when
   LibreOffice is installed) and `T7_reviewer_notes_<date>.md` (how the report
   was produced). `reports/README.md` is rebuilt as the index, newest first.
   If SMTP settings are present the report is also e-mailed.

Any scheduler that can run a command works (cron on a server, Windows Task
Scheduler, Docker). The repository includes a ready schedule for GitHub
Actions, which needs no machine of your own.

## GitHub Actions setup (once, about 10 minutes)

1. Create a **private** repository and upload the contents of this folder
   (the folder that holds `README.md`, `sgx_capstone/` and `.github/`).
2. Repository → *Settings* → *Secrets and variables* → *Actions* → *New
   repository secret*:

   | Name | Value | Needed |
   |---|---|---|
   | `SUPABASE_URL` | the project database URL | yes |
   | `GROQ_API_KEY` | the Groq key used by Task 5 | yes |
   | `HF_TOKEN` | Hugging Face token (faster FinBERT download) | optional |
   | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `REPORT_TO` | e-mail delivery (for Gmail: smtp.gmail.com, 587, your address, an app password, the recipients) | optional |

3. Repository → *Actions* → enable workflows. *Daily SGX signal report* now
   runs at 18:30 Singapore time, Monday to Friday, with a backup run at 21:30
   (`.github/workflows/daily_report.yml`). GitHub can start scheduled runs
   hours late; that is harmless, because the run reports on the latest closed
   session and the backup exits once the report is published.
4. To test at once: *Actions* → *Daily SGX signal report* → *Run workflow*
   (after 17:30 on a trading day, or any time before the next session opens).
   Tick *force* only to test on a non-trading day.

Each run takes roughly 20–40 minutes (news collection and FinBERT dominate).
When it finishes, the report is committed to `reports/<date>/` and also kept
as a downloadable artifact of the run. A failed run is e-mailed to the
repository owner by GitHub and its log shows the step that stopped.

## Giving reviewers access

- Add the supervisor or examiner as a collaborator (*Settings* →
  *Collaborators*, read access). They open `reports/README.md` in the browser
  and click any day's report.
- Or download `reports/` and attach it to the submission.

No credential is stored in the code or the workflow file; they live only in
the repository's encrypted secrets. Scheduled runs also keep the free
database project active.

## Rebuilding the archive after a change

```bash
python -m sgx_capstone republish --start 2026-08-17
```

rebuilds every stored date's report in the current format, with the AI
writer (`--no-llm` for template text), checks each one like `check-dates`,
and publishes it to `reports/<date>/` with a fresh `reports/README.md`. The
daily schedule only ever writes today's report, so after a layout change
this is what keeps the archive consistent. From Colab: notebook Step 7a,
then Step 7b pushes the code and the rebuilt archive.

## Checking stability across dates

```bash
python -m sgx_capstone check-dates --start 2026-08-17
```

builds the report for every date with stored Task 5 and Task 6 outputs
(template text, no AI call unless `--llm`) and checks each one: compliance
patterns R1–R3, all sections present, 15 stocks, five calmer stocks, three
cards, a six-line market read, no unfilled placeholder. The reports and
`summary.csv` land in `outputs/date_check/`.
