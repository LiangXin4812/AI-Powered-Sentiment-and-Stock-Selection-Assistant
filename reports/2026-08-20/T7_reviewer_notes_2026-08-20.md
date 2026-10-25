# T7 Daily Signal Report 2026-08-20: how it was produced

Generated 2026-10-25 12:06 SGT; mode live.

## Steps

- Task 1 news: 538 company articles and 72 political articles, FinBERT-scored; event tags.
- Task 2 technical: RSI, MACD, Bollinger bands (20, 2σ), VWAP and weekly trend per stock.
- Task 3 macro: regime code 1; VIX 14.9; US yield spread.
- Task 4 options: Singapore ETF (EWS) put/call volume ratio 0.76.
- Task 5 evidence: rule layer rl_v8: 13 positive, 0 negative, 2 neutral.
- Task 6 forecast: the stored live champion forecast and probabilities for each stock, with its record against a time-ordered simple rule (✓ / ✗); 'calmer than most' from the same stored probabilities.
- Task 7: patterns, attention, cards, market read, compliance checks R1-R7; narrative AI (openai/gpt-oss-20b) with checks N1-N13.

## Provenance

- Project database: Tasks 1-6 stored outputs for the period; adjusted daily closes.
- Record of each statement: research R14, R10 and the stored champion record.
- Period: close of 13 Aug to close of 20 Aug 2026; no evidence reading stored for Fri 14 Aug
- AI blocks replaced by the template: summary: N1 no response (400 with JSON mode: {"error":{"message":"Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.","typ)

## Measures

- Patterns today: Dormant 6, Muted, fading 2, Contradicted 2, Lag 1, Quiet 1, Momentum 1, Model-only view 1, Muted, pending 1
- Attention today: Routine 9, Elevated 4, Notable 2 (⚑ = Elevated or Notable)
- Forecasts marked ✗: 2 of 15; median record gap 100 points below the simple rule
- Outlook record (R14, frozen): 31 dates; calmer five moved >1% 42% vs 56%; normal swing held 74%

## Terms

- News & signals: Task 5 net rule score beyond ±0.35. News today: (positive − negative) / articles beyond ±0.15; 'too few' below 3 articles.
- Price vs news: today's move against Task 5's label threshold ±0.60%.
- Normal 3-day swing: ±(20-day s.d. × √3). Calmer than most: five lowest 1 − P(no significant move).
- ✗: the record is more than five points below a rule predicting the stock's most common matured outcome.
