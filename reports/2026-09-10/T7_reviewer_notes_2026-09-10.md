# T7 Daily Signal Report 2026-09-10: how it was produced

Generated 2026-10-25 12:34 SGT; mode live.

## Steps

- Task 1 news: 540 company articles and 72 political articles, FinBERT-scored; event tags.
- Task 2 technical: RSI, MACD, Bollinger bands (20, 2σ), VWAP and weekly trend per stock.
- Task 3 macro: regime code 0; VIX 16.5; US yield spread.
- Task 4 options: Singapore ETF (EWS) put/call volume ratio 0.71.
- Task 5 evidence: rule layer rl_v8: 7 positive, 0 negative, 8 neutral.
- Task 6 forecast: the stored live champion forecast and probabilities for each stock, with its record against a time-ordered simple rule (✓ / ✗); 'calmer than most' from the same stored probabilities.
- Task 7: patterns, attention, cards, market read, compliance checks R1-R7; narrative AI (openai/gpt-oss-20b) with checks N1-N13.

## Provenance

- Project database: Tasks 1-6 stored outputs for the period; adjusted daily closes.
- Record of each statement: research R14, R10 and the stored champion record.
- Period: close of 03 Sep to close of 10 Sep 2026
- AI blocks that passed only on a second attempt: c1, c2, c3
- AI blocks replaced by the template: summary: N11 {n_fc_none} is not in its draft line; summary: N11 {n_fc_rise} is not in its draft line; summary: N11 {n_trusted} is not in its draft line; summary: N11 drops 'mapletree industrial'; summary: N11 drops 'st engineering'; summary: N11 drops 'capitaland integ. comm. trust'; summary: N11 adds 'capitaland integrated commercial trust'; summary: N11 adds 'wilmar international'; summary: N11 drops 'wilmar intl'

## Measures

- Patterns today: Lag 4, Model-only view 4, Quiet 4, Muted, pending 3
- Attention today: Routine 11, Elevated 3, Notable 1 (⚑ = Elevated or Notable)
- Forecasts marked ✗: 8 of 15; median record gap 19 points below the simple rule
- Outlook record (R14, frozen): 31 dates; calmer five moved >1% 42% vs 56%; normal swing held 74%

## Terms

- News & signals: Task 5 net rule score beyond ±0.35. News today: (positive − negative) / articles beyond ±0.15; 'too few' below 3 articles.
- Price vs news: today's move against Task 5's label threshold ±0.60%.
- Normal 3-day swing: ±(20-day s.d. × √3). Calmer than most: five lowest 1 − P(no significant move).
- ✗: the record is more than five points below a rule predicting the stock's most common matured outcome.
