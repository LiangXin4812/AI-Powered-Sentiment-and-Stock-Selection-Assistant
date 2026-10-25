# T7 Daily Signal Report 2026-09-11: how it was produced

Generated 2026-10-25 12:37 SGT; mode live.

## Steps

- Task 1 news: 543 company articles and 72 political articles, FinBERT-scored; event tags.
- Task 2 technical: RSI, MACD, Bollinger bands (20, 2σ), VWAP and weekly trend per stock.
- Task 3 macro: regime code 0; VIX 17.1; US yield spread.
- Task 4 options: Singapore ETF (EWS) put/call volume ratio 0.77.
- Task 5 evidence: rule layer rl_v8: 5 positive, 0 negative, 10 neutral.
- Task 6 forecast: the stored live champion forecast and probabilities for each stock, with its record against a time-ordered simple rule (✓ / ✗); 'calmer than most' from the same stored probabilities.
- Task 7: patterns, attention, cards, market read, compliance checks R1-R7; narrative AI (openai/gpt-oss-20b) with checks N1-N13.

## Provenance

- Project database: Tasks 1-6 stored outputs for the period; adjusted daily closes.
- Record of each statement: research R14, R10 and the stored champion record.
- Period: close of 04 Sep to close of 11 Sep 2026
- AI blocks that passed only on a second attempt: c1, c2
- AI blocks replaced by the template: summary: N10 gives Task 6 more than little weight; summary: N7 78 words; summary: N11 drops 'capitaland integ. comm. trust'; summary: N11 adds 'capitaland integrated commercial trust'; summary: N11 adds 'wilmar international'; summary: N11 drops 'wilmar intl'; summary: N11 drops 'capitaland integ. comm. trust'; summary: N11 adds 'capitaland integrated commercial trust'; summary: N11 drops 'capitaland integ. comm. trust'; summary: N11 adds 'capitaland integrated commercial trust'; summary: N11 adds 'wilmar international'; summary: N11 drops 'wilmar intl'; summary: N11 adds 'wilmar international'; summary: N11 drops 'wilmar intl'

## Measures

- Patterns today: Quiet 6, Model-only view 4, Dormant 2, Momentum 1, Muted, pending 1, Noise 1
- Attention today: Routine 12, Elevated 2, Notable 1 (⚑ = Elevated or Notable)
- Forecasts marked ✗: 8 of 15; median record gap 21 points below the simple rule
- Outlook record (R14, frozen): 31 dates; calmer five moved >1% 42% vs 56%; normal swing held 74%

## Terms

- News & signals: Task 5 net rule score beyond ±0.35. News today: (positive − negative) / articles beyond ±0.15; 'too few' below 3 articles.
- Price vs news: today's move against Task 5's label threshold ±0.60%.
- Normal 3-day swing: ±(20-day s.d. × √3). Calmer than most: five lowest 1 − P(no significant move).
- ✗: the record is more than five points below a rule predicting the stock's most common matured outcome.
