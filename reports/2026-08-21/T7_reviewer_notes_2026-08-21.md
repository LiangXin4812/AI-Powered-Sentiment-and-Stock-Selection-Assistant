# T7 Daily Signal Report 2026-08-21: how it was produced

Generated 2026-10-25 12:10 SGT; mode live.

## Steps

- Task 1 news: 544 company articles and 72 political articles, FinBERT-scored; event tags.
- Task 2 technical: RSI, MACD, Bollinger bands (20, 2σ), VWAP and weekly trend per stock.
- Task 3 macro: regime code 0; VIX 16.0; US yield spread.
- Task 4 options: Singapore ETF (EWS) put/call volume ratio 0.74.
- Task 5 evidence: rule layer rl_v8: 7 positive, 0 negative, 8 neutral.
- Task 6 forecast: the stored live champion forecast and probabilities for each stock, with its record against a time-ordered simple rule (✓ / ✗); 'calmer than most' from the same stored probabilities.
- Task 7: patterns, attention, cards, market read, compliance checks R1-R7; narrative AI (openai/gpt-oss-20b) with checks N1-N13.

## Provenance

- Project database: Tasks 1-6 stored outputs for the period; adjusted daily closes.
- Record of each statement: research R14, R10 and the stored champion record.
- Period: close of 14 Aug to close of 21 Aug 2026
- AI blocks replaced by the template: summary: N11 drops 'mapletree industrial'; summary: N11 drops 'ocbc bank'; summary: N11 drops 'raffles medical group'; summary: N11 adds 'wilmar international'; summary: N11 drops 'wilmar intl'

## Measures

- Patterns today: Model-only view 6, Move complete 3, Dormant 2, Quiet 2, Muted, fading 2
- Attention today: Routine 9, Elevated 4, Notable 2 (⚑ = Elevated or Notable)
- Forecasts marked ✗: 2 of 15; median record gap 75 points below the simple rule
- Outlook record (R14, frozen): 31 dates; calmer five moved >1% 42% vs 56%; normal swing held 74%

## Terms

- News & signals: Task 5 net rule score beyond ±0.35. News today: (positive − negative) / articles beyond ±0.15; 'too few' below 3 articles.
- Price vs news: today's move against Task 5's label threshold ±0.60%.
- Normal 3-day swing: ±(20-day s.d. × √3). Calmer than most: five lowest 1 − P(no significant move).
- ✗: the record is more than five points below a rule predicting the stock's most common matured outcome.
