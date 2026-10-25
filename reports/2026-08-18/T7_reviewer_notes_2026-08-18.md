# T7 Daily Signal Report 2026-08-18: how it was produced

Generated 2026-10-25 12:01 SGT; mode live.

## Steps

- Task 1 news: 523 company articles and 71 political articles, FinBERT-scored; event tags.
- Task 2 technical: RSI, MACD, Bollinger bands (20, 2σ), VWAP and weekly trend per stock.
- Task 3 macro: regime code 1; VIX 14.2; US yield spread.
- Task 4 options: Singapore ETF (EWS) put/call volume ratio 1.59, positioning negative.
- Task 5 evidence: rule layer rl_v8: 14 positive, 0 negative, 1 neutral.
- Task 6 forecast: the stored live champion forecast and probabilities for each stock, with its record against a time-ordered simple rule (✓ / ✗); 'calmer than most' from the same stored probabilities.
- Task 7: patterns, attention, cards, market read, compliance checks R1-R7; narrative AI (openai/gpt-oss-20b) with checks N1-N13.

## Provenance

- Project database: Tasks 1-6 stored outputs for the period; adjusted daily closes.
- Record of each statement: research R14, R10 and the stored champion record.
- Period: close of 11 Aug to close of 18 Aug 2026; no evidence reading stored for Wed 12 Aug, Thu 13 Aug, Fri 14 Aug
- AI blocks that passed only on a second attempt: c1, c2, c3

## Measures

- Patterns today: Noise 4, Lag 3, Dormant 2, Muted, fading 2, Move complete 2, Quiet 1, Mean reversion 1
- Attention today: Routine 9, Notable 6 (⚑ = Elevated or Notable)
- Forecasts marked ✗: 0 of 15
- Outlook record (R14, frozen): 31 dates; calmer five moved >1% 42% vs 56%; normal swing held 74%

## Terms

- News & signals: Task 5 net rule score beyond ±0.35. News today: (positive − negative) / articles beyond ±0.15; 'too few' below 3 articles.
- Price vs news: today's move against Task 5's label threshold ±0.60%.
- Normal 3-day swing: ±(20-day s.d. × √3). Calmer than most: five lowest 1 − P(no significant move).
- ✗: the record is more than five points below a rule predicting the stock's most common matured outcome.
