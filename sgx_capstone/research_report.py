"""Credential-free research review from supplied real research tables.

This is a frozen research review, not a current stock report or a live model
rerun. Values are rebuilt from retained CSVs; no observations are invented.
"""
import base64
import html
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def chart_image(frame):
    fig, ax = plt.subplots(figsize=(8,3.6),layout='constrained')
    labels = frame['model'].tolist()
    ax.barh(labels,frame['macro_f1'],color=['#78848e' if 'majority' in x or x=='rev1_rule'
                                        else '#287caf' for x in labels])
    ax.set_xlabel('Macro F1 (fixed relative target, same 330 rows)')
    ax.set_title('R11 relative-target comparison')
    ax.set_xlim(0,max(.5,float(frame.macro_f1.max())+.08))
    for idx, value in enumerate(frame.macro_f1):
        ax.text(value+.006,idx,f'{value:.3f}',va='center',fontsize=9)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140);plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def build(root: Path, output: Path | None = None):
    source=root/'outputs'/'research'
    summary=pd.read_csv(source/'R11_summary.csv')
    verdicts=pd.read_csv(source/'R11_verdicts.csv')
    consistency=pd.read_csv(source/'R13_champion_consistency.csv')
    replay=pd.read_csv(source/'R10_suppression_replay.csv')
    relative=summary[summary.target=='relative'].sort_values('macro_f1')
    displays=[]
    for decision, group in replay.groupby('decision'):
        displays.append({'Display':decision,'Matured rows':len(group),
                         'Accuracy':f'{(group.pred==group.actual).mean():.1%}'})
    passed=int((verdicts.verdict=='PROMOTE-CANDIDATE').sum())
    r14_path=source/'R14_outlook_reliability.csv'
    r14=pd.read_csv(r14_path) if r14_path.exists() else None
    if r14 is not None:
        rf=r14[r14.model=='RandomF'].iloc[0]
        rng=r14[r14.statement.str.startswith('S1')].iloc[0]
        r14_cols=['model','statement','n','dates','quiet_move_rate','rest_move_rate',
                  'dates_quiet_strictly_lower','dates_tied','p_dates_one_sided',
                  'nonoverlap_dates_quiet_strictly_lower','nonoverlap_dates','auc_move',
                  'auc_direction_among_moves','range_coverage']
        r14_html=f"""<h2>What can be said about the next three sessions (R14)</h2>
    <p>Out-of-fold walk-forward predictions, {int(rf.dates)} dates; an event is a
    three-session move of more than 1% ({html.escape(str(rf.event_definition))}). Each day
    the five stocks with the lowest RandomForest move score (1 - P(no significant move))
    moved {rf.quiet_move_rate:.1%} of the time, against {rf.rest_move_rate:.1%} for the
    other ten; their rate was lower on {int(rf.dates_quiet_strictly_lower)} of
    {int(rf.dates)} dates, with {int(rf.dates_tied)} tied (one-sided sign test
    p {rf.p_dates_one_sided:.3f}; on non-overlapping windows
    {int(rf.nonoverlap_dates_quiet_strictly_lower)} of {int(rf.nonoverlap_dates)},
    p {rf.p_nonoverlap_one_sided:.2f}). A ranking from 20-day volatility alone does
    about as well. A 20-day volatility range held {rng.range_coverage:.1%} of the time.
    Direction did not separate (AUC {rf.auc_direction_among_moves:.3f}). The rule was
    defined after the calibration study, so it is exploratory and frozen with an
    acceptance bar for new dates.</p>
    {r14[[c for c in r14_cols if c in r14.columns]].round(3).to_html(index=False,border=0,na_rep='')}"""
    else:
        r14_html=''

    leads=int(consistency.winner_leads_f1.sum())
    cols=['model','n','days','macro_f1','bal_acc','brier','IC','IC_t','spread%','target']
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8">
    <title>SGX capstone frozen research review</title><style>
    body{{font:16px/1.55 system-ui,sans-serif;max-width:1080px;margin:32px auto;padding:0 20px;color:#172b3a}}
    h1,h2{{line-height:1.2}}h2{{margin-top:36px}}p{{max-width:900px}}
    table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}
    th,td{{padding:7px;border-bottom:1px solid #d5dce1;text-align:right}}th:first-child,td:first-child{{text-align:left}}
    th{{background:#eaf1f7}}img{{width:100%;max-width:800px}}.note{{padding:16px;background:#edf3f7}}
    @media print{{body{{font-size:11px;margin:0}}table{{font-size:9px}}}}
    </style><h1>SGX capstone frozen research review</h1>
    <p class="note">This reproducible review uses the supplied historical research outputs.
    It is available without a database or API key. It is not today's investor report and
    does not claim a new live experiment or a profitable stock ranking.</p>
    <h2>What the forecast experiments establish</h2>
    <p>{passed}/{len(verdicts)} candidate tests passed all R11 acceptance criteria.
    The retained R11 evaluation covers {int(relative.days.min())} test dates and
    {int(relative.n.min())} stock-days per model. The criteria are exploratory;
    these are reused research dates, not an independently preregistered future study.</p>
    <img alt="Relative target macro F1 by model" src="data:image/png;base64,{chart_image(relative)}">
    <p>Interpretation: a model must beat simple baselines on matched rows.
    Positive rank correlation or a small gross spread alone is insufficient.
    Spread is expressed in percentage points before trading costs, not a
    demonstrated implementable long-short profit.</p>
    {summary[cols].round(4).to_html(index=False,border=0,escape=True)}
    {r14_html}
    <h2>Stock-wise consistency (R13)</h2>
    <p>The pooled macro-F1 winner leads the other models on {leads}/{len(consistency)} stocks,
    below the 80% target. The pooled winner was selected on the same
    dates, so this is a retrospective consistency diagnostic, not an independent
    forward-date acceptance result. Model agreement is a different statistic.</p>
    {consistency.round(4).to_html(index=False,border=0,escape=True)}
    <h2>Historical display control</h2>
    <p>R10 replay covers {html.escape(str(replay.date.min()))} to
    {html.escape(str(replay.date.max()))}. It reconstructs decisions from prior
    matured observations. The displayed subset's accuracy does not establish
    an investment advantage or validate the gate on future dates.</p>
    {pd.DataFrame(displays).to_html(index=False,border=0)}
    <h2>Investor use that still needs validation</h2>
    <p>The daily evidence page connects a company's dated headlines with its
    observed price move and the model's reliability limits. T5 evidence strength
    is an internal rule score; T6 outputs are model class probabilities. A reader
    study must test whether users can locate the source, separate a common market
    move from a company-specific move, and understand withheld forecasts.</p>
    <h2>Calibration and deployment boundary</h2>
    <p>The bundled artifact passes an out-of-fold metadata check, but its old
    row-based TimeSeriesSplit is not proof of fully purged calendar-date isolation.
    The new candidate utility groups dates, removes labels not matured before each
    calibration fold, and fits preprocessing inside folds. It has offline leakage
    tests; no candidate has been promoted or shown to improve future returns.</p>
    <p>Source tables: R11_summary.csv, R11_verdicts.csv,
    R13_champion_consistency.csv, R10_suppression_replay.csv and
    R14_outlook_reliability.csv.</p></html>'''
    destination=output or root/'outputs'/'research_review'/'index.html'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(body)
    return destination
