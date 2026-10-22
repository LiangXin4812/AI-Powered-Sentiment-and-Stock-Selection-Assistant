"""Offline integration checks; no feed calls, credentials or serving writes."""
import contextlib
import io
import pickle
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sgx_capstone.configuration import ROOT, apply_config, load_config, configure_report_modules
from sgx_capstone.data_access import load_settings, write_primary, latest_complete_date
from sgx_capstone.benchmarks import prior_majority_labels
from sgx_capstone.research.purged_calibration import fit_linear_candidate, purged_date_folds


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.sha = load_config()
        sys.path.insert(0, str(ROOT / 'src'))

    def test_serving_artifacts_infer_real_feature_rows(self):
        with open(self.cfg['paths']['t6_artifacts'], 'rb') as stream:
            artifact = pickle.load(stream)
        rows = pd.read_csv(ROOT / 'tests' / 'fixtures' / 'feature_rows.csv')
        # Missing inputs take the training mean, as in serving (task6_runtime).
        mean = pd.Series(artifact['scaler'].mean_, index=artifact['feature_names'])
        features = rows[artifact['feature_names']].fillna(mean)
        for name in ('logreg_calibrated', 'rf_calibrated', 'xgb_calibrated'):
            model = artifact[name]
            input_rows = artifact['scaler'].transform(features) if name.startswith('logreg') else features
            probabilities = model.predict_proba(input_rows)
            self.assertEqual(probabilities.shape, (len(rows), 3))
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
        configure_report_modules(self.cfg)
        import task7_report
        self.assertEqual(task7_report.T6_THRESHOLD_DEFAULT, artifact['return_threshold'])

    def test_primary_writes_and_complete_date_use_all_covered_stocks(self):
        engine = create_engine('sqlite:///:memory:')
        with engine.begin() as conn:
            for ddl in (
                'CREATE TABLE technical_signals (ticker TEXT, date TEXT, weekly_signal TEXT, UNIQUE(ticker,date))',
                'CREATE TABLE news_articles (ticker TEXT,date TEXT,category TEXT)',
                'CREATE TABLE macro_indicators (date TEXT)',
                'CREATE TABLE market_options (date TEXT)',
            ):
                conn.execute(text(ddl))
            for statement in (
                "INSERT INTO news_articles VALUES ('A','2026-09-21','company')",
                "INSERT INTO news_articles VALUES ('POLITICAL','2026-09-21','political')",
                "INSERT INTO macro_indicators VALUES ('2026-09-21')",
                "INSERT INTO market_options VALUES ('2026-09-21')",
            ):
                conn.execute(text(statement))
        row = dict(ticker='A', date='2026-09-21', weekly_signal='neutral')
        write_primary(engine, 'technical_signals', row)
        write_primary(engine, 'technical_signals', row)
        self.assertIsNone(latest_complete_date(engine, ['A','B']))
        write_primary(engine, 'technical_signals', {**row,'ticker':'B'})
        self.assertEqual(latest_complete_date(engine, ['A','B']), '2026-09-21')
        with engine.connect() as conn:
            self.assertEqual(conn.execute(text('SELECT COUNT(*) FROM technical_signals')).scalar(), 2)

    def test_baseline_cannot_see_future_outcomes(self):
        history = [dict(date='2026-09-01', maturity='2026-09-04', actual='bearish'),
                   dict(date='2026-09-02', maturity='2026-09-05', actual='bullish'),
                   dict(date='2026-09-03', maturity='2026-09-06', actual='bullish')]
        evaluation = [dict(date='2026-09-02'),dict(date='2026-09-05')]
        self.assertEqual(prior_majority_labels(history,evaluation), ['neutral','bearish'])
        changed = [*history[:1],dict(date='2026-09-02',maturity='2026-09-05',actual='neutral')]
        self.assertEqual(prior_majority_labels(changed,evaluation), ['neutral','bearish'])

    def test_purged_folds_and_scaler_never_fit_future_dates(self):
        days = pd.bdate_range('2026-01-01', periods=60)
        dates = np.repeat(days,3)
        maturity = np.repeat(days + pd.offsets.BDay(3),3)
        features = pd.DataFrame({'trend':np.arange(len(dates),dtype=float),
                                 'signal':np.tile([-1.,0.,1.],len(days))})
        labels = np.tile(['bearish','neutral','bullish'],len(days))
        folds, _ = purged_date_folds(dates,maturity,3)
        for fit,cal in folds:
            self.assertFalse(set(dates[fit]) & set(dates[cal]))
            self.assertLess(max(maturity[fit]),min(dates[cal]))
        model, audit = fit_linear_candidate(features,labels,dates,maturity,n_splits=3)
        scaler = model.calibrated_classifiers_[0].estimator.named_steps['standardscaler']
        np.testing.assert_allclose(scaler.mean_,features.iloc[folds[0][0]].mean())
        self.assertFalse(np.allclose(scaler.mean_,features.mean()))
        self.assertEqual(len(audit),3)

    def test_historical_report_hides_future_label_and_uses_same_day_price(self):
        configure_report_modules(self.cfg)
        import task7_report as report
        engine, conn = MagicMock(), MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        t5_result, t6_result, champion = MagicMock(), MagicMock(), MagicMock()
        t5_result.mappings.return_value.fetchone.return_value = {
            'sentiment':'bullish','confidence':.68,'warning_level':None}
        # The database now knows a +3% future move, which was unavailable on 22 Sep.
        t6_result.fetchone.return_value = ('neutral',.2,.6,.2,.6,'RandomF',3.,'bullish','2026-09-25')
        champion.fetchone.return_value = None
        conn.execute.side_effect = [t5_result,t6_result,champion]
        with patch.object(report,'_t5_available_columns',return_value={'select':['sentiment','confidence']}), \
             patch.object(report,'compute_track_record',return_value={'t6_n':0,'t6_accuracy':None}), \
             patch.object(report,'compute_price_context',return_value={'as_of':'2026-09-22','return_pct':-2.}):
            data = report.build_report_input(engine,'TEST.SI','2026-09-22',tickers=['TEST.SI'])
        self.assertIsNone(data['actual']['return_3d'])
        self.assertFalse(data['actual']['matured'])
        self.assertEqual(data['fusion']['axis_confirmation'],'contradicted')

    def test_report_is_blocked_when_deterministic_text_fails_checks(self):
        configure_report_modules(self.cfg)
        import task7_generate as generate
        inp = {'status': 'complete', 'ticker': 'TEST.SI', 'report_date': '2026-09-22',
               'generated_at': '2026-09-22T18:00', 'attention': {'level': 'ROUTINE'},
               'fusion': {'archetype': 'quiet'}, 'reliability': {}, 'actual': {}}
        with patch.object(generate, 'build_report_input', return_value=inp), \
             patch.object(generate, 'fetch_attribution', return_value={}), \
             patch.object(generate, 'render_fallback', return_value={'today': 'You should buy now.'}), \
             patch.object(generate, 'validate_compliance', return_value=(False, ['R1 advice'])):
            out = generate.generate_and_store(MagicMock(), 'TEST.SI', '2026-09-22',
                                              groq_api_key=None, store=False, verbose=False)
        self.assertEqual(out['status'], 'blocked')
        self.assertEqual(out['mode'], 'blocked')
        self.assertNotIn('You should buy', out['full_text'])
        self.assertIn('Report withheld', out['full_text'])
        self.assertIn(generate.DISCLAIMER, out['full_text'])

    def test_selected_runtime_loads_and_technical_math_is_finite(self):
        settings = load_settings(ROOT)
        cfg = {'technical':settings['solution']['technical'],
               'storage':settings['system']['storage']}
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = 0
        conn.execute.return_value.fetchone.return_value = None
        conn.execute.return_value.fetchall.return_value = []
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = conn
        engine.begin.return_value.__enter__.return_value = conn
        namespace = dict(CFG=self.cfg, CONFIG_SHA=self.sha, CONFIG=cfg,
                         SUPABASE_ENGINE=engine, SGX_STOCKS=settings['sgx']['tickers'])
        apply_config(self.cfg, namespace)
        namespace['apply_config'] = lambda cfg: apply_config(cfg,namespace)
        namespace['mirror_to_supabase'] = lambda table,row: write_primary(engine,table,row)
        from sgx_capstone.__main__ import _execute
        with contextlib.redirect_stdout(io.StringIO()):
            for name in ('task6_features','task5_runtime','task5_llm_config',
                         'task5_daily','task6_runtime','task1_news','task2_technical',
                         'task3_macro','task4_options'):
                _execute(name, namespace)
                apply_config(self.cfg,namespace)
        close = 100 + np.arange(120) * .1 + np.sin(np.arange(120))
        frame = pd.DataFrame(dict(Close=close,High=close+1,Low=close-1,
                                  Open=close-.2,Volume=np.full(120,10000)),
                             index=pd.bdate_range('2026-01-01',periods=120))
        signals = namespace['detect_signals'](namespace['calculate_indicators'](frame))
        self.assertTrue(np.isfinite(signals['rsi']))
        self.assertTrue(np.isfinite(signals['atr']))
        self.assertIn('build_llm_prompt',namespace)
        self.assertIn('predict_task6',namespace)


if __name__ == '__main__':
    unittest.main()
