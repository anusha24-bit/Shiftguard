from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from src.dashboard import decision_utils


class TestDecisionUtils(unittest.TestCase):
    def test_auto_confirm_from_detection_writes_normalized_decisions(self) -> None:
        workspace_tmp = Path.cwd() / 'tests_tmp'
        workspace_tmp.mkdir(exist_ok=True)
        tmpdir = workspace_tmp / f'decisions_{uuid.uuid4().hex}'
        detection_dir = tmpdir / 'results' / 'detection'
        detection_dir.mkdir(parents=True)
        try:
            pd.DataFrame([
                {'datetime_utc': '2021-01-05 08:00:00', 'type': 'scheduled', 'severity': 4, 'event_names': 'CPI'},
                {'datetime_utc': '2021-01-06 12:00:00', 'type': 'unexpected', 'severity': 3, 'event_names': 'Shock'},
            ]).to_csv(detection_dir / 'EURUSD_shifts.csv', index=False)

            created = decision_utils.auto_confirm_from_detection('EURUSD', project_root=tmpdir)
            decisions = decision_utils.load_decisions('EURUSD', project_root=tmpdir)

            self.assertEqual(created, 2)
            self.assertEqual(sorted(decisions['decision'].unique().tolist()), ['auto_confirm'])
            self.assertIn('2021-01-05 08:00:00', decisions['datetime_utc'].tolist())
            self.assertIn('2021-01-06 12:00:00', decisions['datetime_utc'].tolist())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
