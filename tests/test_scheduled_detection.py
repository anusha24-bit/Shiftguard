from __future__ import annotations

import unittest

import pandas as pd

from src.detection.scheduled import detect_scheduled_shifts


class TestScheduledDetection(unittest.TestCase):
    def test_detect_scheduled_shifts_aligns_to_nearest_bar(self) -> None:
        datetimes = pd.date_range('2020-12-28 00:00:00', periods=120, freq='4h')
        aligned_bar = pd.Timestamp('2021-01-05 08:00:00')
        aligned_idx = int(datetimes.get_loc(aligned_bar))

        feature_1 = []
        feature_2 = []
        for idx in range(len(datetimes)):
            if idx < aligned_idx:
                feature_1.append(float(idx % 5))
                feature_2.append(float((idx % 7) - 3))
            else:
                feature_1.append(float(100 + (idx % 5)))
                feature_2.append(float(50 + (idx % 7)))

        features_df = pd.DataFrame({
            'datetime_utc': datetimes,
            'feature_a': feature_1,
            'feature_b': feature_2,
        })
        calendar_df = pd.DataFrame([{
            'date': '2021-01-05',
            'time_utc': '7:00am',
            'impact_level': 'High',
            'event_name': 'CPI Release',
        }])

        shifts = detect_scheduled_shifts(
            features_df,
            calendar_df,
            feature_cols=['feature_a', 'feature_b'],
            window_size=20,
        )

        self.assertEqual(len(shifts), 1)
        self.assertEqual(shifts[0]['datetime_utc'], '2021-01-05 08:00:00')
        self.assertEqual(shifts[0]['event_names'], 'CPI Release')


if __name__ == '__main__':
    unittest.main()
