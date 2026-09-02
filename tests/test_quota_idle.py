from __future__ import annotations

import unittest

from app.db.quota import idle_stop_reason


class IdleStopReasonTests(unittest.TestCase):
    def test_keeps_running_while_a_job_is_in_flight(self) -> None:
        self.assertIsNone(
            idle_stop_reason(quota_full=True, has_in_flight=True, has_claimable=False)
        )
        self.assertIsNone(
            idle_stop_reason(quota_full=False, has_in_flight=True, has_claimable=False)
        )

    def test_stops_when_daily_quota_is_full(self) -> None:
        self.assertEqual(
            idle_stop_reason(quota_full=True, has_in_flight=False, has_claimable=True),
            "daily quota full",
        )

    def test_keeps_running_when_a_job_is_claimable(self) -> None:
        self.assertIsNone(
            idle_stop_reason(quota_full=False, has_in_flight=False, has_claimable=True)
        )

    def test_stops_when_only_tomorrow_jobs_remain(self) -> None:
        self.assertEqual(
            idle_stop_reason(quota_full=False, has_in_flight=False, has_claimable=False),
            "nothing claimable",
        )


if __name__ == "__main__":
    unittest.main()
