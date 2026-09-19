import unittest
from g1cap.arena_session import session_limits
class SessionLimitTests(unittest.TestCase):
    def test_defaults_and_explicit_bounded_transfer_limits(self):
        self.assertEqual(session_limits({}),{'wall_timeout_s':900.,'worker_timeout_s':600.})
        self.assertEqual(session_limits({'wall_timeout_s':1800.,'worker_timeout_s':1500.}),
                         {'wall_timeout_s':1800.,'worker_timeout_s':1500.})
    def test_invalid_or_inconsistent_limits_rejected(self):
        for r in [{'wall_timeout_s':True},{'wall_timeout_s':float('inf')},{'wall_timeout_s':3601},
                  {'worker_timeout_s':0},{'worker_timeout_s':1000},
                  {'wall_timeout_s':1200,'worker_timeout_s':1300}]:
            with self.subTest(recipe=r),self.assertRaises(ValueError):session_limits(r)
