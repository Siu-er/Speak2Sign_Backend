"""Request limits per client: the paid routes, the default on every other route,
and the client address read behind the hosting proxy.

Run from the backend directory:
    .venv/Scripts/python.exe -m unittest tests.test_rate_limits
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limits import parse

from app import config, create_app
from app.extensions import limiter

PAID = parse(config.RATE_LIMIT_PAID).amount
DEFAULT = parse(config.RATE_LIMIT_DEFAULT).amount


class RateLimitTest(unittest.TestCase):
    def setUp(self):
        hops = mock.patch.object(config, "TRUSTED_PROXY_HOPS", 1)
        hops.start()
        self.addCleanup(hops.stop)
        self.client = create_app().test_client()
        limiter.reset()

    def translate(self, forwarded_for):
        # An empty body is refused after the limit is counted, so no paid call is made.
        return self.client.post("/translate-to-english", json={"text": ""},
                                headers={"X-Forwarded-For": forwarded_for, "Origin": "https://s2s-app-alpha.vercel.app"})

    def test_a_paid_route_refuses_a_client_past_its_limit_with_a_readable_error(self):
        codes = [self.translate("203.0.113.5").status_code for _ in range(PAID + 1)]
        self.assertEqual(codes, [400] * PAID + [429])
        res = self.translate("203.0.113.5")
        self.assertIn("Too many requests", res.get_json()["error"])
        self.assertEqual(res.headers["Access-Control-Allow-Origin"], "https://s2s-app-alpha.vercel.app")

    def test_a_forged_forwarded_address_does_not_open_a_new_allowance(self):
        for i in range(PAID):
            self.translate(f"198.51.100.{i}, 203.0.113.5")
        self.assertEqual(self.translate("198.51.100.99, 203.0.113.5").status_code, 429)

    def test_each_client_has_its_own_allowance(self):
        for _ in range(PAID):
            self.translate("203.0.113.5")
        self.assertEqual(self.translate("203.0.113.6").status_code, 400)

    def test_free_routes_are_limited_by_the_default(self):
        get = lambda: self.client.get("/", headers={"X-Forwarded-For": "203.0.113.7"}).status_code
        codes = [get() for _ in range(DEFAULT + 1)]
        self.assertEqual(codes[-2:], [200, 429])


if __name__ == "__main__":
    unittest.main()
