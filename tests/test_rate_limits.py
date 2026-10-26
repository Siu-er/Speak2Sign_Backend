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

    def device(self, device_id):
        return self.client.post("/translate-to-english", json={"text": ""}, headers={
            "X-Forwarded-For": "203.0.113.5", "X-S2S-Device-Id": device_id}).status_code

    def test_devices_on_one_network_have_their_own_allowances(self):
        first, second = "6f1c2b9e-3d4a-4c5b-8e7f-0a1b2c3d4e5f", "0d9e8f7a-6b5c-4d3e-9f2a-1b0c9d8e7f6a"
        self.assertEqual([self.device(first) for _ in range(PAID + 1)][-1], 429)
        self.assertEqual(self.device(second), 400)

    def test_a_malformed_device_id_is_counted_by_network_address(self):
        for i in range(PAID):
            self.device(f"not-a-uuid-{i}")
        self.assertEqual(self.translate("203.0.113.5").status_code, 429)

    def test_a_team_device_key_lifts_the_limit(self):
        with mock.patch.object(config, "RATE_LIMIT_EXEMPT_KEYS", frozenset({"team-booth-key"})):
            post = lambda key: self.client.post("/translate-to-english", json={"text": ""}, headers={
                "X-Forwarded-For": "203.0.113.5", "X-S2S-Device-Key": key}).status_code
            self.assertEqual([post("team-booth-key") for _ in range(PAID * 3)], [400] * PAID * 3)
            self.assertEqual([post("guessed-key") for _ in range(PAID + 1)][-1], 429)

    def test_the_key_header_is_ignored_when_no_keys_are_set(self):
        post = lambda: self.client.post("/translate-to-english", json={"text": ""}, headers={
            "X-Forwarded-For": "203.0.113.5", "X-S2S-Device-Key": ""}).status_code
        self.assertEqual([post() for _ in range(PAID + 1)][-1], 429)

    def test_free_routes_are_limited_by_the_default(self):
        get = lambda: self.client.get("/", headers={"X-Forwarded-For": "203.0.113.7"}).status_code
        codes = [get() for _ in range(DEFAULT + 1)]
        self.assertEqual(codes[-2:], [200, 429])


if __name__ == "__main__":
    unittest.main()
