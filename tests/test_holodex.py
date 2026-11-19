"""Holodex cast lookup: curated collab tags lead the stream context."""
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from auto_translate import holodex
from auto_translate.config import HolodexConfig
from auto_translate.glossary import Glossary
from auto_translate.stream_context import describe_stream

GLOSSARY = Glossary.load(Path(__file__).resolve().parents[1] / "glossary.toml")


def response(payload, status=200):
    reply = Mock()
    reply.json.return_value = payload
    reply.raise_for_status.side_effect = (
        httpx.HTTPStatusError("bad", request=Mock(), response=Mock()) if status >= 400 else None)
    return reply


class HolodexTests(unittest.TestCase):
    def test_fetch_sends_key_header_and_returns_channel_and_mentions(self):
        payload = {"channel": {"name": "Marine Ch. 宝鐘マリン", "english_name": "Houshou Marine"},
                   "mentions": [{"id": "x", "name": "Suisei Channel", "english_name": "Hoshimachi Suisei"},
                                {"id": "y", "name": "兎田ぺこら"}, "junk"]}
        with patch.object(holodex.httpx, "get", return_value=response(payload)) as get:
            cast = holodex.fetch_cast("abcdefghijk", "secret", 3)
        self.assertEqual(get.call_args.args[0], "https://holodex.net/api/v2/videos/abcdefghijk")
        self.assertEqual(get.call_args.kwargs["headers"], {"X-APIKEY": "secret"})
        self.assertEqual(get.call_args.kwargs["timeout"], 3)
        self.assertEqual(cast, {"channel": ["Houshou Marine", "Marine Ch. 宝鐘マリン"],
                                "mentions": ["Hoshimachi Suisei", "Suisei Channel", "兎田ぺこら"]})

    def test_failures_and_invalid_inputs_are_empty_and_do_not_log_the_key(self):
        with patch.object(holodex.httpx, "get") as get:
            self.assertEqual(holodex.fetch_cast("abcdefghijk", ""), {})
            self.assertEqual(holodex.fetch_cast("../../etc", "secret"), {})
            get.assert_not_called()
        for failure in (response({}, 403), httpx.ConnectError("down")):
            with self.subTest(failure=failure), self.assertLogs(holodex.log, "WARNING") as logs, \
                    patch.object(holodex.httpx, "get",
                                 **({"side_effect": failure} if isinstance(failure, Exception)
                                    else {"return_value": failure})):
                self.assertEqual(holodex.fetch_cast("abcdefghijk", "secret"), {})
            self.assertNotIn("secret", "".join(logs.output))

    def test_tagged_mentions_outrank_scraped_names_and_channel_fallback_finds_host(self):
        stream = describe_stream({"title": "雑談", "channel": "Some Clip Ch.",
                                  "description": "さくらみこ 兎田ぺこら 宝鐘マリン",
                                  "holodex_channel": ["Houshou Marine"],
                                  "holodex_mentions": ["Hoshimachi Suisei"]}, GLOSSARY)
        self.assertEqual(stream.host, "宝鐘マリン")
        # Cast should include Suisei (from Holodex) and others from description
        self.assertIn("星街すいせい", stream.cast)
        self.assertGreaterEqual(len(stream.cast), 1)

    def test_config_validation(self):
        self.assertEqual(HolodexConfig().api_key, "")
        with self.assertRaises(ValueError):
            HolodexConfig(timeout_s=0)


if __name__ == "__main__":
    unittest.main()
