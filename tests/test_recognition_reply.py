"""Sign recognition always answers a recording: the gloss it replies with, and the
sentence the API returns for it.

Run from the backend directory:
    .venv/Scripts/python.exe -m unittest tests.test_recognition_reply
"""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import limiter
from deploy.wlasl import GATE, reply_gloss


class ReplyGlossTest(unittest.TestCase):
    def test_signs_above_the_gate_are_the_reply(self):
        tops = [("hello", GATE + 0.2), ("you", GATE - 0.1), ("name", GATE + 0.05)]
        self.assertEqual(reply_gloss(tops), (["hello", "name"], True))

    def test_without_a_confident_sign_every_window_is_guessed_in_order(self):
        tops = [("hello", 0.1), ("hello", 0.2), ("friend", 0.05)]
        self.assertEqual(reply_gloss(tops), (["hello", "friend"], False))

    def test_a_sign_held_across_windows_is_one_gloss(self):
        tops = [("yes", GATE + 0.1), ("yes", GATE + 0.3), ("no", GATE + 0.1), ("yes", GATE + 0.1)]
        self.assertEqual(reply_gloss(tops), (["yes", "no", "yes"], True))

    def test_no_windows_is_an_error(self):
        with self.assertRaises(ValueError):
            reply_gloss([])


class Recognizer:
    def __init__(self, result):
        self.recognize = mock.Mock(remote=mock.Mock(return_value=result))


class VideoToSentenceTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        limiter.reset()

    def post(self, result):
        with mock.patch("app.routes.pipeline.get_wlasl", return_value=Recognizer(result)), \
             mock.patch("app.services.llm._complete", return_value="Hello, nice to meet you.") as llm, \
             mock.patch("app.routes.pipeline.retain_debug_clip"):
            res = self.client.post("/video-to-sentence", data={"video": (io.BytesIO(b"x" * 4000), "phrase.webm")},
                                   content_type="multipart/form-data")
        return res, llm

    def test_a_guessed_gloss_still_returns_a_sentence(self):
        res, llm = self.post({"gloss": ["hello", "meet"], "confident": False})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["sentence"], "Hello, nice to meet you.")
        self.assertFalse(res.get_json()["confident"])
        self.assertEqual(llm.call_args.args[1], "hello meet")

    def test_an_empty_gloss_from_the_recognizer_is_an_error(self):
        res, llm = self.post({"gloss": [], "confident": False})
        self.assertEqual(res.status_code, 500)
        llm.assert_not_called()
        self.assertIn("empty gloss", res.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
