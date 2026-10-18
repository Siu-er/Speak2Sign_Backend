"""Room relay behaviour, exercised through Flask-SocketIO's test client.

Run from the backend directory:
    .venv/Scripts/python.exe -m unittest tests.test_relay
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

from app.extensions import socketio
from app.relay import sockets

MEETING = "abc-defg-hij"

app = Flask(__name__)
socketio.init_app(app, async_mode="threading", manage_session=False)


def client():
    return socketio.test_client(app)


def received(c, event):
    # The test client delivers the reserved "message" event as its bare payload
    # and every other event as an argument list.
    return [m["args"] if event == "message" else m["args"][0]
            for m in c.get_received() if m["name"] == event]


class MeetingRoomTest(unittest.TestCase):
    def setUp(self):
        sockets.rooms.clear()

    def join(self, role, code=MEETING):
        c = client()
        ack = c.emit("join_meeting", {"meetingCode": code, "role": role}, callback=True)
        return c, ack

    def test_first_participant_creates_the_room(self):
        _, ack = self.join("signer")
        self.assertEqual(ack["result"]["roomId"], "meet:" + MEETING)
        self.assertEqual(ack["result"]["counts"], {"speaker": 0, "signer": 1})

    def test_room_holds_any_number_of_members(self):
        for _ in range(4):
            self.join("speaker")
        _, ack = self.join("signer")
        self.assertTrue(ack["ok"])
        self.assertEqual(ack["result"]["counts"], {"speaker": 4, "signer": 1})

    def test_message_reaches_every_member_of_the_opposite_role_only(self):
        speaker_a, ack = self.join("speaker")
        speaker_b, _ = self.join("speaker")
        signer_a, _ = self.join("signer")
        signer_b, _ = self.join("signer")
        for c in (speaker_a, speaker_b, signer_a, signer_b):
            c.get_received()

        room_id = ack["result"]["roomId"]
        speaker_a.emit("message", {"roomId": room_id, "kind": "speech", "text": "hello"})

        expected = [{"kind": "speech", "text": "hello", "fromRole": "speaker"}]
        self.assertEqual(received(signer_a, "message"), expected)
        self.assertEqual(received(signer_b, "message"), expected)
        self.assertEqual(received(speaker_a, "message"), [])
        self.assertEqual(received(speaker_b, "message"), [])

    def test_presence_reports_counts_on_join_and_leave(self):
        signer, _ = self.join("signer")
        signer.get_received()
        speaker, _ = self.join("speaker")
        self.assertEqual(received(signer, "presence")[-1]["counts"], {"speaker": 1, "signer": 1})
        speaker.disconnect()
        self.assertEqual(received(signer, "presence")[-1]["counts"], {"speaker": 0, "signer": 1})

    def test_room_closes_when_the_last_member_leaves(self):
        c, _ = self.join("speaker")
        c.disconnect()
        self.assertEqual(sockets.rooms, {})

    def test_meetings_are_isolated_from_each_other(self):
        speaker, ack = self.join("speaker", "abc-defg-hij")
        other_signer, _ = self.join("signer", "xyz-wxyz-xyz")
        other_signer.get_received()
        speaker.emit("message", {"roomId": ack["result"]["roomId"], "kind": "speech", "text": "hi"})
        self.assertEqual(received(other_signer, "message"), [])

    def test_rejects_a_malformed_meeting_code(self):
        for code in ("ABC-DEFG-HIJ", "abcdefghij", "", None):
            _, ack = self.join("speaker", code)
            self.assertEqual(ack, {"ok": False, "error": "invalid_meeting_code"})

    def test_rejects_an_unknown_role(self):
        _, ack = self.join("viewer")
        self.assertEqual(ack, {"ok": False, "error": "invalid_role"})


class PairRoomTest(unittest.TestCase):
    def setUp(self):
        sockets.rooms.clear()

    def test_pair_relays_between_its_two_members_and_refuses_a_third(self):
        speaker = client()
        room_id = speaker.emit("create_room", callback=True)["roomId"]
        speaker.emit("join_room", {"roomId": room_id, "role": "speaker"}, callback=True)
        signer = client()
        ack = signer.emit("join_room", {"roomId": room_id, "role": "signer"}, callback=True)
        self.assertEqual(ack["result"]["counts"], {"speaker": 1, "signer": 1})

        signer.get_received()
        speaker.emit("message", {"roomId": room_id, "kind": "speech", "text": "hello"})
        self.assertEqual(received(signer, "message"),
                         [{"kind": "speech", "text": "hello", "fromRole": "speaker"}])

        third = client()
        ack = third.emit("join_room", {"roomId": room_id, "role": "signer"}, callback=True)
        self.assertEqual(ack, {"ok": False, "error": "room_full"})

    def test_join_of_an_unknown_pair_fails(self):
        ack = client().emit("join_room", {"roomId": "ABCDEF", "role": "signer"}, callback=True)
        self.assertEqual(ack, {"ok": False, "error": "room_not_found"})


if __name__ == "__main__":
    unittest.main()
