"""Room relay over Socket.IO. The server holds no conversation state; it relays
each message from a member to every member of the opposite role in the room.

Two kinds of room exist. A pair is created by one device and joined by a second
through its code, and holds at most two members. A meeting room is keyed by a
Google Meet meeting code, is created by whichever participant joins first, and
holds any number of speakers and signers.
"""

import logging
import re
import secrets
from dataclasses import dataclass, field
from typing import Dict, Optional

from flask import request
from flask_socketio import emit, join_room, leave_room

from app.extensions import socketio

logger = logging.getLogger(__name__)

ROLES = ("speaker", "signer")
OPPOSITE_ROLE = {"speaker": "signer", "signer": "speaker"}
PAIR_CAPACITY = 2
# Google Meet meeting codes: three, four and three lowercase letters.
MEETING_CODE = re.compile(r"^[a-z]{3}-[a-z]{4}-[a-z]{3}$")
MEETING_ROOM_PREFIX = "meet:"


@dataclass
class Room:
    capacity: Optional[int]
    members: Dict[str, str] = field(default_factory=dict)  # sid -> role

    def counts(self):
        return {role: sum(1 for r in self.members.values() if r == role) for role in ROLES}


rooms: Dict[str, Room] = {}


def _role_channel(room_id, role):
    return f"{room_id}/{role}"


def _new_pair_id():
    while True:
        room_id = secrets.token_hex(3).upper()
        if room_id not in rooms:
            return room_id


def _broadcast_presence(room_id):
    emit("presence", {"counts": rooms[room_id].counts()}, to=room_id)


def _join(room_id, role):
    room = rooms[room_id]
    room.members[request.sid] = role
    join_room(room_id)
    join_room(_role_channel(room_id, role))
    _broadcast_presence(room_id)
    logger.info(f"Join {room_id} as {role} (members={len(room.members)})")
    return {
        "ok": True,
        "result": {"roomId": room_id, "role": role, "counts": room.counts()},
    }


@socketio.on("create_room")
def on_create_room():
    room_id = _new_pair_id()
    rooms[room_id] = Room(capacity=PAIR_CAPACITY)
    logger.info(f"Room created: {room_id}")
    return {"roomId": room_id}


@socketio.on("join_room")
def on_join_room(data):
    data = data or {}
    room_id = data.get("roomId")
    role = data.get("role")

    room = rooms.get(room_id)
    if room is None:
        return {"ok": False, "error": "room_not_found"}
    if role not in ROLES:
        return {"ok": False, "error": "invalid_role"}
    if room.capacity is not None and len(room.members) >= room.capacity:
        return {"ok": False, "error": "room_full"}
    return _join(room_id, role)


@socketio.on("join_meeting")
def on_join_meeting(data):
    data = data or {}
    code = data.get("meetingCode")
    role = data.get("role")

    if not isinstance(code, str) or not MEETING_CODE.match(code):
        return {"ok": False, "error": "invalid_meeting_code"}
    if role not in ROLES:
        return {"ok": False, "error": "invalid_role"}
    room_id = MEETING_ROOM_PREFIX + code
    if room_id not in rooms:
        rooms[room_id] = Room(capacity=None)
        logger.info(f"Room created: {room_id}")
    return _join(room_id, role)


@socketio.on("message")
def on_message(data):
    data = data or {}
    room_id = data.get("roomId")
    room = rooms.get(room_id)
    if room is None or request.sid not in room.members:
        return
    role = room.members[request.sid]
    emit(
        "message",
        {"kind": data.get("kind"), "text": data.get("text"), "fromRole": role},
        to=_role_channel(room_id, OPPOSITE_ROLE[role]),
    )


@socketio.on("disconnect")
def on_disconnect():
    for room_id, room in list(rooms.items()):
        role = room.members.pop(request.sid, None)
        if role is None:
            continue
        leave_room(room_id)
        leave_room(_role_channel(room_id, role))
        logger.info(f"Leave {room_id} (members={len(room.members)})")
        if room.members:
            _broadcast_presence(room_id)
        else:
            rooms.pop(room_id, None)
            logger.info(f"Room closed: {room_id}")
