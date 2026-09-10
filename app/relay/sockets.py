"""Two-device room relay over Socket.IO. The server holds no conversation state;
it only relays messages between the two members of a room."""

import logging
import secrets

from flask import request
from flask_socketio import emit, join_room, leave_room

from app.extensions import socketio

logger = logging.getLogger(__name__)

rooms = {}  # roomId -> { sid: role }


def _new_room_id():
    while True:
        room_id = secrets.token_hex(3).upper()  # 6 hex chars
        if room_id not in rooms:
            return room_id


@socketio.on("create_room")
def on_create_room():
    room_id = _new_room_id()
    rooms[room_id] = {}
    logger.info(f"Room created: {room_id}")
    return {"roomId": room_id}


@socketio.on("join_room")
def on_join_room(data):
    data = data or {}
    room_id = data.get("roomId")
    role = data.get("role")

    if room_id not in rooms:
        return {"ok": False, "error": "room_not_found"}
    if role not in ("speaker", "signer"):
        return {"ok": False, "error": "invalid_role"}
    if len(rooms[room_id]) >= 2:
        return {"ok": False, "error": "room_full"}

    join_room(room_id)
    rooms[room_id][request.sid] = role
    emit("peer_joined", {"role": role}, to=room_id, include_self=False)
    logger.info(f"Join {room_id} as {role} (peers={len(rooms[room_id])})")
    return {
        "ok": True,
        "result": {"roomId": room_id, "role": role, "peers": len(rooms[room_id])},
    }


@socketio.on("message")
def on_message(data):
    data = data or {}
    room_id = data.get("roomId")
    members = rooms.get(room_id)
    if not members or request.sid not in members:
        return
    emit(
        "message",
        {
            "kind": data.get("kind"),
            "text": data.get("text"),
            "fromRole": members[request.sid],
        },
        to=room_id,
        include_self=False,
    )


@socketio.on("disconnect")
def on_disconnect():
    for room_id, members in list(rooms.items()):
        if request.sid in members:
            del members[request.sid]
            leave_room(room_id)
            emit("peer_left", to=room_id)
            logger.info(f"Leave {room_id} (peers={len(members)})")
            if not members:
                rooms.pop(room_id, None)
                logger.info(f"Room closed: {room_id}")
