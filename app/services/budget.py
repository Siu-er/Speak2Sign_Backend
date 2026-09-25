"""Spend ceilings for the routes that call a paid service.

The counters live in a Modal Dict rather than in process memory because the API
host restarts freely, and a count held in memory would reset with it and leave
the ceiling unenforced.
"""

from app import config

_store = None


class BudgetExhausted(RuntimeError):
    """A counter has reached its ceiling; the caller must not make the paid call."""


def _dict():
    global _store
    if _store is None:
        import modal
        _store = modal.Dict.from_name(config.BUDGET_DICT_NAME, create_if_missing=True)
    return _store


def consume(counter: str, limit: int) -> None:
    """Record one paid call, raising BudgetExhausted once the ceiling is reached."""
    store = _dict()
    used = store.get(counter, 0)
    if used >= limit:
        raise BudgetExhausted(f"{counter} budget exhausted ({used}/{limit} used)")
    store[counter] = used + 1
