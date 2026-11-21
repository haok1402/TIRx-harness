"""Pytest plugin: after collection, write one line per item to $DUMP_COLLECTION:
'<nodeid>\t<skip|run>\t<xfail|->\t<skip reason>'. Static markers only: skip markers the suite gates add, and
xfail markers whose condition is true (or unconditional). Runtime request.applymarker xfails are not seen."""
import os


def _active_xfail(item):
    for mark in item.iter_markers("xfail"):
        cond = mark.args[0] if mark.args else mark.kwargs.get("condition", True)
        if bool(cond):
            return True
    return False


def pytest_collection_finish(session):
    path = os.environ["DUMP_COLLECTION"]
    with open(path, "w") as f:
        for item in session.items:
            skips = list(item.iter_markers("skip"))
            reason = skips[0].kwargs.get("reason", "") if skips else ""
            f.write(f"{item.nodeid}\t{'skip' if skips else 'run'}\t{'xfail' if _active_xfail(item) else '-'}\t{reason}\n")
