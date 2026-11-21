"""Pytest plugin: after collection, write one line per item to $DUMP_COLLECTION:
'<nodeid>\t<skip|run>\t<skip reason>' (static skip markers only, e.g. those the suite gates add)."""
import os


def pytest_collection_finish(session):
    path = os.environ["DUMP_COLLECTION"]
    with open(path, "w") as f:
        for item in session.items:
            skips = [m for m in item.iter_markers("skip")]
            reason = skips[0].kwargs.get("reason", "") if skips else ""
            f.write(f"{item.nodeid}\t{'skip' if skips else 'run'}\t{reason}\n")
