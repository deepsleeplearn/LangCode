from pathlib import Path

from langcode_agent.storage.session_store import SessionStore


def test_upsert_partial_updates_only_in_progress_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("session-1", str(tmp_path))
    store.save_messages("session-1", str(tmp_path), [{"role": "user", "content": "hello"}])

    store.upsert_partial("session-1", 1, "assistant", "first")
    store.upsert_partial("session-1", 1, "assistant", "first second")

    stored = store.load_session("session-1")
    assert stored is not None
    assert stored["messages"] == [
        {"role": "user", "content": "hello", "tool_call_id": None},
        {"role": "assistant", "content": "first second", "tool_call_id": None},
    ]


def test_session_revision_and_title_follow_writes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("session-1", str(tmp_path), title="Original")
    initial = store.load_revision("session-1")

    store.save_messages("session-1", str(tmp_path), [{"role": "user", "content": "hello"}])
    saved = store.load_revision("session-1")
    store.rename_session("session-1", "Renamed")

    assert initial == 0
    assert saved == 1
    assert store.load_revision("session-1") == 2
    assert store.load_title("session-1") == "Renamed"


def test_session_revision_changes_when_workspace_changes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("session-1", "/ws/a")

    store.ensure_session("session-1", "/ws/b")

    assert store.load_revision("session-1") == 1
    assert store.load_session("session-1")["workspace"] == "/ws/b"


def test_load_revision_distinguishes_deleted_from_missing(tmp_path: Path) -> None:
    """Item 4: a soft-deleted session is not the same as a session that never existed."""
    from langcode_agent.storage.session_store import DELETED_REVISION

    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("session-1", str(tmp_path))

    assert store.load_revision("never-created") is None
    assert store.load_revision("session-1") == 0

    store.delete_session("session-1")

    assert store.load_revision("session-1") == DELETED_REVISION


def test_save_messages_does_not_resurrect_a_deleted_session(tmp_path: Path) -> None:
    """Item 4: create -> delete -> late save must leave the session deleted."""
    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("doomed", str(tmp_path))
    store.save_messages("doomed", str(tmp_path), [{"role": "user", "content": "hello"}])

    store.delete_session("doomed")
    store.save_messages("doomed", str(tmp_path), [{"role": "user", "content": "late write"}])

    assert store.load_session("doomed") is None
    assert store.load_title("doomed") is None
    assert [item["id"] for item in store.list_sessions()] == []
    assert store.load_revision("doomed") == -1

    # ...unless undeleting is the explicit intent
    store.save_messages("doomed", str(tmp_path), [{"role": "user", "content": "back"}], restore=True)
    restored = store.load_session("doomed")
    assert restored is not None
    assert restored["messages"][0]["content"] == "back"


def test_upsert_partial_skips_a_deleted_session(tmp_path: Path) -> None:
    """Item 8: an in-flight partial save must not write into a deleted session."""
    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("gone", str(tmp_path))
    store.save_messages("gone", str(tmp_path), [{"role": "user", "content": "hello"}])
    store.delete_session("gone")
    revision_after_delete = store.load_revision("gone")

    store.upsert_partial("gone", 1, "assistant", "orphan text")

    assert store.load_session("gone") is None
    assert store.load_revision("gone") == revision_after_delete
    with store._connect() as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", ("gone",)).fetchone()
    assert rows["n"] == 0


def test_save_agent_dialogue_bumps_the_session_revision(tmp_path: Path) -> None:
    """Item 7: it was the one write path that left the revision untouched."""
    store = SessionStore(tmp_path / "web.sqlite")
    store.ensure_session("threaded", str(tmp_path))
    before = store.load_revision("threaded")

    store.save_agent_dialogue(
        "threaded",
        "thread-1",
        kind="debate",
        title="设计评审",
        participants=[{"agent_id": "a", "agent_name": "A"}],
        messages=[{"agent_id": "a", "agent_name": "A", "role": "assistant", "content": "观点"}],
    )

    assert store.load_revision("threaded") == before + 1
    assert store.load_agent_thread("thread-1")["messages"][0]["content"] == "观点"
