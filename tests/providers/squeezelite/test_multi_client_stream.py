"""Tests for the squeezelite multi-client stream task."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.providers.squeezelite.multi_client_stream import MultiClientStream

PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    sample_rate=44100,
    bit_depth=16,
    channels=2,
)


@pytest.fixture(name="instant_sleep")
def instant_sleep_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the runner's retry delays elapse immediately while still yielding control."""
    real_sleep = asyncio.sleep

    async def _sleep(_delay: float, *args: Any, **kwargs: Any) -> Any:
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _sleep)


@pytest.mark.usefixtures("instant_sleep")
async def test_runner_closes_the_source_when_no_client_connects() -> None:
    """
    Giving up on a stream nobody subscribed to tears the source down.

    The source counts as active playback for as long as it is open, so leaving it
    suspended here would hold audio analysis in its reduced-CPU mode indefinitely.
    """
    closed = asyncio.Event()

    async def _source() -> AsyncGenerator[bytes]:
        try:
            while True:
                yield b"chunk"
        finally:
            closed.set()

    stream = MultiClientStream(
        audio_source=_source(),
        audio_format=PCM_FORMAT,
        queue_id="queue-1",
        session_id="session-1",
    )
    await stream.task

    assert closed.is_set()


async def _next_chunk(subscription: AsyncGenerator[bytes]) -> bytes | None:
    """Return the next chunk of a subscription, or None once it has ended."""
    try:
        return await asyncio.wait_for(anext(subscription), 1)
    except StopAsyncIteration:
        return None


@pytest.mark.usefixtures("instant_sleep")
async def test_stop_ends_every_subscriber() -> None:
    """
    Stopping the stream ends the subscriptions instead of leaving them waiting.

    Each subscription feeds the encoder of one speaker, and that encoder only exits
    when its input ends, so a subscription left waiting leaks a process per speaker
    on every restart of the group.
    """

    async def _source() -> AsyncGenerator[bytes]:
        while True:
            yield b"chunk"

    stream = MultiClientStream(
        audio_source=_source(),
        audio_format=PCM_FORMAT,
        queue_id="queue-1",
        session_id="session-1",
        expected_clients=2,
    )
    subscriptions = [stream.subscribe_raw(), stream.subscribe_raw()]
    for subscription in subscriptions:
        assert await _next_chunk(subscription) == b"chunk"

    await stream.stop()

    for subscription in subscriptions:
        assert await _next_chunk(subscription) is None
    assert stream.subscribers == []


@pytest.mark.usefixtures("instant_sleep")
async def test_failing_source_ends_every_subscriber() -> None:
    """A source that raises ends the subscriptions as well."""

    async def _source() -> AsyncGenerator[bytes]:
        yield b"chunk"
        raise RuntimeError("source failed")

    stream = MultiClientStream(
        audio_source=_source(),
        audio_format=PCM_FORMAT,
        queue_id="queue-1",
        session_id="session-1",
    )
    subscription = stream.subscribe_raw()
    assert await _next_chunk(subscription) == b"chunk"

    with pytest.raises(RuntimeError):
        await stream.task

    assert await _next_chunk(subscription) is None
