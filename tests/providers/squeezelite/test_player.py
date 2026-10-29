"""Tests for the Squeezelite player."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from aioslimproto.models import PlayerState as SlimPlayerState
from music_assistant_models.enums import MediaType, PlaybackState, RepeatMode
from music_assistant_models.player import PlayerMedia

from music_assistant.constants import (
    CONF_ENTRY_PREFER_WAV_FOR_LIVE_SOURCES_DEFAULT_ENABLED,
    CONF_ENTRY_SYNC_ADJUST,
    CONF_OUTPUT_CODEC,
    CONF_PREFER_WAV_FOR_LIVE_SOURCES,
    CONF_SYNC_ADJUST,
)
from music_assistant.providers.squeezelite.constants import (
    CONF_SYNC_JUMP_IGNORE_MS,
    CONF_SYNC_MAX_SKIP_AHEAD_MS,
    CONF_SYNC_MIN_ADJUST_MS,
    CONF_SYNC_PLAYPOINTS,
    DEVIATION_JUMP_IGNORE,
    MAX_SKIP_AHEAD_MS,
    MIN_DEVIATION_ADJUST,
    MIN_REQ_PLAYPOINTS,
    SyncPlayPoint,
)
from music_assistant.providers.squeezelite.player import SqueezelitePlayer, is_protocol_only_device
from tests.common import scheduled_call, use_real_create_task


async def test_squeezelite_prefers_wav_for_live_sources_by_default() -> None:
    """Squeezelite players default to the known-compatible low-latency WAV path."""

    async def _no_library_items(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        return
        yield  # pragma: no cover - makes this an async generator

    player = SqueezelitePlayer.__new__(SqueezelitePlayer)
    player.mass = mass = MagicMock()
    mass.music.playlists.iter_library_items = _no_library_items
    mass.music.radio.iter_library_items = _no_library_items

    entries = await player.get_config_entries()
    entry = next(entry for entry in entries if entry.key == CONF_PREFER_WAV_FOR_LIVE_SOURCES)

    assert (
        entry.default_value == CONF_ENTRY_PREFER_WAV_FOR_LIVE_SOURCES_DEFAULT_ENABLED.default_value
    )


def _member_player(*, prefer_wav_for_live_sources: bool) -> MagicMock:
    """Return a mocked member player with the given effective WAV preference."""
    member = MagicMock()
    member.config.get_value.side_effect = lambda key, default=None: (
        prefer_wav_for_live_sources if key == CONF_PREFER_WAV_FOR_LIVE_SOURCES else default
    )
    return member


def _player_with_mocked_mass() -> tuple[SqueezelitePlayer, MagicMock]:
    """Return an uninitialized SqueezelitePlayer alongside its mocked mass instance."""
    player = SqueezelitePlayer.__new__(SqueezelitePlayer)
    player.mass = mass = MagicMock()
    return player, mass


def test_group_audio_source_uses_wav_when_member_prefers_it() -> None:
    """A sync group member that opted into low-latency WAV gets WAV for AudioSource media."""
    player, mass = _player_with_mocked_mass()
    mass.players.get_player.return_value = _member_player(prefer_wav_for_live_sources=True)

    codec = player._get_member_output_codec(
        "member_1", PlayerMedia(uri="fake://x", media_type=MediaType.AUDIO_SOURCE)
    )

    assert codec == "wav"
    mass.config.get_raw_player_config_value.assert_not_called()


def test_group_audio_source_falls_back_to_output_codec_when_disabled() -> None:
    """A sync group member that disabled the WAV preference uses its configured output codec."""
    player, mass = _player_with_mocked_mass()
    mass.players.get_player.return_value = _member_player(prefer_wav_for_live_sources=False)
    mass.config.get_raw_player_config_value.return_value = "aac"

    codec = player._get_member_output_codec(
        "member_1", PlayerMedia(uri="fake://x", media_type=MediaType.AUDIO_SOURCE)
    )

    assert codec == "aac"
    mass.config.get_raw_player_config_value.assert_called_once_with(
        "member_1", CONF_OUTPUT_CODEC, "flac"
    )


def test_group_regular_track_ignores_wav_preference() -> None:
    """Regular track playback in a sync group always uses the member's configured codec."""
    player, mass = _player_with_mocked_mass()
    mass.config.get_raw_player_config_value.return_value = "flac"

    codec = player._get_member_output_codec(
        "member_1", PlayerMedia(uri="fake://x", media_type=MediaType.TRACK)
    )

    assert codec == "flac"
    mass.players.get_player.assert_not_called()


@pytest.mark.parametrize(
    ("media_type", "expected"),
    [
        (MediaType.RADIO, (64, 1)),
        (MediaType.AUDIO_SOURCE, (64, 1)),
        (MediaType.TRACK, (200, 20)),
    ],
)
async def test_live_streams_start_on_a_smaller_buffer(
    media_type: MediaType, expected: tuple[int, int]
) -> None:
    """Radio and AudioSource streams start on a smaller buffer, also when repeated."""
    player, mass = _player_with_mocked_mass()
    player._extra_data = {}
    mass.player_queues.get.return_value = MagicMock(
        repeat_mode=RepeatMode.ONE, shuffle_enabled=False
    )
    slimplayer = MagicMock()
    slimplayer.play_url = AsyncMock()

    await player._handle_play_url_for_slimplayer(
        slimplayer,
        url="http://127.0.0.1:8097/stream.flac",
        media=PlayerMedia(uri="fake://x", media_type=media_type, source_id="queue_1"),
    )
    # the repeat-one re-enqueue is scheduled, not awaited here
    mass.call_later.call_args.args[1].close()

    assert slimplayer.play_url.call_count == 2
    for play_call in slimplayer.play_url.call_args_list:
        thresholds = (play_call.kwargs["stream_threshold"], play_call.kwargs["output_threshold"])
        assert thresholds == expected


@pytest.mark.parametrize(
    ("device_model", "expected"),
    [
        # generic squeezelite (software) players are full players
        ("SqueezeLite", False),
        ("SqueezeLite-HA-Addon", False),
        ("SqueezePlay", False),
        ("SqueezeESP32", False),
        # hardware players are full players
        ("Squeezebox Boom", False),
        ("Transporter", False),
        # unknown/absent model info defaults to a full player
        ("", False),
        ("Unknown", False),
        # WiiM/LinkPlay devices use squeezelite as a secondary protocol
        ("WiiM Player", True),
        ("wiim mini", True),
        # LMS bridge tools represent devices that are already players themselves
        ("RaopBridge", True),
        ("CastBridge", True),
        ("UPnPBridge", True),
    ],
)
def test_is_protocol_only_device(device_model: str, expected: bool) -> None:
    """Test protocol-only device detection based on the reported device model."""
    assert is_protocol_only_device(device_model) is expected


@pytest.mark.parametrize(
    ("url", "mime_type", "expected"),
    [
        # sync group member urls carry the codec in the query string, not the path
        (
            "http://127.0.0.1:8097/slimproto/multi?player_id=x&fmt=flac&child_player_id=y",
            "audio/flac",
            "audio/flac",
        ),
        (
            "http://127.0.0.1:8097/slimproto/multi?player_id=x&fmt=mp3&child_player_id=y",
            "audio/mpeg",
            "audio/mpeg",
        ),
        # without an explicit mime type it is derived from the url extension
        ("http://127.0.0.1:8097/stream.flac", None, "audio/flac"),
    ],
)
async def test_play_url_mime_type_is_explicit_for_sync_members(
    url: str, mime_type: str | None, expected: str
) -> None:
    """Sync group member urls pass the member codec mime type instead of deriving it."""
    player, mass = _player_with_mocked_mass()
    player._extra_data = {}
    mass.player_queues.get.return_value = None
    slimplayer = MagicMock()
    slimplayer.play_url = AsyncMock()

    await player._handle_play_url_for_slimplayer(
        slimplayer,
        url=url,
        media=PlayerMedia(uri="fake://x", media_type=MediaType.TRACK, source_id="queue_1"),
        mime_type=mime_type,
    )

    assert slimplayer.play_url.call_args.kwargs["mime_type"] == expected


async def test_grouped_play_media_passes_per_member_mime_type() -> None:
    """Grouped play_media gives each sync member its own codec's mime type, not a derived one."""
    player, mass = _player_with_mocked_mass()
    player._extra_data = {}
    player._player_id = "leader"
    player._attr_group_members = ["leader", "member_flac", "member_mp3"]
    player.multi_client_stream = None
    use_real_create_task(mass)

    mass.player_queues.get_item.return_value = None
    mass.player_queues.get.return_value = None
    mass.streams.audio.select_flow_pcm_format = AsyncMock()
    mass.streams.base_url = "http://127.0.0.1:8097"

    member_codecs = {"member_flac": "flac", "member_mp3": "mp3"}
    clients = []
    for member_id in ("member_flac", "member_mp3"):
        client = MagicMock()
        client.player_id = member_id
        client.play_url = AsyncMock()
        client.pause = AsyncMock()
        clients.append(client)

    media = PlayerMedia(
        uri="fake://x",
        media_type=MediaType.TRACK,
        source_id="queue_1",
        queue_item_id="item_1",
    )

    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch("music_assistant.providers.squeezelite.player.MultiClientStream"),
        patch.object(player, "_get_sync_clients", return_value=clients),
        patch.object(
            player,
            "_get_member_output_codec",
            side_effect=lambda member_player_id, _media: member_codecs[member_player_id],
        ),
    ):
        await player.play_media(media)

    flac_client, mp3_client = clients
    assert flac_client.play_url.call_args.kwargs["mime_type"] == "audio/flac"
    assert mp3_client.play_url.call_args.kwargs["mime_type"] == "audio/mpeg"


async def test_regroup_supersedes_pending_resume() -> None:
    """A member removed and re-added while the leader is busy restarts the leader once."""
    player, mass = _player_with_mocked_mass()
    player._player_id = "leader"
    player._attr_group_members = ["leader", "member"]
    player._attr_playback_state = PlaybackState.PLAYING
    use_real_create_task(mass)
    member = MagicMock()
    member.player_id = "member"
    member.state = SlimPlayerState.STOPPED
    mass.players.get_player.return_value = member
    # the playback lock an announcement holds on the leader while it plays
    leader_busy = asyncio.Event()
    resumed: list[str] = []

    async def _cmd_resume(player_id: str) -> None:
        await leader_busy.wait()
        resumed.append(player_id)

    mass.players.cmd_resume = _cmd_resume

    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch.object(SqueezelitePlayer, "state", new_callable=PropertyMock),
        patch.object(player, "_get_sync_clients", return_value=[member]),
        patch.object(player, "update_state"),
    ):
        await player.set_members(player_ids_to_remove=["member"])
        await player.set_members(player_ids_to_add=["member"])
        leader_busy.set()
        await asyncio.gather(*mass._tracked_tasks.values(), return_exceptions=True)

    assert resumed == ["leader"]


async def _no_items(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
    return
    yield  # pragma: no cover - makes this an async generator


def _synced_member_player(
    *,
    min_adjust_ms: int = MIN_DEVIATION_ADJUST,
    playpoints: int = MIN_REQ_PLAYPOINTS,
    jump_ignore_ms: int = DEVIATION_JUMP_IGNORE,
    max_skip_ahead_ms: int = MAX_SKIP_AHEAD_MS,
) -> tuple[SqueezelitePlayer, MagicMock, MagicMock]:
    """Return a playing sync member with the given tuning, its leader client and the mocked mass."""
    player, mass = _player_with_mocked_mass()
    player.logger = MagicMock()
    player._config = MagicMock()
    player._cache = {}  # backs the cached display_name property
    player.client = MagicMock(player_id="member", state=SlimPlayerState.PLAYING, elapsed_seconds=10)
    player.client.name = "Member"
    leader = MagicMock(player_id="leader", state=SlimPlayerState.PLAYING, elapsed_seconds=10)
    leader.name = "Leader"
    player._provider = MagicMock()
    player._provider.slimproto.get_player.return_value = leader
    player._sync_playpoints = deque(maxlen=playpoints)
    player._do_not_resync_before = 0.0
    player._sync_min_adjust_ms = min_adjust_ms
    player._sync_jump_ignore_ms = jump_ignore_ms
    player._sync_max_skip_ahead_ms = max_skip_ahead_ms
    return player, leader, mass


def _feed_heartbeats(player: SqueezelitePlayer, leader: MagicMock, diffs: list[int]) -> None:
    """Run the sync check once per diff, the member reporting that many ms behind the leader."""
    pending = list(diffs)

    def _corrected(client: MagicMock) -> int:
        if client is leader:
            return 10_000 + pending[0]
        return 10_000

    provider = cast("MagicMock", player._provider)
    provider.get_corrected_elapsed_milliseconds.side_effect = _corrected
    with patch.object(
        SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value="leader"
    ):
        while pending:
            player._handle_sync()
            pending.pop(0)


async def test_sync_adjust_range_is_widened_for_squeezelite_only() -> None:
    """Squeezelite players accept a much wider sync delay than the shared entry allows."""
    player, mass = _player_with_mocked_mass()
    mass.music.playlists.iter_library_items = _no_items
    mass.music.radio.iter_library_items = _no_items

    entries = await player.get_config_entries()
    entry = next(entry for entry in entries if entry.key == CONF_SYNC_ADJUST)

    assert entry.range == (-5000, 5000)
    # the shared entry, used by other providers, is untouched
    assert CONF_ENTRY_SYNC_ADJUST.range == (-500, 500)


@pytest.mark.parametrize(
    ("key", "default"),
    [
        (CONF_SYNC_MIN_ADJUST_MS, MIN_DEVIATION_ADJUST),
        (CONF_SYNC_PLAYPOINTS, MIN_REQ_PLAYPOINTS),
        (CONF_SYNC_JUMP_IGNORE_MS, DEVIATION_JUMP_IGNORE),
        (CONF_SYNC_MAX_SKIP_AHEAD_MS, MAX_SKIP_AHEAD_MS),
    ],
)
async def test_sync_tuning_entries_default_to_the_former_constants(key: str, default: int) -> None:
    """The per-player tuning entries default to the values that used to be hardcoded."""
    player, mass = _player_with_mocked_mass()
    mass.music.playlists.iter_library_items = _no_items
    mass.music.radio.iter_library_items = _no_items

    entries = await player.get_config_entries()
    entry = next(entry for entry in entries if entry.key == key)

    assert entry.default_value == default
    assert entry.advanced is True
    assert entry.requires_reload is False
    assert entry.category == "protocol_generic"


async def test_on_config_updated_resizes_the_playpoint_window() -> None:
    """A changed measurement count rebuilds the playpoint window and keeps the newest points."""
    player, _mass = _player_with_mocked_mass()
    values = {CONF_SYNC_PLAYPOINTS: 4, CONF_SYNC_MIN_ADJUST_MS: 30}
    player._config = MagicMock()
    player._config.get_value.side_effect = lambda key, default=None: values.get(key, default)
    player._sync_playpoints = deque(
        (SyncPlayPoint(float(index), "leader", index) for index in range(6)), maxlen=8
    )
    with (
        patch.object(player, "_set_preset_items", AsyncMock()),
        patch.object(player, "_set_display", AsyncMock()),
    ):
        await player.on_config_updated()

    assert player._sync_playpoints.maxlen == 4
    assert [point.diff for point in player._sync_playpoints] == [2, 3, 4, 5]
    assert player._sync_min_adjust_ms == 30
    assert player._sync_jump_ignore_ms == DEVIATION_JUMP_IGNORE
    assert player._sync_max_skip_ahead_ms == MAX_SKIP_AHEAD_MS


def test_handle_sync_zero_max_skip_ahead_never_pauses_the_leader() -> None:
    """With the maximum skip-ahead disabled a far-behind member catches up by itself."""
    player, leader, mass = _synced_member_player(max_skip_ahead_ms=0)

    _feed_heartbeats(player, leader, [2000] * MIN_REQ_PLAYPOINTS)

    player.client.skip_over.assert_called_once_with(2000)
    mass.create_task.assert_called_once()


def test_handle_sync_pauses_the_leader_above_max_skip_ahead() -> None:
    """A member lagging more than the maximum skip-ahead makes the leader wait for it."""
    player, leader, mass = _synced_member_player(max_skip_ahead_ms=800)

    _feed_heartbeats(player, leader, [2000] * MIN_REQ_PLAYPOINTS)

    player.client.skip_over.assert_not_called()
    name, arguments = scheduled_call(mass.create_task.call_args.args[0])
    assert name == "pause_and_unpause"
    assert arguments["slim_client"] is leader
    assert arguments["pause_duration_ms"] == 2000


@pytest.mark.parametrize(("min_adjust_ms", "corrected"), [(30, False), (8, True)])
def test_handle_sync_respects_the_configured_minimum(min_adjust_ms: int, corrected: bool) -> None:
    """Drift below the configured minimum is left alone, drift above it is corrected."""
    player, leader, _mass = _synced_member_player(min_adjust_ms=min_adjust_ms)

    _feed_heartbeats(player, leader, [20] * MIN_REQ_PLAYPOINTS)

    assert player.client.skip_over.called is corrected
    if corrected:
        player.client.skip_over.assert_called_once_with(20)


@pytest.mark.parametrize(("jump_ignore_ms", "corrected"), [(500, False), (2000, True)])
def test_handle_sync_ignores_jumps_above_the_configured_window(
    jump_ignore_ms: int, corrected: bool
) -> None:
    """A single measurement far from the running average is ignored within the configured window."""
    player, leader, _mass = _synced_member_player(jump_ignore_ms=jump_ignore_ms)

    _feed_heartbeats(player, leader, [20] * (MIN_REQ_PLAYPOINTS - 1) + [900])

    assert player.client.skip_over.called is corrected


def test_handle_sync_waits_for_the_configured_playpoint_count() -> None:
    """No correction is sent before the configured number of measurements is collected."""
    player, leader, _mass = _synced_member_player(playpoints=4)

    _feed_heartbeats(player, leader, [100] * 3)
    player.client.skip_over.assert_not_called()

    _feed_heartbeats(player, leader, [100])
    player.client.skip_over.assert_called_once_with(100)
