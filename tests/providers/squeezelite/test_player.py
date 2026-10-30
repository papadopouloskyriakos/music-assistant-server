"""Tests for the Squeezelite player."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from aioslimproto.errors import PlayerClockUnknown
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
    CONF_SYNC_MAINTAIN,
    CONF_SYNC_MIN_ADJUST_MS,
    CONF_SYNC_PACKET_LATENCY_MS,
    DEFAULT_SYNC_MIN_ADJUST_MS,
    DEFAULT_SYNC_PACKET_LATENCY_MS,
    SYNC_CATEGORY,
)
from music_assistant.providers.squeezelite.player import SqueezelitePlayer, is_protocol_only_device
from tests.common import use_real_create_task


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
    player._sync_hold_off_until = {}
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
    player._sync_hold_off_until = {}
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


NOW = 1_700_000_000.0


async def _no_items(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
    return
    yield  # pragma: no cover - makes this an async generator


def _group_leader(
    members: dict[str, dict[str, object]],
) -> tuple[SqueezelitePlayer, dict[str, MagicMock], MagicMock]:
    """
    Return a sync leader "leader" with mocked slimproto clients for it and its members.

    Each member spec may carry config values (sync_adjust, sync_maintain,
    sync_min_adjust_ms) and client facts (state, apparent_start, measured_at, playpoint).
    """
    player, mass = _player_with_mocked_mass()
    player.logger = MagicMock()
    player._config = MagicMock()
    player._cache = {}  # backs the cached display_name property
    player._attr_name = "leader"
    player._player_id = "leader"
    player._attr_group_members = ["leader", *[pid for pid in members if pid != "leader"]]
    player._sync_hold_off_until = {}
    player._last_sync_check = 0.0
    player.multi_client_stream = None
    player._provider = MagicMock(sync_start_delay_ms=200)
    clients: dict[str, MagicMock] = {}
    config: dict[tuple[str, str], object] = {}
    for player_id, spec in members.items():
        client = MagicMock(player_id=player_id, state=spec.get("state", SlimPlayerState.PLAYING))
        client.name = player_id
        client.playpoint = (
            MagicMock(
                apparent_start_time=spec.get("apparent_start", NOW - 10),
                measured_at=spec.get("measured_at", NOW - 0.5),
            )
            if spec.get("playpoint", True)
            else None
        )
        client.start_at = AsyncMock()
        client.unpause_at = AsyncMock()
        client.stop = AsyncMock()
        client.skip_over = MagicMock()
        clients[player_id] = client
        for key in (CONF_SYNC_ADJUST, CONF_SYNC_MAINTAIN, CONF_SYNC_MIN_ADJUST_MS):
            if key in spec:
                config[(player_id, key)] = spec[key]
    player.client = clients["leader"]
    player._provider.slimproto.get_player.side_effect = lambda pid: clients[pid]
    mass.config.get_raw_player_config_value.side_effect = lambda pid, key, default=None: config.get(
        (pid, key), default
    )
    return player, clients, mass


def _frozen_clock(*values: float) -> MagicMock:
    """Return a time module stand-in whose time() yields the values, then repeats the last."""
    sequence = list(values)

    def _time() -> float:
        if len(sequence) > 1:
            return sequence.pop(0)
        return sequence[0]

    return MagicMock(time=_time)


async def test_sync_adjust_range_is_widened_for_squeezelite_only() -> None:
    """Squeezelite players accept a much wider sync delay than the shared entry allows."""
    player, mass = _player_with_mocked_mass()
    mass.music.playlists.iter_library_items = _no_items
    mass.music.radio.iter_library_items = _no_items

    entries = await player.get_config_entries()
    entry = next(entry for entry in entries if entry.key == CONF_SYNC_ADJUST)

    assert entry.range == (-5000, 5000)
    assert entry.category == SYNC_CATEGORY
    assert entry.requires_reload is False
    assert entry.immediate_apply is True
    # the shared entry, used by other providers, is untouched
    assert CONF_ENTRY_SYNC_ADJUST.range == (-500, 500)
    assert CONF_ENTRY_SYNC_ADJUST.requires_reload is True


@pytest.mark.parametrize(
    ("key", "default"),
    [
        (CONF_SYNC_MAINTAIN, True),
        (CONF_SYNC_MIN_ADJUST_MS, DEFAULT_SYNC_MIN_ADJUST_MS),
        (CONF_SYNC_PACKET_LATENCY_MS, DEFAULT_SYNC_PACKET_LATENCY_MS),
    ],
)
async def test_sync_entries_apply_live(key: str, default: object) -> None:
    """The per-player sync entries have Lyrion's defaults and apply without a restart."""
    player, mass = _player_with_mocked_mass()
    mass.music.playlists.iter_library_items = _no_items
    mass.music.radio.iter_library_items = _no_items

    entries = await player.get_config_entries()
    entry = next(entry for entry in entries if entry.key == key)

    assert entry.default_value == default
    assert entry.advanced is True
    assert entry.requires_reload is False
    assert entry.category == SYNC_CATEGORY


async def test_on_config_updated_pushes_packet_latency_to_the_client() -> None:
    """The configured status packet latency is handed to the slimproto client."""
    player, _mass = _player_with_mocked_mass()
    player.client = MagicMock()
    player._config = MagicMock()
    player._config.get_value.side_effect = lambda key, default=None: (
        7 if key == CONF_SYNC_PACKET_LATENCY_MS else default
    )
    with (
        patch.object(player, "_set_preset_items", AsyncMock()),
        patch.object(player, "_set_display", AsyncMock()),
    ):
        await player.on_config_updated()

    assert player.client.packet_latency_ms == 7


async def test_group_start_is_scheduled_on_one_shared_instant_with_per_player_delay() -> None:
    """Every member starts at the same instant, shifted earlier by its own delay."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"state": SlimPlayerState.BUFFER_READY},
            "m1": {"state": SlimPlayerState.BUFFER_READY, CONF_SYNC_ADJUST: 2000},
            "m2": {"state": SlimPlayerState.BUFFER_READY, CONF_SYNC_ADJUST: -300},
        }
    )
    use_real_create_task(_mass)
    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch("music_assistant.providers.squeezelite.player.time", _frozen_clock(NOW)),
        patch("music_assistant.providers.squeezelite.player.asyncio.sleep", AsyncMock()),
    ):
        await player._handle_buffer_ready()

    # lead = largest delay (2.0 s) + 200 ms headroom
    assert clients["leader"].start_at.await_args.args[0] == pytest.approx(NOW + 2.2)
    assert clients["m1"].start_at.await_args.args[0] == pytest.approx(NOW + 0.2)
    assert clients["m2"].start_at.await_args.args[0] == pytest.approx(NOW + 2.5)
    assert all(hold == pytest.approx(NOW + 4.2) for hold in player._sync_hold_off_until.values())


async def test_group_start_still_arms_a_member_that_never_buffered() -> None:
    """A member stuck buffering does not keep the others silent: all are started anyway."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"state": SlimPlayerState.BUFFER_READY},
            "m1": {"state": SlimPlayerState.BUFFERING},
        }
    )
    # the deadline is computed at NOW; the next reading is already past it
    use_real_create_task(_mass)
    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch("music_assistant.providers.squeezelite.player.time", _frozen_clock(NOW, NOW + 9)),
        patch("music_assistant.providers.squeezelite.player.asyncio.sleep", AsyncMock()),
    ):
        await player._handle_buffer_ready()

    clients["leader"].start_at.assert_awaited_once()
    clients["m1"].start_at.assert_awaited_once()


async def test_group_start_never_schedules_beyond_the_squeezelite_limit() -> None:
    """No start is scheduled further ahead than squeezelite honours."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"state": SlimPlayerState.BUFFER_READY},
            "m1": {"state": SlimPlayerState.BUFFER_READY, CONF_SYNC_ADJUST: 5000},
            "m2": {"state": SlimPlayerState.BUFFER_READY, CONF_SYNC_ADJUST: -5000},
        }
    )
    use_real_create_task(_mass)
    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch("music_assistant.providers.squeezelite.player.time", _frozen_clock(NOW)),
        patch("music_assistant.providers.squeezelite.player.asyncio.sleep", AsyncMock()),
    ):
        await player._handle_buffer_ready()

    starts = [client.start_at.await_args.args[0] for client in clients.values()]
    assert max(starts) <= NOW + 9.0
    assert clients["m1"].start_at.await_args.args[0] == pytest.approx(NOW + 0.2)


async def test_group_start_falls_back_to_an_immediate_start_without_a_clock() -> None:
    """A member whose clock is unknown is started at once instead of being left silent."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"state": SlimPlayerState.BUFFER_READY},
            "m1": {"state": SlimPlayerState.BUFFER_READY},
        }
    )
    clients["m1"].start_at.side_effect = PlayerClockUnknown
    clients["m1"].jiffies = 4321
    use_real_create_task(_mass)
    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch("music_assistant.providers.squeezelite.player.time", _frozen_clock(NOW)),
        patch("music_assistant.providers.squeezelite.player.asyncio.sleep", AsyncMock()),
    ):
        await player._handle_buffer_ready()

    clients["m1"].unpause_at.assert_awaited_once_with(4321)
    clients["leader"].unpause_at.assert_not_awaited()


async def test_solo_buffer_ready_is_unchanged() -> None:
    """A player outside any group starts as soon as its buffer is ready."""
    player, clients, _mass = _group_leader({"leader": {"state": SlimPlayerState.BUFFER_READY}})
    player._attr_group_members = []
    clients["leader"].jiffies = 1234
    with patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None):
        await player._handle_buffer_ready()

    clients["leader"].unpause_at.assert_awaited_once_with(1234)
    clients["leader"].start_at.assert_not_awaited()


def _run_sync_check(player: SqueezelitePlayer, now: float = NOW) -> None:
    """Run the leader's sync check at the given server time."""
    with patch("music_assistant.providers.squeezelite.player.time", _frozen_clock(now)):
        player._check_group_sync()


def test_group_sync_skips_members_behind_the_most_ahead_player() -> None:
    """Members behind the player furthest ahead skip forward; that player is left alone."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"apparent_start": 100.0},
            "m1": {"apparent_start": 100.5},
            "m2": {"apparent_start": 99.9},
        }
    )

    _run_sync_check(player)

    clients["leader"].skip_over.assert_called_once_with(100)
    clients["m1"].skip_over.assert_called_once_with(600)
    clients["m2"].skip_over.assert_not_called()


def test_group_sync_applies_the_leaders_own_play_delay() -> None:
    """The leader is just another member: its own delay correction applies to it too."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"apparent_start": 100.0, CONF_SYNC_ADJUST: 300},
            "m1": {"apparent_start": 100.0},
        }
    )

    _run_sync_check(player)

    clients["leader"].skip_over.assert_called_once_with(300)
    clients["m1"].skip_over.assert_not_called()


def test_group_sync_respects_min_adjust_and_maintain() -> None:
    """Drift under a member's minimum is left alone; an unmaintained member is ignored."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"apparent_start": 100.0},
            "m1": {"apparent_start": 100.02, CONF_SYNC_MIN_ADJUST_MS: 30},
            # furthest ahead, but not maintained: neither reference nor corrected
            "m2": {"apparent_start": 99.0, CONF_SYNC_MAINTAIN: False},
        }
    )

    _run_sync_check(player)

    for client in clients.values():
        client.skip_over.assert_not_called()


@pytest.mark.parametrize("spec", [{"playpoint": False}, {"measured_at": NOW - 7}])
def test_group_sync_aborts_on_a_stale_or_missing_playpoint(spec: dict[str, object]) -> None:
    """Without a fresh play point from every member no correction is sent at all."""
    player, clients, _mass = _group_leader(
        {
            "leader": {"apparent_start": 100.0},
            "m1": {"apparent_start": 100.0, **spec},
            "m2": {"apparent_start": 101.0},
        }
    )

    _run_sync_check(player)

    for client in clients.values():
        client.skip_over.assert_not_called()


def test_group_sync_ignores_deltas_over_ten_seconds() -> None:
    """A member that far out is not drifting; a restart lines it up again."""
    player, clients, _mass = _group_leader(
        {"leader": {"apparent_start": 100.0}, "m1": {"apparent_start": 111.0}}
    )

    _run_sync_check(player)

    clients["m1"].skip_over.assert_not_called()


def test_group_sync_honours_cooldown_and_rate_limit() -> None:
    """Checks run at most about once a second and a corrected member gets a cooldown."""
    player, clients, _mass = _group_leader(
        {"leader": {"apparent_start": 100.0}, "m1": {"apparent_start": 100.6}}
    )

    _run_sync_check(player, NOW)
    assert clients["m1"].skip_over.call_count == 1
    _run_sync_check(player, NOW + 0.5)  # rate limited
    assert clients["m1"].skip_over.call_count == 1
    _run_sync_check(player, NOW + 1.5)  # cooldown passed, still behind
    assert clients["m1"].skip_over.call_count == 2


def test_group_sync_runs_from_the_leaders_heartbeat_only() -> None:
    """A member's heartbeat never runs the group check; the leader's does."""
    player, _clients, _mass = _group_leader({"leader": {}, "m1": {}})
    player._attr_playback_state = PlaybackState.PLAYING
    player._attr_elapsed_time_last_updated = 0.0
    player.client.elapsed_seconds = 5.0
    with (
        patch.object(
            SqueezelitePlayer,
            "state",
            new_callable=PropertyMock,
            return_value=MagicMock(corrected_elapsed_time=5.0),
        ),
        patch.object(player, "_check_group_sync") as check,
    ):
        player._handle_player_heartbeat()
        check.assert_called_once()
        player._attr_group_members = []
        player._handle_player_heartbeat()
        check.assert_called_once()


def test_member_underrun_clears_its_hold_off() -> None:
    """A member that ran dry is corrected on the next check instead of waiting out a cooldown."""
    leader, _clients, mass = _group_leader({"leader": {}, "m1": {}})
    leader._sync_hold_off_until = {"m1": NOW + 5, "leader": NOW + 5}
    member, member_mass = _player_with_mocked_mass()
    member._player_id = "m1"
    member.logger = MagicMock()
    member._attr_group_members = []
    member_mass.players.get_player.return_value = leader
    with patch.object(
        SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value="leader"
    ):
        member._handle_output_underrun()

    assert "m1" not in leader._sync_hold_off_until
    assert leader._sync_hold_off_until["leader"] == NOW + 5
    assert mass.create_task.call_count == 0


async def test_removing_a_member_does_not_restart_the_group() -> None:
    """A member leaving a playing group is stopped; the others play on uninterrupted."""
    player, clients, mass = _group_leader({"leader": {}, "m1": {}, "m2": {}})
    player._attr_playback_state = PlaybackState.PLAYING
    player._sync_hold_off_until = {"m1": NOW + 5}
    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch.object(
            SqueezelitePlayer,
            "state",
            new_callable=PropertyMock,
            return_value=MagicMock(current_media=MagicMock()),
        ),
        patch.object(player, "update_state"),
    ):
        await player.set_members(player_ids_to_remove=["m1"])

    clients["m1"].stop.assert_awaited_once()
    assert player._attr_group_members == ["leader", "m2"]
    assert "m1" not in player._sync_hold_off_until
    mass.create_task.assert_not_called()


async def test_adding_a_member_restarts_the_group() -> None:
    """A member joining a playing group still restarts it (no late join into a stream)."""
    player, _clients, mass = _group_leader({"leader": {}, "m1": {}})
    player._attr_playback_state = PlaybackState.PLAYING
    joiner = MagicMock(state=SlimPlayerState.STOPPED)
    mass.players.get_player.return_value = joiner
    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch.object(
            SqueezelitePlayer,
            "state",
            new_callable=PropertyMock,
            return_value=MagicMock(current_media=MagicMock()),
        ),
        patch.object(player, "update_state"),
    ):
        await player.set_members(player_ids_to_add=["m3"])

    assert player._attr_group_members == ["leader", "m1", "m3"]
    mass.players.cmd_resume.assert_called_once_with("leader")
    mass.create_task.assert_called_once()


async def test_setup_refreshes_the_other_players_so_a_group_can_take_the_newcomer() -> None:
    """A player that connects is groupable at once: every other player's state is recomputed."""
    player, mass = _player_with_mocked_mass()
    player.client = MagicMock(player_id="new", name="new", stop=AsyncMock(), mute=AsyncMock())
    player.client.volume_set = AsyncMock()
    player.logger = MagicMock()
    mass.cache.get = AsyncMock(return_value=None)
    mass.players.register_or_update = AsyncMock()
    others = [MagicMock(player_id="a"), MagicMock(player_id="b")]
    itself = MagicMock(player_id="new")
    mass.players.iter_players.return_value = [*others, itself]
    with (
        patch.object(SqueezelitePlayer, "player_id", new_callable=PropertyMock, return_value="new"),
        patch.object(
            SqueezelitePlayer,
            "provider",
            new_callable=PropertyMock,
            return_value=MagicMock(instance_id="squeezelite"),
        ),
        patch.object(player, "update_attributes"),
    ):
        await player.setup()

    mass.players.register_or_update.assert_awaited_once_with(player)
    for other in others:
        other.update_state.assert_called_once_with()
    itself.update_state.assert_not_called()
