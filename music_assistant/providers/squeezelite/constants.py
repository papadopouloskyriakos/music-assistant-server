"""Constants for the Squeezelite player provider."""

from __future__ import annotations

from dataclasses import dataclass, replace

from aioslimproto.client import PlayerState as SlimPlayerState
from aioslimproto.models import VisualisationType as SlimVisualisationType
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, PlaybackState, RepeatMode

from music_assistant.constants import CONF_ENTRY_SYNC_ADJUST

CONF_CLI_TELNET_PORT = "cli_telnet_port"
CONF_CLI_JSON_PORT = "cli_json_port"
CONF_DISCOVERY = "discovery"
CONF_PORT = "port"
DEFAULT_SLIMPROTO_PORT = 3483
CONF_DISPLAY = "display"
CONF_VISUALIZATION = "visualization"

DEFAULT_PLAYER_VOLUME = 20
DEFAULT_VISUALIZATION = SlimVisualisationType.NONE

# sync: a group starts on one shared instant and the leader keeps its members on the
# player that is furthest ahead (the Lyrion Music Server algorithm)
CONF_SYNC_MAINTAIN = "sync_maintain"
CONF_SYNC_MIN_ADJUST_MS = "sync_min_adjust_ms"
CONF_SYNC_PACKET_LATENCY_MS = "sync_packet_latency_ms"
CONF_SYNC_START_DELAY_MS = "sync_start_delay_ms"
DEFAULT_SYNC_MIN_ADJUST_MS = 8  # milliseconds
DEFAULT_SYNC_PACKET_LATENCY_MS = 2  # milliseconds
DEFAULT_SYNC_START_DELAY_MS = 200  # milliseconds of headroom for the start command
SYNC_CHECK_INTERVAL_S = 0.95
SYNC_PLAYPOINT_MAX_AGE_S = 6.0  # the 5 s heartbeat is the slowest status cadence
SYNC_MIN_CORRECTION_S = 0.010
SYNC_MAX_CORRECTION_S = 10.0
SYNC_SKIP_COOLDOWN_S = 1.0
SYNC_START_HOLD_OFF_S = 2.0
SYNC_START_MAX_LEAD_S = 9.0  # squeezelite starts at once when told to start > 10 s ahead
BUFFER_READY_TIMEOUT_S = 8.0
SYNC_CATEGORY = "squeezelite_sync"

STATE_MAP = {
    SlimPlayerState.BUFFERING: PlaybackState.PLAYING,
    SlimPlayerState.BUFFER_READY: PlaybackState.PLAYING,
    SlimPlayerState.PAUSED: PlaybackState.PAUSED,
    SlimPlayerState.PLAYING: PlaybackState.PLAYING,
    SlimPlayerState.STOPPED: PlaybackState.IDLE,
}

REPEATMODE_MAP = {RepeatMode.OFF: 0, RepeatMode.ONE: 1, RepeatMode.ALL: 2}

CONF_ENTRY_DISPLAY = ConfigEntry(
    key=CONF_DISPLAY,
    type=ConfigEntryType.BOOLEAN,
    default_value=False,
    required=False,
    advanced=True,
)
CONF_ENTRY_VISUALIZATION = ConfigEntry(
    key=CONF_VISUALIZATION,
    type=ConfigEntryType.STRING,
    default_value=DEFAULT_VISUALIZATION,
    options=[
        ConfigValueOption(x.value, title=x.name.replace("_", " ").title())
        for x in SlimVisualisationType
    ],
    required=False,
    advanced=True,
    depends_on=CONF_DISPLAY,
)
# squeezelite (software) players accept a far larger correction than the generic entry
# allows, e.g. a converted speaker whose amplifier adds seconds of processing delay; the
# value applies to the next scheduled start and to the running check without a restart
CONF_ENTRY_SYNC_ADJUST_SQUEEZELITE = replace(
    CONF_ENTRY_SYNC_ADJUST,
    range=(-5000, 5000),
    category=SYNC_CATEGORY,
    requires_reload=False,
    immediate_apply=True,
)
CONF_ENTRY_SYNC_MAINTAIN = ConfigEntry(
    key=CONF_SYNC_MAINTAIN,
    type=ConfigEntryType.BOOLEAN,
    default_value=True,
    category=SYNC_CATEGORY,
    advanced=True,
)
CONF_ENTRY_SYNC_MIN_ADJUST_MS = ConfigEntry(
    key=CONF_SYNC_MIN_ADJUST_MS,
    type=ConfigEntryType.INTEGER,
    range=(1, 500),
    default_value=DEFAULT_SYNC_MIN_ADJUST_MS,
    category=SYNC_CATEGORY,
    advanced=True,
)
CONF_ENTRY_SYNC_PACKET_LATENCY_MS = ConfigEntry(
    key=CONF_SYNC_PACKET_LATENCY_MS,
    type=ConfigEntryType.INTEGER,
    range=(0, 500),
    default_value=DEFAULT_SYNC_PACKET_LATENCY_MS,
    category=SYNC_CATEGORY,
    advanced=True,
)
CONF_ENTRY_SYNC_START_DELAY_MS = ConfigEntry(
    key=CONF_SYNC_START_DELAY_MS,
    type=ConfigEntryType.INTEGER,
    range=(0, 5000),
    default_value=DEFAULT_SYNC_START_DELAY_MS,
    category=SYNC_CATEGORY,
    advanced=True,
)


@dataclass(frozen=True)
class SyncSettings:
    """Per-player sync tuning as read from the player config."""

    play_delay_ms: int  # sync_adjust: positive plays earlier (Lyrion's playDelay)
    maintain: bool
    min_adjust_ms: int
