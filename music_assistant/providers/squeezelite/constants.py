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

# sync constants: the defaults of the per-player sync tuning entries below
MIN_DEVIATION_ADJUST = 8  # milliseconds
MIN_REQ_PLAYPOINTS = 8  # we need at least 8 measurements
DEVIATION_JUMP_IGNORE = 500  # ignore a sudden unrealistic jump
MAX_SKIP_AHEAD_MS = 800  # 0.8 seconds

CONF_SYNC_MIN_ADJUST_MS = "sync_min_adjust_ms"
CONF_SYNC_PLAYPOINTS = "sync_playpoints"
CONF_SYNC_JUMP_IGNORE_MS = "sync_jump_ignore_ms"
CONF_SYNC_MAX_SKIP_AHEAD_MS = "sync_max_skip_ahead_ms"

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
# squeezelite (software) players accept a far larger correction than the generic entry allows,
# e.g. a converted speaker whose amplifier adds seconds of processing delay
CONF_ENTRY_SYNC_ADJUST_SQUEEZELITE = replace(CONF_ENTRY_SYNC_ADJUST, range=(-5000, 5000))
CONF_ENTRY_SYNC_MIN_ADJUST_MS = ConfigEntry(
    key=CONF_SYNC_MIN_ADJUST_MS,
    type=ConfigEntryType.INTEGER,
    range=(1, 500),
    default_value=MIN_DEVIATION_ADJUST,
    category="protocol_generic",
    advanced=True,
)
CONF_ENTRY_SYNC_PLAYPOINTS = ConfigEntry(
    key=CONF_SYNC_PLAYPOINTS,
    type=ConfigEntryType.INTEGER,
    range=(2, 60),
    default_value=MIN_REQ_PLAYPOINTS,
    category="protocol_generic",
    advanced=True,
)
CONF_ENTRY_SYNC_JUMP_IGNORE_MS = ConfigEntry(
    key=CONF_SYNC_JUMP_IGNORE_MS,
    type=ConfigEntryType.INTEGER,
    range=(50, 10000),
    default_value=DEVIATION_JUMP_IGNORE,
    category="protocol_generic",
    advanced=True,
)
CONF_ENTRY_SYNC_MAX_SKIP_AHEAD_MS = ConfigEntry(
    key=CONF_SYNC_MAX_SKIP_AHEAD_MS,
    type=ConfigEntryType.INTEGER,
    range=(0, 10000),
    default_value=MAX_SKIP_AHEAD_MS,
    category="protocol_generic",
    advanced=True,
)


@dataclass
class SyncPlayPoint:
    """Simple structure to describe a Sync Playpoint."""

    timestamp: float
    sync_master: str
    diff: int
