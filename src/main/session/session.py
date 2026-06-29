"""
Session state management (Challenges #8 and #9).

Participant ID and condition are set before recording starts.
Once locked, condition cannot change without explicit confirmation.
Pre-calibration flag tracks whether calibration has occurred.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

VALID_CONDITIONS = frozenset({"IR", "Webcam", "WebcamFiltered", "NoGaze", "Demo"})
_FORMULA_CHARS = frozenset("=+-@")


def _sanitize(value: str) -> str:
    if value and value[0] in _FORMULA_CHARS:
        return "'" + value
    return value


@dataclass
class Session:
    participant_id: str
    condition: str
    session_id: str = field(default_factory=lambda: str(int(time.time())))
    _locked: bool = field(default=False, init=False, repr=False)
    _calibrated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.participant_id = _sanitize(self.participant_id)
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                f"condition must be one of {sorted(VALID_CONDITIONS)}, got {self.condition!r}"
            )

    def lock(self) -> None:
        """Lock condition to prevent accidental changes during recording."""
        self._locked = True

    def unlock(self) -> None:
        """Unlock to allow condition change (requires explicit call — Challenge #8)."""
        self._locked = False

    def mark_calibrated(self) -> None:
        """Mark that successful calibration has occurred (clears pre-calibration flag)."""
        self._calibrated = True

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_pre_calibration(self) -> bool:
        return not self._calibrated

    def ir_required(self) -> bool:
        """IR hardware is present in all conditions except NoGaze."""
        return self.condition != "NoGaze"
