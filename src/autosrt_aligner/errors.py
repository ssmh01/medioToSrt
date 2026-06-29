"""Project-specific exceptions with user-facing messages."""


class AutosrtError(Exception):
    """Base class for expected application errors."""


class InputError(AutosrtError):
    """Raised when user input is missing or unsupported."""


class DependencyError(AutosrtError):
    """Raised when a required local dependency is unavailable."""


class AlignmentError(AutosrtError):
    """Raised when forced alignment cannot produce reliable timestamps."""


class ExportValidationError(AutosrtError):
    """Raised when exported subtitle text fails continuity validation."""

