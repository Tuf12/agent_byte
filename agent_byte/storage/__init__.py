"""Storage backend implementations."""

from .base import StorageBase
from .json_numpy_storage import JsonNumpyStorage

# Optional imports with graceful fallback
try:
    from .vector_db_storage import VectorDBStorage
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VectorDBStorage = None
    VECTOR_DB_AVAILABLE = False

try:
    from .experience_buffer import ExperienceBuffer, StreamingExperienceBuffer
    EXPERIENCE_BUFFER_AVAILABLE = True
except ImportError:
    ExperienceBuffer = None
    StreamingExperienceBuffer = None
    EXPERIENCE_BUFFER_AVAILABLE = False

try:
    from .recovery_utils import StorageRecoveryManager
    RECOVERY_UTILS_AVAILABLE = True
except ImportError:
    StorageRecoveryManager = None
    RECOVERY_UTILS_AVAILABLE = False

# Build __all__ dynamically based on what's available
__all__ = [
    "StorageBase",
    "JsonNumpyStorage",
]

if VectorDBStorage is not None:
    __all__.append("VectorDBStorage")

if ExperienceBuffer is not None:
    __all__.extend(["ExperienceBuffer", "StreamingExperienceBuffer"])

if StorageRecoveryManager is not None:
    __all__.append("StorageRecoveryManager")