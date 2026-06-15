"""
Health checks module for SpeechShield.

Provides:
  • Component health status tracking
  • Readiness and liveness probes
  • Dependency checks

Usage:
    from core.health import health_checks
    
    health_checks.mark_ready("ASRAgent")
    health_checks.mark_error("Ollama", "Connection refused")
    
    status = health_checks.get_status()
    print(status)  # {'ready': True, 'components': {...}}
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import time


class HealthStatus(str, Enum):
    """Component health status."""

    UNKNOWN = "unknown"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: Optional[str] = None
    last_error: Optional[str] = None
    last_check_time: float = field(default_factory=time.time)
    error_count: int = 0

    def mark_ready(self, message: str = "") -> None:
        """Mark component as ready."""
        self.status = HealthStatus.READY
        self.message = message
        self.last_check_time = time.time()

    def mark_error(self, error: str) -> None:
        """Mark component as errored."""
        self.status = HealthStatus.ERROR
        self.last_error = error
        self.error_count += 1
        self.last_check_time = time.time()

    def mark_degraded(self, message: str = "") -> None:
        """Mark component as degraded but still functional."""
        self.status = HealthStatus.DEGRADED
        self.message = message
        self.last_check_time = time.time()

    def mark_starting(self) -> None:
        """Mark component as starting up."""
        self.status = HealthStatus.STARTING
        self.last_check_time = time.time()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "last_check_time": self.last_check_time,
        }


class HealthChecks:
    """Centralized health check management."""

    def __init__(self) -> None:
        """Initialize health checks."""
        self.components: Dict[str, ComponentHealth] = {}
        self._startup_time = time.time()

        # Register core components
        self._register_core_components()

    def _register_core_components(self) -> None:
        """Register standard components."""
        core_components = [
            "EventBus",
            "AudioAgent",
            "ASRAgent",
            "ToxicityAgent",
            "RewriteAgent",
            "VoiceAgent",
            "OrchestratorAgent",
        ]
        for comp in core_components:
            self.components[comp] = ComponentHealth(name=comp)

    def register(self, name: str) -> None:
        """Register a new component."""
        if name not in self.components:
            self.components[name] = ComponentHealth(name=name)

    def mark_ready(self, name: str, message: str = "") -> None:
        """Mark a component as ready."""
        if name not in self.components:
            self.register(name)
        self.components[name].mark_ready(message)

    def mark_error(self, name: str, error: str) -> None:
        """Mark a component as errored."""
        if name not in self.components:
            self.register(name)
        self.components[name].mark_error(error)

    def mark_degraded(self, name: str, message: str = "") -> None:
        """Mark a component as degraded."""
        if name not in self.components:
            self.register(name)
        self.components[name].mark_degraded(message)

    def mark_starting(self, name: str) -> None:
        """Mark a component as starting."""
        if name not in self.components:
            self.register(name)
        self.components[name].mark_starting()

    def get_component(self, name: str) -> Optional[ComponentHealth]:
        """Get component health status."""
        return self.components.get(name)

    @property
    def is_ready(self) -> bool:
        """
        Check if system is ready.

        Ready = all components are either READY or DEGRADED.
        """
        if not self.components:
            return False

        for component in self.components.values():
            if component.status not in (HealthStatus.READY, HealthStatus.DEGRADED):
                return False

        return True

    @property
    def is_alive(self) -> bool:
        """
        Check if system is alive.

        Alive = no components are in ERROR state.
        """
        for component in self.components.values():
            if component.status == HealthStatus.ERROR:
                return False

        return True

    def get_status(self) -> dict:
        """
        Get overall system health status.

        Returns:
            Dictionary with ready, alive, and component statuses
        """
        statuses = {}
        for name, component in self.components.items():
            statuses[name] = component.to_dict()

        return {
            "ready": self.is_ready,
            "alive": self.is_alive,
            "uptime_seconds": time.time() - self._startup_time,
            "components": statuses,
        }

    def reset(self) -> None:
        """Reset all health checks."""
        self.components.clear()
        self._register_core_components()
        self._startup_time = time.time()


# Global health checks instance
_health_checks_instance: Optional[HealthChecks] = None


def health_checks() -> HealthChecks:
    """Get global health checks instance (lazy-loaded)."""
    global _health_checks_instance
    if _health_checks_instance is None:
        _health_checks_instance = HealthChecks()
    return _health_checks_instance


def reset_health_checks() -> None:
    """Reset health checks (for testing)."""
    global _health_checks_instance
    _health_checks_instance = None
