"""Knowledge-resource routing (§6.2).

Decides which resources a task may query, checks them against the registry
before constructing anything, and owns the providers' lifecycle. Querying only
task-relevant resources is not just an efficiency measure — it is how the
framework keeps outcome-associated knowledge out of signal-discovery tasks.
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType

from biosemalign.enums import ResourceMode
from biosemalign.exceptions import ConfigurationError, ResourceNotPermitted
from biosemalign.knowledge.providers.rxnorm import RxNavProvider
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.resources.registry import DeploymentEnvironment, get_registry
from biosemalign.schemas.provenance import ResourceProvenance
from biosemalign.settings import Settings, get_settings
from biosemalign.storage.cache import ResponseCache

__all__ = ["KnowledgeRouter"]

log = get_logger(__name__)

#: Provider implementations available in this release. Resources declared in
#: the registry but absent here are rejected with a clear message.
_PROVIDER_FACTORIES = {"rxnav"}


class KnowledgeRouter:
    """Resolves a profile's declared resources into live providers."""

    def __init__(
        self,
        profile: TaskProfile,
        *,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
        environment: DeploymentEnvironment | str | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings or get_settings()
        self.cache = cache or ResponseCache(
            self.settings.cache_dir,
            mode=self.settings.mode,
            retention_days=self.settings.cache_retention_days,
            dir_mode=self.settings.cache_dir_mode,
            file_mode=self.settings.cache_file_mode,
            store_raw_queries=self.settings.cache_store_raw_queries,
            allow_legacy_offline_entries=self.settings.cache_allow_legacy_offline_entries,
        )
        raw_environment = environment or self.settings.deployment_environment
        if raw_environment is None:
            if self.cache.mode in {ResourceMode.ONLINE, ResourceMode.CACHE_PREFER}:
                raise ConfigurationError(
                    f"{self.cache.mode.value} resource access requires an explicit "
                    "BIOSEMALIGN_DEPLOYMENT_ENVIRONMENT; external access refused"
                )
            self.environment: DeploymentEnvironment | None = None
        else:
            try:
                self.environment = DeploymentEnvironment(raw_environment)
            except ValueError as exc:
                raise ConfigurationError(
                    f"unknown deployment environment {raw_environment!r}; resource access refused"
                ) from exc
        self._providers: dict[str, RxNavProvider] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> KnowledgeRouter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()

    # -- resolution --------------------------------------------------------

    def primary_provider(self) -> RxNavProvider:
        """The provider for the profile's target terminology."""
        target = self.profile.task.target_terminology
        return self.provider_for(target)

    def provider_for(self, resource_name: str) -> RxNavProvider:
        """Construct (or reuse) the provider for a named resource."""
        registry = get_registry()
        if self.environment is None:
            # An unset environment conveys no egress authorization, but strict
            # cache replay still checks implementation and licensing gates.
            spec = registry.get(resource_name)
            if not spec.implemented:
                raise ResourceNotPermitted(
                    f"resource {spec.name!r} is declared in the registry but not implemented "
                    "in this version of BioSemAlign"
                )
            if spec.license_required and spec.credential_env_var:
                field = spec.credential_env_var.removeprefix("BIOSEMALIGN_").lower()
                if getattr(self.settings, field, None) is None:
                    raise ResourceNotPermitted(
                        f"resource {spec.name!r} requires a licence credential; set "
                        f"{spec.credential_env_var} in the environment. "
                        f"{spec.license_note or ''}".strip()
                    )
        else:
            spec = registry.require(resource_name, environment=self.environment)

        if spec.provider not in _PROVIDER_FACTORIES:
            raise ResourceNotPermitted(
                f"resource {resource_name!r} maps to provider {spec.provider!r}, which is "
                f"not implemented in this release. Implemented: {sorted(_PROVIDER_FACTORIES)}"
            )

        cached = self._providers.get(spec.provider)
        if cached is not None:
            return cached

        provider = RxNavProvider(
            # Settings carry the registry's public default but allow a
            # deployment to select an institutional mirror explicitly.
            base_url=self.settings.rxnav_base_url,
            cache=self.cache,
            timeout=self.settings.http_timeout_seconds,
            max_attempts=self.settings.http_max_attempts,
            requests_per_second=self.settings.rxnav_requests_per_second,
            resource_name=spec.name,
            expected_release=self.settings.rxnav_terminology_release,
            permitted_egress_classes=self.settings.permitted_egress_classes,
        )
        self._providers[spec.provider] = provider
        log.debug(
            "router.provider_ready",
            resource=spec.name,
            provider=spec.provider,
            mode=self.cache.mode.value,
        )
        return provider

    # -- provenance --------------------------------------------------------

    def resource_provenance(self) -> list[ResourceProvenance]:
        """Provenance for every resource this router actually touched."""
        entries: list[ResourceProvenance] = []
        for provider in self._providers.values():
            metadata = provider.last_access_metadata
            if not metadata and provider.cache_hits == 0 and provider.network_calls == 0:
                continue
            raw_accessed_at = metadata.get("accessed_at") or metadata.get("cache_stored_at")
            accessed_at: datetime | None = None
            if isinstance(raw_accessed_at, str):
                try:
                    accessed_at = datetime.fromisoformat(raw_accessed_at)
                except ValueError:
                    log.warning(
                        "router.invalid_provider_access_timestamp",
                        provider=provider.provider_name,
                    )

            payload = {
                "resource_name": provider.resource_name,
                "provider_name": provider.provider_name,
                "provider_version": provider.provider_version,
                "terminology_release": (
                    metadata.get("terminology_release") or provider.known_terminology_release
                ),
                "api_version": metadata.get("api_version") or provider.api_version,
                "mode": self.cache.mode,
                "cache_hits": provider.cache_hits,
                "network_calls": provider.network_calls,
            }
            if accessed_at is not None:
                payload["accessed_at"] = accessed_at
            entries.append(ResourceProvenance.model_validate(payload))
        return entries

    def reset_counters(self) -> None:
        """Reset provider accounting at a run boundary without dropping clients or caches."""
        for provider in self._providers.values():
            provider.reset_counters()

    @property
    def mode(self) -> ResourceMode:
        return self.cache.mode
