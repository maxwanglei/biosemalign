"""Knowledge Resource Registry (§20.2).

Consulted before any provider is constructed, so that a profile asking for a
resource this environment may not use fails at startup with an explanation
rather than at request time with a 401 or an ``AttributeError``.

The registry ships inside the package; ``./configs/resources.yaml`` overrides it
when present.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from biosemalign.exceptions import ConfigurationError, ResourceNotPermitted
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.settings import get_settings

__all__ = ["DeploymentEnvironment", "ResourceRegistry", "ResourceSpec", "get_registry"]

_PACKAGED_REGISTRY = ("biosemalign.resources", "registry.yaml")


class DeploymentEnvironment(StrEnum):
    """Where the package is running (§20.3, §18.8)."""

    LOCAL = "local"
    INSTITUTIONAL_SERVER = "institutional_server"
    HPC = "hpc"
    RESTRICTED_ENCLAVE = "restricted_enclave"


class ResourceSpec(BioSemAlignModel):
    """One external resource and the terms on which it may be used."""

    name: str
    provider: str
    version: str | None = None
    access_method: str
    base_url: str | None = None
    license_required: bool = False
    license_note: str | None = None
    redistribution_allowed: bool = False
    credential_env_var: str | None = None
    operations: list[str] = Field(default_factory=list)
    permitted_environments: list[DeploymentEnvironment] = Field(default_factory=list)
    cache_policy: str = "aggressive"
    rate_limit_per_second: float | None = None
    update_policy: str | None = None
    implemented: bool = True

    def check_usable(self, *, environment: DeploymentEnvironment) -> None:
        """Raise if this resource may not be used as configured.

        Three separate gates, because they fail for different reasons and a
        caller deserves to know which: not built yet, not permitted here, or
        missing credentials.
        """
        if not self.implemented:
            raise ResourceNotPermitted(
                f"resource {self.name!r} is declared in the registry but not implemented "
                "in this version of BioSemAlign"
            )
        if self.permitted_environments and environment not in self.permitted_environments:
            raise ResourceNotPermitted(
                f"resource {self.name!r} is not permitted in the {environment.value!r} "
                f"environment; permitted: {[e.value for e in self.permitted_environments]}"
            )
        if self.license_required and self.credential_env_var:
            settings = get_settings()
            field = self.credential_env_var.removeprefix("BIOSEMALIGN_").lower()
            secret = getattr(settings, field, None)
            if secret is None:
                raise ResourceNotPermitted(
                    f"resource {self.name!r} requires a licence credential; set "
                    f"{self.credential_env_var} in the environment. {self.license_note or ''}".strip()
                )


class ResourceRegistry(BioSemAlignModel):
    """The full set of declared resources."""

    resources: list[ResourceSpec] = Field(default_factory=list)

    def get(self, name: str) -> ResourceSpec:
        """Look up a resource by name, case-insensitively."""
        folded = name.casefold()
        for spec in self.resources:
            if spec.name.casefold() == folded:
                return spec
        raise ResourceNotPermitted(
            f"resource {name!r} is not in the registry. Declared resources: "
            f"{[r.name for r in self.resources]}"
        )

    def require(
        self,
        name: str,
        *,
        environment: DeploymentEnvironment = DeploymentEnvironment.LOCAL,
    ) -> ResourceSpec:
        """Look up a resource and assert it is usable here."""
        spec = self.get(name)
        spec.check_usable(environment=environment)
        return spec


@lru_cache(maxsize=1)
def get_registry() -> ResourceRegistry:
    """Load the registry, preferring a project override over the packaged copy."""
    override = Path.cwd() / "configs" / "resources.yaml"
    if override.is_file():
        text = override.read_text(encoding="utf-8")
        origin = str(override)
    else:
        package, filename = _PACKAGED_REGISTRY
        text = importlib_resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        origin = f"{package}/{filename}"

    try:
        raw = yaml.safe_load(text)
        return ResourceRegistry.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigurationError(f"{origin} is not a valid resource registry:\n{exc}") from exc
