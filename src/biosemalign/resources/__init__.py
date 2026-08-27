"""Knowledge Resource Registry: what each external resource is and who may use it."""

from biosemalign.resources.registry import (
    DeploymentEnvironment,
    ResourceRegistry,
    ResourceSpec,
    get_registry,
)

__all__ = ["DeploymentEnvironment", "ResourceRegistry", "ResourceSpec", "get_registry"]
