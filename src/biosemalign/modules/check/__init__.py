"""Check module: core policy-pack checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, model_validator

from biosemalign.modules.base import OpContext
from biosemalign.modules.registry import register
from biosemalign.schemas.base import BioSemAlignModel
from biosemalign.validation.policy_dsl import PolicyCheck, load_policy_pack

__all__ = ["PolicyCheck"]


class _PolicyOptions(BioSemAlignModel):
    path: str | None = Field(default=None, description="YAML policy pack file.")
    pack: dict[str, Any] | None = Field(default=None, description="An inline policy pack.")

    @model_validator(mode="after")
    def _exactly_one(self) -> _PolicyOptions:
        if (self.path is None) == (self.pack is None):
            raise ValueError("give exactly one of 'path' or 'pack'")
        return self


def _policy_factory(context: OpContext, options: Mapping[str, Any]) -> PolicyCheck:
    parsed = _PolicyOptions.model_validate(options)
    source: str | dict[str, Any] = parsed.path if parsed.path is not None else parsed.pack or {}
    return PolicyCheck(
        load_policy_pack(source),
        target_terminology=context.profile.target.terminology,
        profile=context.profile,
    )


register("check", "policy", _policy_factory)
