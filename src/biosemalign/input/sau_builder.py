"""Module 1: the SAU Builder.

Converts a request into one or more Semantic Alignment Units. It decides *what*
needs mapping and preserves the information needed to map it correctly; it
never assigns a terminology concept (§5.1).
"""

from __future__ import annotations

from biosemalign._version import __version__
from biosemalign.input.adapters.base import ADAPTER_VERSION, get_adapter
from biosemalign.input.adapters.structured import register_default_adapters
from biosemalign.input.context import build_attributes, build_envelope
from biosemalign.input.readiness import assess_readiness
from biosemalign.logging import get_logger
from biosemalign.profiles.schema import TaskProfile
from biosemalign.schemas.provenance import Provenance, RunContext
from biosemalign.schemas.request import AlignmentRequest
from biosemalign.schemas.sau import Mention, SemanticAlignmentUnit
from biosemalign.schemas.source import SourceRecord
from biosemalign.util import normalize_text, stable_id

__all__ = ["SAUBuilder"]

log = get_logger(__name__)

register_default_adapters()


class SAUBuilder:
    """Builds Semantic Alignment Units from alignment requests."""

    def __init__(self, profile: TaskProfile, *, identifier_key: str | bytes | None = None) -> None:
        self.profile = profile
        self.identifier_key = identifier_key

    def build(self, request: AlignmentRequest, run: RunContext) -> list[SemanticAlignmentUnit]:
        """Produce every SAU implied by one request."""
        source_type = request.source_type or self.profile.input.source_type
        adapter = get_adapter(source_type)

        run.source_adapter = adapter.adapter_name
        run.source_adapter_version = adapter.adapter_version

        records = adapter.to_source_records(request, self.profile)
        units = [self._build_one(record, request, run) for record in records]

        log.debug(
            "sau.built",
            count=len(units),
            source_type=source_type.value,
            adapter=adapter.adapter_name,
        )
        return units

    def _build_one(
        self, record: SourceRecord, request: AlignmentRequest, run: RunContext
    ) -> SemanticAlignmentUnit:
        envelope = build_envelope(record)
        attributes = build_attributes(record)

        mention = Mention(
            text=record.raw_value,
            normalized_text=normalize_text(record.raw_value),
            semantic_types=[self.profile.input.default_semantic_type],
            detection_method=(
                "user_specified"
                if not self.profile.input.automatic_mention_detection
                else "profile_default"
            ),
        )

        readiness = assess_readiness(
            mention_text=record.raw_value,
            envelope=envelope,
            attributes=attributes,
            profile=self.profile,
        )

        task_identity = {
            "operation": self.profile.task.operation,
            "mapping_intent": self.profile.task.mapping_intent,
            "target_terminology": self.profile.task.target_terminology,
            "desired_granularity": self.profile.task.desired_granularity,
        }
        contextual_key: dict[str, object] = {}
        for field in self.profile.input.mapping_key_context:
            value = getattr(attributes, field, None)
            if value is None:
                value = getattr(envelope, field, None)
            if value is not None:
                contextual_key[field] = value

        mapping_key = stable_id(
            "MAP",
            {
                "normalized_value": mention.normalized_text,
                "task": task_identity,
                "profile": self.profile.profile_name,
                "profile_version": self.profile.profile_version,
                "context": contextual_key,
            },
            length=32,
            key=self.identifier_key,
        )

        # Occurrence identity includes the complete semantic context and source
        # identity.  A 128-bit digest keeps collision risk negligible at study
        # scale while remaining deterministic for reproducible reruns.
        sau_id = stable_id(
            "SAU",
            {
                "value": record.raw_value,
                "source_system": record.source_system,
                "record_id": record.record_id,
                "source_type": record.source_type,
                "row": record.row,
                "source_extensions": record.extensions,
                "task": task_identity,
                "profile": self.profile.profile_name,
                "profile_version": self.profile.profile_version,
                "mention_mode": request.mention_mode,
                "context": envelope.model_dump(mode="json"),
                "attributes": attributes.model_dump(mode="json"),
            },
            length=32,
            key=self.identifier_key,
        )

        return SemanticAlignmentUnit(
            sau_id=sau_id,
            mapping_key=mapping_key,
            source=record,
            mention=mention,
            context=envelope,
            attributes=attributes,
            task=request.to_task(),
            provenance=self._stub_provenance(run),
            readiness=readiness,
        )

    def _stub_provenance(self, run: RunContext) -> Provenance:
        """Provenance as of SAU construction.

        The full record is assembled at decision time; what the SAU carries is
        the subset that is known before retrieval has happened.
        """
        return Provenance(
            run_id=run.run_id,
            package_version=__version__,
            profile_name=self.profile.profile_name,
            profile_version=self.profile.profile_version,
            source_adapter=run.source_adapter,
            source_adapter_version=run.source_adapter_version or ADAPTER_VERSION,
            identifier_hmac_key_fingerprint=run.identifier_hmac_key_fingerprint,
        )
