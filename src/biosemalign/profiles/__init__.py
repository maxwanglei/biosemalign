"""Task profiles: project behaviour as configuration rather than code (§5.8)."""

from biosemalign.profiles.loader import available_profiles, load_profile, load_profile_file
from biosemalign.profiles.migrate import (
    is_schema_1_profile,
    migrate_profile_data,
    migrate_profile_yaml,
)
from biosemalign.profiles.schema import TaskProfile

__all__ = [
    "TaskProfile",
    "available_profiles",
    "is_schema_1_profile",
    "load_profile",
    "load_profile_file",
    "migrate_profile_data",
    "migrate_profile_yaml",
]
