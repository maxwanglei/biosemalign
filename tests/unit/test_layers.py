"""Architecture test enforcing the Section 5.9 layer import rules.

Modules are classified by their Section 5.9 / Appendix B target layers so that
the violation lists can only shrink as work packages move code into ``modules/``
and ``domains/drug/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "biosemalign"

LAYERS: dict[str, int] = {
    "contracts": 0,
    "foundation": 1,
    "input": 2,
    "knowledge": 3,
    "modules": 4,
    "kernel": 5,
    "workflows": 6,
    "domains": 7,
    "surfaces": 8,
}

EXACT_LAYERS: dict[str, str] = {
    "biosemalign": "surfaces",
}

PREFIX_LAYERS: list[tuple[str, str]] = sorted(
    [
        ("biosemalign.enums", "contracts"),
        ("biosemalign.schemas", "contracts"),
        ("biosemalign._version", "foundation"),
        ("biosemalign.exceptions", "foundation"),
        ("biosemalign.logging", "foundation"),
        ("biosemalign.plugins", "foundation"),
        ("biosemalign.profiles", "foundation"),
        ("biosemalign.settings", "foundation"),
        ("biosemalign.storage.crosswalk_memo", "kernel"),
        ("biosemalign.storage", "foundation"),
        ("biosemalign.util", "foundation"),
        ("biosemalign.input", "input"),
        ("biosemalign.knowledge", "knowledge"),
        ("biosemalign.resources", "knowledge"),
        ("biosemalign.targets", "knowledge"),
        ("biosemalign.modules", "modules"),
        ("biosemalign.kernel", "kernel"),
        ("biosemalign.routing", "kernel"),
        ("biosemalign.validation", "kernel"),
        ("biosemalign.workflows", "workflows"),
        ("biosemalign.domains", "domains"),
        ("biosemalign.api", "surfaces"),
        ("biosemalign.cli", "surfaces"),
        ("biosemalign.evaluation", "surfaces"),
        ("biosemalign.run_manifest", "surfaces"),
        ("biosemalign.testing", "surfaces"),
        ("biosemalign.verification", "surfaces"),
    ],
    key=lambda entry: -len(entry[0]),
)

# Section 5.9 intentional exception: self-verifying artifacts lazily import
# target specs and linking rules.
INTENTIONAL_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("biosemalign.schemas.artifact", "biosemalign.modules.link.targets"),
        ("biosemalign.schemas.artifact", "biosemalign.targets"),
        ("biosemalign.schemas.knowledge", "biosemalign.targets"),
        ("biosemalign.schemas.knowledge", "biosemalign.targets.spec"),
    }
)

# Current baseline upward violations under Section 5.9 / Appendix B destinations;
# this set may only shrink across work packages (21 in WP0 -> 14 in WP1).
KNOWN_VIOLATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # Pre-existing non-WP0 upward edges (tracked in Section 11 Q8):
        ("biosemalign.input.adapters.existing", "biosemalign.api"),
        ("biosemalign.modules.check", "biosemalign.validation.policy_dsl"),
        ("biosemalign.profiles.schema", "biosemalign.targets"),
        ("biosemalign.profiles.schema", "biosemalign.targets.spec"),
        ("biosemalign.routing.conformal", "biosemalign.evaluation.metrics"),
        ("biosemalign.schemas.artifact", "biosemalign.routing.features"),
        ("biosemalign.schemas.artifact", "biosemalign.util"),
        ("biosemalign.schemas.provenance", "biosemalign.util"),
    }
)

# Non-domain modules never import domain packs statically.
KNOWN_DOMAIN_IMPORTS: frozenset[tuple[str, str]] = frozenset()


def _discover_modules() -> dict[str, tuple[Path, bool]]:
    modules: dict[str, tuple[Path, bool]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT)
        parts = list(rel.with_suffix("").parts)
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        mod = "biosemalign" + ("." + ".".join(parts) if parts else "")
        modules[mod] = (path, is_pkg)
    return modules


def _layer_of(mod: str) -> str:
    if mod in EXACT_LAYERS:
        return EXACT_LAYERS[mod]
    for prefix, layer in PREFIX_LAYERS:
        if mod == prefix or mod.startswith(prefix + "."):
            return layer
    raise AssertionError(f"module {mod!r} is not assigned to any Section 5.9 layer")


def _extract_imports(mod: str, path: Path, is_pkg: bool, all_mods: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "biosemalign" or alias.name.startswith("biosemalign."):
                    imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level > 0:
                pkg_parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
                up = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(up + ([base] if base else []))
            if base == "biosemalign" or base.startswith("biosemalign."):
                for alias in node.names:
                    if alias.name == "*":
                        imported.add(base)
                        continue
                    candidate = f"{base}.{alias.name}"
                    if candidate in all_mods:
                        imported.add(candidate)
                    else:
                        imported.add(base)
    return imported


def _upward_edges() -> set[tuple[str, str]]:
    modules = _discover_modules()
    all_mods = set(modules)
    upward: set[tuple[str, str]] = set()
    for mod, (path, is_pkg) in modules.items():
        src_layer = _layer_of(mod)
        for target in _extract_imports(mod, path, is_pkg, all_mods):
            tgt_layer = _layer_of(target)
            if LAYERS[tgt_layer] > LAYERS[src_layer]:
                upward.add((mod, target))
    return upward


def _domain_import_edges() -> set[tuple[str, str]]:
    modules = _discover_modules()
    all_mods = set(modules)
    domain_edges: set[tuple[str, str]] = set()
    for mod, (path, is_pkg) in modules.items():
        if _layer_of(mod) == "domains":
            continue
        for target in _extract_imports(mod, path, is_pkg, all_mods):
            if _layer_of(target) == "domains":
                domain_edges.add((mod, target))
    return domain_edges


def test_every_module_belongs_to_a_layer() -> None:
    modules = _discover_modules()
    assert modules
    for mod in modules:
        assert _layer_of(mod) in LAYERS


def test_no_new_core_or_surface_imports_of_domains() -> None:
    unexpected = sorted(_domain_import_edges() - KNOWN_DOMAIN_IMPORTS)
    assert unexpected == [], f"new non-domain imports of domain modules introduced: {unexpected}"


def test_known_domain_imports_shrink_only() -> None:
    resolved = sorted(KNOWN_DOMAIN_IMPORTS - _domain_import_edges())
    assert resolved == [], (
        "these known domain imports have been fixed — remove them from KNOWN_DOMAIN_IMPORTS: "
        f"{resolved}"
    )


def test_layer_dependencies_have_no_new_violations() -> None:
    upward = _upward_edges()
    unexpected = sorted(upward - INTENTIONAL_EXCEPTIONS - KNOWN_VIOLATIONS)
    assert unexpected == [], f"new upward layer dependencies introduced: {unexpected}"


def test_known_layer_violations_shrink_only() -> None:
    upward = _upward_edges()
    resolved = sorted(KNOWN_VIOLATIONS - upward)
    assert resolved == [], (
        "these known layer violations have been fixed — remove them from KNOWN_VIOLATIONS: "
        f"{resolved}"
    )
