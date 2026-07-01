"""Architecture guardrails for algorithm package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src/rl_from_scratch"
ALGORITHM_PACKAGES = {
    path.name
    for path in PACKAGE_ROOT.iterdir()
    if path.is_dir() and path.name not in {"core", "__pycache__"}
}
LEGACY_ROOT_PLUMBING = {
    "artifacts",
    "base",
    "config",
    "env",
    "metrics",
    "normalization",
    "reporting",
    "rollout",
    "utils",
}


def _module_imports(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, []))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.module, [alias.name for alias in node.names]))
    return imports


def _top_level_symbol_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        )
    }


def test_algorithm_packages_do_not_ship_rollout_modules() -> None:
    offending = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for package in sorted(ALGORITHM_PACKAGES)
        for path in (PACKAGE_ROOT / package).glob("rollout.py")
    )

    assert offending == [], f"Legacy rollout modules must stay deleted: {offending}"


def test_algorithm_packages_do_not_import_rollout_modules() -> None:
    offending: dict[str, list[str]] = {}

    for package in sorted(ALGORITHM_PACKAGES):
        package_dir = PACKAGE_ROOT / package
        for path in sorted(package_dir.glob("*.py")):
            rollout_imports = sorted(
                {
                    module_name
                    for module_name, _ in _module_imports(path)
                    if module_name.startswith("rl_from_scratch.")
                    and module_name.endswith(".rollout")
                }
            )
            if rollout_imports:
                offending[path.relative_to(PROJECT_ROOT).as_posix()] = rollout_imports

    assert offending == {}, f"Legacy rollout imports found: {offending}"


def test_algorithm_packages_do_not_define_runner_symbols() -> None:
    offending: dict[str, list[str]] = {}

    for package in sorted(ALGORITHM_PACKAGES):
        package_dir = PACKAGE_ROOT / package
        for path in sorted(package_dir.glob("*.py")):
            runner_symbols = sorted(
                name for name in _top_level_symbol_names(path) if name.endswith("Runner")
            )
            if runner_symbols:
                offending[path.relative_to(PROJECT_ROOT).as_posix()] = runner_symbols

    assert offending == {}, (
        "Legacy Runner-named classes/functions should stay out of algorithm packages: "
        f"{offending}"
    )


def test_algorithm_packages_do_not_import_sibling_algorithms_or_legacy_root_plumbing() -> None:
    offending: dict[str, list[str]] = {}

    for package in sorted(ALGORITHM_PACKAGES):
        package_dir = PACKAGE_ROOT / package
        for path in sorted(package_dir.glob("*.py")):
            bad_imports: list[str] = []
            for module_name, imported_names in _module_imports(path):
                if module_name.startswith("rl_from_scratch.core"):
                    continue
                if module_name.startswith(f"rl_from_scratch.{package}"):
                    continue

                sibling_hits = [
                    sibling
                    for sibling in ALGORITHM_PACKAGES - {package}
                    if module_name.startswith(f"rl_from_scratch.{sibling}")
                ]
                if sibling_hits:
                    bad_imports.append(module_name)
                    continue

                if module_name == "rl_from_scratch":
                    legacy_names = sorted(
                        name for name in imported_names if name in LEGACY_ROOT_PLUMBING
                    )
                    bad_imports.extend(f"{module_name}.{name}" for name in legacy_names)
                    continue

                for legacy_name in LEGACY_ROOT_PLUMBING:
                    if module_name == f"rl_from_scratch.{legacy_name}":
                        bad_imports.append(module_name)
                        break

            if bad_imports:
                rel_path = path.relative_to(PROJECT_ROOT).as_posix()
                offending[rel_path] = sorted(set(bad_imports))

    assert offending == {}, f"Architecture boundary violations found: {offending}"
