#!/usr/bin/env python3
"""Check that each documented module's ``__all__`` matches the API docs.

Every ``desc.<module>.<name>`` entry in ``docs/api*.rst`` must appear in that
module's ``__all__``, every name in ``__all__`` must be documented, and every
name in ``__all__`` must be defined or imported in the module. Modules are
parsed with ``ast`` rather than imported, so the check is fast and does not
need DESC's dependencies.
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY_RE = re.compile(r"^\s+(desc\.[\w.]+)\s*$", re.M)


def documented_names():
    """Return a mapping from module name to the set of names documented in it."""
    documented = defaultdict(set)
    for rst in sorted((ROOT / "docs").glob("api*.rst")):
        for entry in ENTRY_RE.findall(rst.read_text()):
            modname, name = entry.rsplit(".", 1)
            documented[modname].add(name)
    return documented


def module_path(modname):
    """Return the source file for a dotted module name, or None if not found."""
    base = ROOT.joinpath(*modname.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def top_level_names(tree):
    """Return names bound at module level, including inside top-level if/try."""
    names = set()
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            # descend into blocks that execute at import time, but not into
            # function or class bodies
            for field in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(getattr(node, field, []))
        elif isinstance(node, ast.ExceptHandler):
            stack.extend(node.body)
    return names


def declared_all(tree):
    """Return the literal value of the module's top-level ``__all__``, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return None


def main():
    """Run the check and return a process exit code."""
    errors = []
    for modname, names in sorted(documented_names().items()):
        path = module_path(modname)
        if path is None:
            errors.append(f"{modname}: documented module not found")
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        public = declared_all(tree)
        rel = path.relative_to(ROOT)
        if public is None:
            errors.append(f"{rel}: no __all__, but docs list {sorted(names)}")
            continue
        if names - public:
            errors.append(
                f"{rel}: documented but not in __all__: {sorted(names - public)}"
            )
        if public - names:
            errors.append(
                f"{rel}: in __all__ but not documented: {sorted(public - names)}"
            )
        undefined = public - top_level_names(tree)
        if undefined:
            errors.append(f"{rel}: in __all__ but not defined: {sorted(undefined)}")

    for err in errors:
        print(err)
    if errors:
        print(
            "\nThe public API is defined by the docs/api*.rst files. Keep each "
            "module's __all__ in sync with the names documented for it."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
