#!/usr/bin/env python3
"""Check that the public API declared by ``__all__`` matches the API docs.

The following must hold:

- Every ``desc.<module>.<name>`` entry in ``docs/api*.rst`` appears in that
  module's ``__all__``, and every name in a documented module's ``__all__`` is
  documented and defined or imported in the module. Submodules listed in a
  package's ``__all__`` are exempt.
- Every entry on a topical ``docs/api_*.rst`` page is also in the alphabetical
  listing in ``docs/api.rst``.
- Every module that defines ``__all__`` has each of those names documented,
  either under the module itself or under a parent package that re-exports it.

Modules are parsed with ``ast`` rather than imported, so the check is fast and
does not need DESC's dependencies.
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY_RE = re.compile(r"^\s+(desc\.[\w.]+)\s*$", re.M)


ALPHABETICAL_PAGE = "api.rst"


def documented_entries():
    """Return a mapping from API docs file name to the set of entries in it."""
    return {
        rst.name: set(ENTRY_RE.findall(rst.read_text()))
        for rst in sorted((ROOT / "docs").glob("api*.rst"))
    }


def documented_names(entries):
    """Return a mapping from module name to the set of names documented in it."""
    documented = defaultdict(set)
    for page in entries.values():
        for entry in page:
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


def module_name(path):
    """Return the dotted module name for a source file under ``desc/``."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def source_modules():
    """Yield the source files of all modules in the ``desc`` package."""
    for path in sorted((ROOT / "desc").rglob("*.py")):
        rel = path.relative_to(ROOT)
        # skip editor/notebook artifacts such as .ipynb_checkpoints
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path


def submodule_names(path):
    """Return names of the submodules of a package, or an empty set for a module."""
    if path.name != "__init__.py":
        return set()
    pkg = path.parent
    return {
        child.stem if child.suffix == ".py" else child.name
        for child in pkg.iterdir()
        if (child.suffix == ".py" and child.stem != "__init__")
        or (child / "__init__.py").is_file()
    }


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
    entries = documented_entries()
    documented = documented_names(entries)

    alphabetical = entries.get(ALPHABETICAL_PAGE, set())
    for page, page_entries in entries.items():
        if page == ALPHABETICAL_PAGE:
            continue
        missing = page_entries - alphabetical
        if missing:
            errors.append(
                f"docs/{page}: not listed in docs/{ALPHABETICAL_PAGE}: "
                f"{sorted(missing)}"
            )

    for modname, names in sorted(documented.items()):
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
        # a package may list its submodules in __all__ to support lazy import;
        # those are documented through their own entries
        public -= submodule_names(path)
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

    # modules whose public names are documented through a re-exporting parent
    # package are not checked above, so check them separately
    for path in source_modules():
        modname = module_name(path)
        if modname in documented:
            continue
        public = declared_all(ast.parse(path.read_text(), filename=str(path)))
        if not public:
            continue
        parents = [modname.rsplit(".", i)[0] for i in range(1, modname.count(".") + 1)]
        reexported = set().union(*(documented.get(p, set()) for p in parents))
        undocumented = public - reexported - submodule_names(path)
        if undocumented:
            errors.append(
                f"{path.relative_to(ROOT)}: in __all__ but not documented: "
                f"{sorted(undocumented)}"
            )

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
