#!/usr/bin/env python3
"""Fast static-analysis check for unmarked pytest test functions.

Replaces the slow ``pytest --collect-only`` approach by parsing files with the
``ast`` module — no test-module imports, no plugin loading.
"""

import ast
import sys

REQUIRED_MARKS = {"unit", "regression", "benchmark", "memory"}


def _pytest_marks(decorators):
    """Return set of pytest mark names present in a decorator list."""
    marks = set()
    for dec in decorators:
        # @pytest.mark.<name>  or  @pytest.mark.<name>(...)
        node = dec.func if isinstance(dec, ast.Call) else dec
        # this if statement makes sure it is a pytest.mark.<name>
        # which corresponds to the first value being mark and
        # second value being a Name
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pytest"
        ):
            marks.add(node.attr)
    return marks


def check_file(filepath):
    """Return list of (lineno, qualname) for unmarked test functions in *filepath*."""
    try:
        with open(filepath, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=filepath)
    except (OSError, SyntaxError):
        # skip if can't open the file
        return []

    unmarked = []

    # Walk top-level and class-body nodes only (so ignores any helper functions inside)
    top_level = list(ast.iter_child_nodes(tree))
    for node in top_level:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                marks = _pytest_marks(node.decorator_list)
                if not marks & REQUIRED_MARKS:
                    unmarked.append((node.lineno, node.name))
        elif isinstance(node, ast.ClassDef):
            # search the methods of class for tests
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("test_"):
                        marks = _pytest_marks(child.decorator_list)
                        if not marks & REQUIRED_MARKS:
                            unmarked.append(
                                (child.lineno, f"{node.name}::{child.name}")
                            )

    return unmarked


def main(paths):
    total_unmarked = 0
    for path in paths:
        results = check_file(path)
        for lineno, name in results:
            print(f"{path}:{lineno}: {name}")
            total_unmarked += 1

    if total_unmarked:
        print(f"\n---- found {total_unmarked} unmarked tests ----")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
