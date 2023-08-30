"""Alphabetize classes and functions in a python source file.

Run on a single file at a time, and by default prints the sorted file to the terminal
rather than modify in place.

``python alphabetize.py desc/foo.py > desc/foo_sorted.py``
"""

import re
import sys

import black
import isort


def verify_same_lines(lines1, lines2, report=True):
    """Make sure we didn't accidentally delete or add extra lines.

    Just compares set equality, not ordering, so should return True
    if all we did is reorder/alphabetize things.

    Ignores blank lines.

    Parameters
    ----------
    lines1, lines2 : list of str
        Lines of text to compare. Need not be in the same order.
    report : bool
        Whether to print any mismatched lines.

    Returns
    -------
    valid : bool
        Whether the two sets of lines contain the same strings.
    """
    # remove empty lines and sort
    lines1 = [line.strip("\n") for line in lines1 if not line.isspace()]
    lines2 = [line.strip("\n") for line in lines2 if not line.isspace()]
    valid = set(lines1) == set(lines2)
    if not valid and report:
        print("########## lines in 1 that are not in 2: ##########")
        for line in lines1:
            if line not in lines2:
                print(line)
        print("########## lines in 2 that are not in 1: ##########")
        for line in lines2:
            if line not in lines1:
                print(line)

    return valid


def extract_functions(lines):
    """Get the source for any functions defined in a module.

    Assumes files are formatted with black so there are 2 blank lines
    between definitions.

    Parameters
    ----------
    lines : list of str
        Lines of source to search

    Returns
    -------
    funcs : list of str
        Sorted source code for any declared functions.
    """
    function_pattern = re.compile(r"^def\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE)
    functions = []
    names = []
    thisfunc = []
    temp = []
    flag = False

    for i, line in enumerate(lines):
        if (lines[i].isspace() or len(lines[i]) == 0) and (
            lines[i + 1].isspace() or len(lines[i + 1]) == 0
        ):
            flag = False
            temp = []
            if len(thisfunc):
                functions.append("\n".join(thisfunc))
                thisfunc = []
        if line.startswith("@"):
            temp.append(line)
        if function_pattern.match(line):
            flag = True
            names.append(line)
            thisfunc.extend(temp)
            temp = []
        if flag:
            thisfunc.append(line)

    if len(thisfunc):
        functions.append("\n".join(thisfunc))

    funcs = [(name, fun) for name, fun in zip(names, functions)]
    funcs = sorted(funcs)
    funcs = [fun for name, fun in funcs]
    return funcs


def extract_classes(lines):
    """Get the source for any classes defined in a module.

    Assumes files are formatted with black so there are 2 blank lines
    between definitions.

    Parameters
    ----------
    lines : list of str
        Lines of source to search

    Returns
    -------
    funcs : list of str
        Sorted source code for any declared classes.
    """
    class_pattern = re.compile(r"^class\s+([a-zA-Z_]\w*)\s*(\(|:)", re.MULTILINE)
    classes = []
    names = []
    thisclass = []
    temp = []
    flag = False

    for i, line in enumerate(lines):
        if (lines[i].isspace() or len(lines[i]) == 0) and (
            lines[i + 1].isspace() or len(lines[i + 1]) == 0
        ):
            flag = False
            temp = []
            if len(thisclass):
                classes.append("\n".join(thisclass))
                thisclass = []
        if line.startswith("@"):
            temp.append(line)
        if class_pattern.match(line):
            flag = True
            names.append(line)
            thisclass.extend(temp)
            temp = []
        if flag:
            thisclass.append(line)

    if len(thisclass):
        classes.append("\n".join(thisclass))

    classes = [sort_methods(cls) for cls in classes]

    classes = [(name, cls) for name, cls in zip(names, classes)]
    classes = sorted(classes)
    classes = [cls for name, cls in classes]
    return classes


def sort_methods(cls):
    """Sort class methods.

    Parameters
    ----------
    cls : str
        Source code of class definition

    Returns
    -------
    cls : str
        Source code with methods sorted alphabetically
    """
    lines = cls.split("\n")
    method_pattern = re.compile(r"^    def\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE)
    # first grab everything up to the first def of decorator, ie docstring and io_attrs
    for i in range(len(lines)):
        if method_pattern.match(lines[i]) or "@" in lines[i]:
            break
    preamble = lines[:i]
    lines = lines[i:]

    methods = []
    names = []
    thismethod = []
    flag = True

    for line in lines:
        if method_pattern.match(line) or "@" in line:
            if flag:
                flag = False
                # start of a new thing, save the old thing
                temp = "\n".join(thismethod)
                if len(temp) and not temp.isspace():
                    methods.append(temp)
                thismethod = []
        else:
            flag = True
        if method_pattern.match(line):
            names.append(line)
        thismethod.append(line)

    temp = "\n".join(thismethod)
    if len(temp) and not temp.isspace():
        methods.append(temp)

    props = []
    meths = []
    init = ("", "")
    # split between properties and methods, and separate __init__ method
    for name, method in zip(names, methods):
        if "__init__" in name:
            init = (name, method)
        elif "@property" in method or ("@" in method and "setter" in method):
            props.append((name, method))
        else:
            meths.append((name, method))

    meths = sorted(meths)
    props = sorted(props)

    out = "\n".join(preamble)
    out += "\n"
    out += init[1]
    out += "\n"
    for name, method in meths:
        out += method
        out += "\n"
    for name, prop in props:
        out += prop
        out += "\n"

    assert verify_same_lines(out.split("\n"), cls.split("\n"))

    return out


def extract_constants(lines):
    """Get the source for any constants defined in a module.

    Constants are anything of the form `thing = ` at the outermost
    level of indentation.

    Parameters
    ----------
    lines : list of str
        Lines of source to search

    Returns
    -------
    constants : list of str
        Sorted source code for any declared constants.
    """
    constant_pattern = re.compile(r"^[a-zA-Z_0-9]\w* = ", re.MULTILINE)
    constants = []
    this_constant = []
    flag = False
    for i, line in enumerate(lines):
        if line.isspace() or (len(line) == 0):
            flag = False
            if len(this_constant):
                constants.append("\n".join(this_constant))
                this_constant = []
        if constant_pattern.match(line):
            flag = True
            # backtrack to find comments etc that we may have missed
            for j in range(i - 1, 0, -1):
                if lines[j].isspace() or (len(lines[j]) == 0):
                    break
                this_constant = [lines[j]] + this_constant
        if flag:
            this_constant.append(line)

    # might not be blank line at end of file
    if len(this_constant):
        constants.append("\n".join(this_constant))

    return sorted(constants)


def extract_preamble(lines):
    """Get the block of text at the top of the file with imports.

    Should capture module docstring, imports, ``__all__`` etc,
    assumes file has been formatted with ``black`` and ``isort``

    Parameters
    ----------
    lines : list of str
        Lines of source to search

    Returns
    -------
    preamble : str
        Single string for block of text at the start of the file.
    """
    for i in range(len(lines)):
        if (lines[i].isspace() or len(lines[i]) == 0) and (
            lines[i + 1].isspace() or len(lines[i + 1]) == 0
        ):
            break
    return "\n".join(lines[:i])


def cleanup(code):
    """Apply black formatting and isort to a block of code."""
    # for some reason black throws a blank exception when nothing is changed
    try:
        code = black.format_file_contents(code, fast=False, mode=black.FileMode())
    except Exception as e:
        if str(e):  # if its not a blank exception
            raise e

    code = isort.code(code)
    return code


def sort_file(pathin):
    """Alphabetize the functions and classes in a python source file.

    Orders classes before functions before constants.

    Class methods are sorted alphabetically, with ``__init__`` first, and properties
    at the end.

    Also applies black and isort formatting.
    """
    with open(pathin) as f:
        linesin = "".join(f.readlines())

    linesin = cleanup(linesin).split("\n")

    # strip newline characters
    for i in range(len(linesin)):
        linesin[i] = linesin[i].strip("\n")

    preamble = extract_preamble(linesin)
    classes = extract_classes(linesin)
    functions = extract_functions(linesin)
    constants = extract_constants(linesin)

    out = preamble + "\n\n"  # 2 blank lines between preamble and classes
    for cls in classes:
        out += "\n"
        out += cls + "\n\n"  # 2 blank lines between classes
    for fun in functions:
        out += "\n"
        out += fun + "\n\n"  # 2 blank lines between functions
    for con in constants:
        out += "\n"  # 1 blank lines between constants
        out += con + "\n"

    out = cleanup(out)
    linesout = out.split("\n")
    assert verify_same_lines(linesin, linesout)
    return out


if __name__ == "__main__":
    out = sort_file(sys.argv[1])
    print(out)
