"""Alphabetize classes and functions in a python source file.

Run on a single file at a time like so:

``python alphabetize.py desc/foo.py desc/foo_sorted.py``

Should throw an error (and not overwrite anything) if the sorted file doesn't contain
the same info as the original.

Most common source of errors is the presence of two blank lines in the middle of a
function or class docstring body. The parser treats this as the end of the declaration,
and starts reading the next lines as a new thing, so you may need to clean up the
original file before running this.
"""

import re
import subprocess
import sys


def keyfun(d):
    """By default _ gets sorted between Z and a, this makes _ get sorted first."""
    if isinstance(d, tuple):
        return d[0].replace("_", "&")
    return d.replace("_", "&")


def verify_same_lines(orig_lines, sort_lines, report=True):
    """Make sure we didn't accidentally delete or add extra lines.

    Just compares set equality, not ordering, so should return True
    if all we did is reorder/alphabetize things.

    Ignores blank lines.

    Parameters
    ----------
    orig_lines, sort_lines : list of str
        Lines of text to compare. Need not be in the same order.
    report : bool
        Whether to print any mismatched lines.

    Returns
    -------
    valid : bool
        Whether the two sets of lines contain the same strings.
    """
    # remove empty lines and sort
    orig_lines = [line.strip("\n") for line in orig_lines if not line.isspace()]
    sort_lines = [line.strip("\n") for line in sort_lines if not line.isspace()]
    valid = set(orig_lines) == set(sort_lines)
    if not valid and report:
        print("########## lines in original that are not in sorted: ##########")
        for line in orig_lines:
            if line not in sort_lines:
                print(line)
        print("########## lines in sorted that are not in original: ##########")
        for line in sort_lines:
            if line not in orig_lines:
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
    funcflag = False  # are we reading a function
    decflag = False  # are we reading a decorator

    for i, line in enumerate(lines):
        # double blank line indicates end of function
        if (i == len(lines) - 1) or (
            (lines[i].isspace() or len(lines[i]) == 0)
            and (lines[i + 1].isspace() or len(lines[i + 1]) == 0)
        ):
            # if we're at the end of the file be sure to include last line
            if (i == len(lines) - 1) and funcflag:
                thisfunc.append(line)
            funcflag = False
            temp = []
            if len(thisfunc):
                functions.append("\n".join(thisfunc))
                thisfunc = []
        if line.startswith("@"):
            decflag = True
        if function_pattern.match(line):
            funcflag = True
            decflag = False
            names.append(line)
            # prepend any decorator we already read
            thisfunc.extend(temp)
            temp = []
        if funcflag:
            thisfunc.append(line)
        if decflag:
            temp.append(line)

    if len(thisfunc):
        functions.append("\n".join(thisfunc))

    assert len(names) == len(functions)

    funcs = [(name, fun) for name, fun in zip(names, functions)]
    funcs = sorted(funcs, key=keyfun)
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
    classflag = False  # are we reading a class
    decflag = False  # are we reading a decorator

    for i, line in enumerate(lines):
        # look for double blank line between definitions
        if (i == len(lines) - 1) or (
            (lines[i].isspace() or len(lines[i]) == 0)
            and (lines[i + 1].isspace() or len(lines[i + 1]) == 0)
        ):
            # be sure to include line at end of file
            if (i == len(lines) - 1) and classflag:
                thisclass.append(line)
            classflag = False
            temp = []
            if len(thisclass):
                classes.append("\n".join(thisclass))
                thisclass = []
        if line.startswith("@"):
            decflag = True
        if class_pattern.match(line):
            classflag = True
            decflag = False
            names.append(line)
            thisclass.extend(temp)
            temp = []
        if classflag:
            thisclass.append(line)
        if decflag:
            temp.append(line)

    if len(thisclass):
        classes.append("\n".join(thisclass))

    assert len(names) == len(classes)

    classes = [(name, cls) for name, cls in zip(names, classes)]
    classes = sorted(classes, key=keyfun)
    classes = [cls for name, cls in classes]
    return classes


def sort_methods(klass):
    """Sort class methods

    Parameters
    ----------
    klass : str
        Source code of class definition

    Returns
    -------
    cls : str
        Source code with methods sorted alphabetically
    """
    lines = klass.split("\n")
    method_pattern = re.compile(r"^    def\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE)
    decorator_pattern = re.compile("^    @", re.MULTILINE)
    # first grab everything up to the first def or decorator, ie docstring and io_attrs
    for i in range(len(lines)):
        if method_pattern.match(lines[i]) or decorator_pattern.match(lines[i]):
            break
    preamble = lines[:i]
    lines = lines[i:]

    methods = []
    names = []
    thismethod = []
    flag = True  # are we looking for the start of a method

    for line in lines:
        if method_pattern.match(line) or decorator_pattern.match(line):
            # found the start
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

    for i in range(len(methods)):
        if not methods[i].endswith("\n"):
            methods[i] = methods[i] + "\n"

    assert len(names) == len(methods)
    props = []
    meths = []
    init = ("", "")
    # split between properties and methods, and separate __init__ method
    for name, method in zip(names, methods):
        if "__init__" in name:
            init = (name, method)
        elif "@property" in method or ("    @" in method and "setter" in method):
            props.append((name, method))
        else:
            meths.append((name, method))

    meths = sorted(meths, key=keyfun)
    props = sorted(props, key=keyfun)

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

    assert verify_same_lines(klass.split("\n"), out.split("\n"))

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
    constant_pattern = re.compile(r"^[a-zA-Z_0-9\.].* = ", re.MULTILINE)
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

    return sorted(constants, key=keyfun)


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
        if (i == len(lines) - 1) or (
            (lines[i].isspace() or len(lines[i]) == 0)
            and (lines[i + 1].isspace() or len(lines[i + 1]) == 0)
        ):
            break
    return "\n".join(lines[:i])


def cleanup(path):
    """Apply black formatting and isort to a file."""
    subprocess.run(["black", "-q", path])
    subprocess.run(["isort", "-q", path])


def sort_file(pathin, pathout):
    """Alphabetize the functions and classes in a python source file.

    Orders classes before functions before constants.

    Class methods are sorted alphabetically, with ``__init__`` first, and properties
    at the end.

    Also applies black and isort formatting.
    """
    cleanup(pathin)

    with open(pathin) as f:
        linesin = f.readlines()

    # strip newline characters
    for i in range(len(linesin)):
        linesin[i] = linesin[i].strip("\n")

    preamble = extract_preamble(linesin)
    classes = extract_classes(linesin)
    functions = extract_functions(linesin)
    constants = extract_constants(linesin)

    classes = [sort_methods(cls) for cls in classes]

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

    linesout = out.split("\n")
    assert verify_same_lines(linesin, linesout)
    with open(pathout, "w+") as f:
        f.write(out)
    cleanup(pathout)


if __name__ == "__main__":
    sort_file(sys.argv[1], sys.argv[2])
