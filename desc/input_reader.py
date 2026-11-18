"""Class for reading and writing DESC and VMEC input files."""

import argparse
import io
import os
import pathlib
import re
import warnings
from datetime import datetime

import f90nml
import numpy as np
from termcolor import colored

from desc import set_device

# shouldn't import anything else from DESC here, since that will initialize JAX
# before we have a chance to parse user inputs to see if they want to use GPU
# instead, do local imports where needed.


def _vmec_signgs(surface):
    """Sign of VMEC's Jacobian for a boundary given as rows (..., m, n, R, Z).

    VMEC's signgs is -1 when the cross-section at zeta = 0 runs counterclockwise
    in (R, Z) as theta increases.
    """
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    R = np.zeros_like(theta)
    Z = np.zeros_like(theta)
    for m, n, r, z in np.atleast_2d(surface)[:, -4:]:
        if n < 0:
            continue
        basis = np.cos(m * theta) if m >= 0 else np.sin(-m * theta)
        R += r * basis
        Z += z * basis
    area = 0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
    return -1.0 if area > 0 else 1.0


class InputReader:
    """Reads command line arguments and parses input files.

    Parameters
    ----------
    cl_args : None, str, or list (Default = None)
        command line arguments to parse
        If these are not None, then it is assumed that this
        is being invoked as a call of desc from the command
        line. In that case, if called on a vmec input file, the
        corresponding converted DESC input file will be saved
        automatically.
    save_converted_vmec_input : bool (Default = False)
        If True, then if a VMEC input file is converted to
        DESC input, will save the input file. False by default,
        but True when DESC module is invoked from the command line
        on an input file like ``python -m desc input.vmec``.

    """

    def __init__(self, cl_args=None, save_converted_vmec_input=False):
        """Initialize InputReader instance."""
        self._args = None
        self._inputs = None
        self._input_path = None
        self._output_path = None
        self._save_converted_vmec_input = save_converted_vmec_input

        if cl_args is not None:
            if isinstance(cl_args, os.PathLike):
                cl_args = list(cl_args)
            elif isinstance(cl_args, str):
                cl_args = cl_args.split(" ")
            self._args = self.parse_args(cl_args=cl_args)
            if not self.args.version:
                self._inputs = self.parse_inputs()

    @property
    def args(self):
        """Namespace : parsed namespace of all command line arguments."""
        return self._args

    @property
    def inputs(self):
        """List of dictionaries with values from input file."""
        return self._inputs

    @property
    def input_path(self):
        """Path to input file."""
        return self._input_path

    @property
    def output_path(self):
        """Path to output file."""
        return self._output_path

    def parse_args(self, cl_args=None):
        """Parse command line arguments.

        Parameters
        ----------
        cl_args : None, str, or list (Default = None)
            command line arguments to parse

        Returns
        -------
        args : namespace
            parsed arguments

        """
        if cl_args is None:
            return

        if isinstance(cl_args, os.PathLike):
            cl_args = list(cl_args)
        elif isinstance(cl_args, str):
            cl_args = cl_args.split(" ")

        self.parser = self._get_parser_()
        args = self.parser.parse_args(cl_args)

        if args.version:
            import desc

            print(desc.__version__)
            return args

        if len(args.input_file) == 0:
            raise NameError("Input file path must be specified")

        self._input_path = pathlib.Path(args.input_file[0]).resolve()
        if self.input_path.is_file():
            self._input_path = str(self.input_path)
        else:
            raise FileNotFoundError(
                "Input file '{}' does not exist.".format(str(self.input_path))
            )

        if args.output:
            self._output_path = args.output
        else:
            self._output_path = self.input_path + "_output.h5"

        if args.numpy:
            os.environ["DESC_BACKEND"] = "numpy"
        else:
            os.environ["DESC_BACKEND"] = "jax"

        if args.quiet:
            args.verbose = 0

        if args.gpu:
            set_device("gpu")
        else:
            set_device("cpu")

        return args

    def _get_parser_(self):
        """Get parser for command line arguments.

        Returns
        -------
        parser : argparse object
            argument parser

        """
        return get_parser()

    def parse_inputs(self, fname=None):  # noqa: C901
        """Read input from DESC input file; converts from VMEC input if necessary.

        Parameters
        ----------
        fname : string
            filename of input file

        Returns
        -------
        inputs : list of dictionaries
            all the input parameters and options. One dictionary for each resolution
            level.
        """
        if fname is None:
            fname = self.input_path
        else:
            self._input_path = fname

        # default values
        inputs = {
            "sym": False,
            "NFP": 1,
            "Psi": 1.0,
            "L": np.atleast_1d(None),
            "M": np.atleast_1d(0),
            "N": np.atleast_1d(0),
            "L_grid": np.atleast_1d(None),
            "M_grid": np.atleast_1d(0),
            "N_grid": np.atleast_1d(0),
            "pres_ratio": np.atleast_1d(None),
            "curr_ratio": np.atleast_1d(None),
            "bdry_ratio": np.atleast_1d(None),
            "pert_order": np.atleast_1d(2),
            "ftol": np.atleast_1d(None),
            "xtol": np.atleast_1d(None),
            "gtol": np.atleast_1d(None),
            "maxiter": np.atleast_1d(None),
            "objective": "force",
            "optimizer": "lsq-exact",
            "spectral_indexing": "ansi",
            "bdry_mode": "lcfs",
            "pressure": np.atleast_2d((0, 0.0)),
            "iota": np.atleast_2d((0, 0.0)),
            "current": np.atleast_2d((0, 0.0)),
            "surface": np.atleast_2d((0, 0, 0, 0.0, 0.0)),
            "axis": np.atleast_2d((0, 0.0, 0.0)),
        }

        iota_flag = False
        pres_flag = False
        curr_flag = False
        vac_flag = False
        splines = {}
        inputs["output_path"] = self.output_path

        if self.args is not None and self.args.quiet:
            inputs["verbose"] = 0
        elif self.args is not None:
            inputs["verbose"] = self.args.verbose
        else:
            inputs["verbose"] = 1

        # open files, unless they are already open files
        if not isinstance(fname, io.IOBase):
            file = open(fname)
        else:
            file = fname
        file.seek(0)
        lines = file.readlines()
        file.close()

        num_form = r"[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?"

        for line in lines:
            # check if VMEC input file format
            isVMEC = re.search(r"&INDATA", line, re.IGNORECASE)
            if isVMEC:
                print("Converting VMEC input to DESC input")
                # use a buffer here to avoid having to read/write a file
                # unless specified by flag
                path = (
                    self.input_path + "_desc"
                    if self._save_converted_vmec_input
                    else io.StringIO()
                )
                InputReader.vmec_to_desc_input(self.input_path, path)
                if self._save_converted_vmec_input:
                    print("Generated DESC input file {}:".format(path))
                else:
                    # put buffer back to start so it may be read again
                    path.seek(0)
                return self.parse_inputs(path)

            # extract numbers & words
            # strip first: comment and equals are indexes into the stripped line
            line = line.strip()
            match = re.search(r"[!#]", line)
            if match:
                comment = match.start()
            else:
                comment = len(line)
            match = re.search(r"=", line)
            if match:
                equals = match.start()
            else:
                equals = len(line)
            command = (line + " ")[0:comment]
            argument = (command.strip() + " ")[0:equals]
            data = command[equals + 1 :]
            words = data.split()
            num_list = re.split(r"[\s,;]", data)
            numbers = np.array([])
            for txt in num_list:
                # format like 4:2:12 = 4,6,8,10,12
                if re.search(num_form + ":" + num_form + ":" + num_form, txt):
                    nums = [
                        float(x)
                        for x in re.findall(num_form, txt)
                        if re.search(r"\d", x)
                    ]
                    if len(nums):
                        numbers = np.append(
                            numbers, np.arange(nums[0], nums[2] + nums[1], nums[1])
                        )
                # format like 12x4 = 12,12,12,12
                elif re.search(num_form + "x" + num_form, txt):
                    nums = [
                        float(x)
                        for x in re.findall(num_form, txt)
                        if re.search(r"\d", x)
                    ]
                    if len(nums):
                        numbers = np.append(numbers, np.tile(nums[0], int(nums[1])))
                # individual numbers
                else:
                    nums = [
                        float(x)
                        for x in re.findall(num_form, txt)
                        if re.search(r"\d", x)
                    ]
                    if len(nums):
                        numbers = np.append(numbers, nums)
            flag = False  # flag for valid input lines

            # global parameters
            match = re.search(r"sym", argument, re.IGNORECASE)
            if match:
                inputs["sym"] = int(numbers[0])
                flag = True
            match = re.search(r"NFP", argument, re.IGNORECASE)
            if match:
                assert numbers[0] == int(numbers[0]), "NFP should be an integer"
                inputs["NFP"] = int(numbers[0])
                flag = True
            match = re.search(r"Psi", argument, re.IGNORECASE)
            if match:
                inputs["Psi"] = numbers[0]
                flag = True

            # spectral resolution
            match = re.search(r"L_rad", argument, re.IGNORECASE)
            if match:
                inputs["L"] = numbers.astype(int)
                flag = True
            match = re.search(r"M_pol", argument, re.IGNORECASE)
            if match:
                inputs["M"] = numbers.astype(int)
                flag = True
            match = re.search(r"N_tor", argument, re.IGNORECASE)
            if match:
                inputs["N"] = numbers.astype(int)
                flag = True
            match = re.search(r"L_grid", argument, re.IGNORECASE)
            if match:
                inputs["L_grid"] = np.array(numbers).astype(int)
                flag = True
            match = re.search(r"M_grid", argument, re.IGNORECASE)
            if match:
                inputs["M_grid"] = numbers.astype(int)
                flag = True
            match = re.search(r"N_grid", argument, re.IGNORECASE)
            if match:
                inputs["N_grid"] = numbers.astype(int)
                flag = True

            # continuation parameters
            match = re.search(r"pres_ratio", argument, re.IGNORECASE)
            if match:
                inputs["pres_ratio"] = numbers.astype(float)
                flag = True
            match = re.search(r"curr_ratio", argument, re.IGNORECASE)
            if match:
                inputs["curr_ratio"] = numbers.astype(float)
                flag = True
            match = re.search(r"bdry_ratio", argument, re.IGNORECASE)
            if match:
                inputs["bdry_ratio"] = numbers.astype(float)
                flag = True
            match = re.search(r"pert_order", argument, re.IGNORECASE)
            if match:
                inputs["pert_order"] = numbers.astype(int)
                flag = True

            # solver tolerances
            match = re.search(r"ftol", argument, re.IGNORECASE)
            if match:
                inputs["ftol"] = numbers.astype(float)
                flag = True
            match = re.search(r"xtol", argument, re.IGNORECASE)
            if match:
                inputs["xtol"] = numbers.astype(float)
                flag = True
            match = re.search(r"gtol", argument, re.IGNORECASE)
            if match:
                inputs["gtol"] = numbers.astype(float)
                flag = True
            match = re.search(r"nfev", argument, re.IGNORECASE)
            if match:
                warnings.warn(
                    DeprecationWarning("nfev is deprecated, please use maxiter instead")
                )
                inputs["maxiter"] = np.array(
                    [None if i == 0 else int(i) for i in numbers]
                )
                flag = True
            match = re.search(r"maxiter", argument, re.IGNORECASE)
            if match:
                inputs["maxiter"] = np.array(
                    [None if i == 0 else int(i) for i in numbers]
                )
                flag = True

            # solver methods
            match = re.search(r"objective", argument, re.IGNORECASE)
            if match:
                method = words[0].lower()
                if method == "vacuum":
                    method = "force"
                    vac_flag = True
                inputs["objective"] = method
                flag = True
            match = re.search(r"optimizer", argument, re.IGNORECASE)
            if match:
                inputs["optimizer"] = words[0].lower()
                flag = True
            match = re.search(r"spectral_indexing", argument, re.IGNORECASE)
            if match:
                inputs["spectral_indexing"] = words[0].lower()
                flag = True
            match = re.search(r"node_pattern", argument, re.IGNORECASE)
            if match:
                warnings.warn(
                    "node_pattern parameter has been deprecated. To use a custom node "
                    + "pattern, pass in the desired grid when creating the objective "
                    + "in a python script."
                )
                flag = True
            match = re.search(r"bdry_mode", argument, re.IGNORECASE)
            if match:
                inputs["bdry_mode"] = words[0].lower()
                flag = True
            match = re.search(
                r"spline_(rho|pressure|iota|current)", argument, re.IGNORECASE
            )
            if match:
                splines[match.group(1).lower()] = numbers.astype(float)
                flag = True

            # coefficient indices
            match = re.search(r"l\s*:\s*" + num_form, command, re.IGNORECASE)
            if match:
                l = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
            else:
                l = 0
            match = re.search(r"m\s*:\s*" + num_form, command, re.IGNORECASE)
            if match:
                m = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
            else:
                m = 0
            match = re.search(r"n\s*:\s*" + num_form, command, re.IGNORECASE)
            if match:
                n = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
            else:
                n = 0

            # profile coefficients
            match = re.search(r"\sp\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                pres_flag = True
                p_l = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
                prof_idx = np.where(inputs["pressure"][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs["pressure"].shape[0])
                    inputs["pressure"] = np.pad(
                        inputs["pressure"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["pressure"][prof_idx[0], 0] = l
                inputs["pressure"][prof_idx[0], 1] = p_l
                flag = True
            match = re.search(r"\si\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                iota_flag = True
                i_l = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
                prof_idx = np.where(inputs["iota"][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs["iota"].shape[0])
                    inputs["iota"] = np.pad(
                        inputs["iota"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["iota"][prof_idx[0], 0] = l
                inputs["iota"][prof_idx[0], 1] = i_l
                flag = True
            match = re.search(r"\sc\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                curr_flag = True
                c_l = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
                prof_idx = np.where(inputs["current"][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs["current"].shape[0])
                    inputs["current"] = np.pad(
                        inputs["current"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["current"][prof_idx[0], 0] = l
                inputs["current"][prof_idx[0], 1] = c_l
                flag = True

            # boundary surface coefficients
            match = re.search(r"R1\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                R1 = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][1]
                bdry_idx = np.where(
                    (inputs["surface"][:, :3] == [l, m, n]).all(axis=1)
                )[0]
                if bdry_idx.size == 0:
                    bdry_idx = np.atleast_1d(inputs["surface"].shape[0])
                    inputs["surface"] = np.pad(
                        inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["surface"][bdry_idx[0], 0] = l
                    inputs["surface"][bdry_idx[0], 1] = m
                    inputs["surface"][bdry_idx[0], 2] = n
                inputs["surface"][bdry_idx[0], 3] = R1
                flag = True
            match = re.search(r"Z1\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                Z1 = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][1]
                bdry_idx = np.where(
                    (inputs["surface"][:, :3] == [l, m, n]).all(axis=1)
                )[0]
                if bdry_idx.size == 0:
                    bdry_idx = np.atleast_1d(inputs["surface"].shape[0])
                    inputs["surface"] = np.pad(
                        inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["surface"][bdry_idx[0], 0] = l
                    inputs["surface"][bdry_idx[0], 1] = m
                    inputs["surface"][bdry_idx[0], 2] = n
                inputs["surface"][bdry_idx[0], 4] = Z1
                flag = True

            # magnetic axis coefficients
            match = re.search(r"R0\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                R0_n = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][1]
                axis_idx = np.where(inputs["axis"][:, 0] == n)[0]
                if axis_idx.size == 0:
                    axis_idx = np.atleast_1d(inputs["axis"].shape[0])
                    inputs["axis"] = np.pad(
                        inputs["axis"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["axis"][axis_idx[0], 0] = n
                inputs["axis"][axis_idx[0], 1] = R0_n
                flag = True
            match = re.search(r"Z0\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                Z0_n = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][1]
                axis_idx = np.where(inputs["axis"][:, 0] == n)[0]
                if axis_idx.size == 0:
                    axis_idx = np.atleast_1d(inputs["axis"].shape[0])
                    inputs["axis"] = np.pad(
                        inputs["axis"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["axis"][axis_idx[0], 0] = n
                inputs["axis"][axis_idx[0], 2] = Z0_n
                flag = True

            # catch lines that don't match a valid command
            if not flag and command not in ["", " "]:
                raise OSError(
                    colored(
                        "The following line is not a valid input:\n" + command, "red"
                    )
                )

        # error handling
        if np.any(inputs["M"] == 0):
            raise OSError(colored("M_pol is not assigned.", "red"))
        if np.sum(inputs["surface"]) == 0:
            raise OSError(colored("Fixed-boundary surface is not assigned.", "red"))
        if curr_flag and iota_flag:
            raise OSError(colored("Cannot specify both iota and current.", "red"))

        if vac_flag and (pres_flag or iota_flag or curr_flag):
            warnings.warn(
                "Vacuum objective assumes 0 pressure and 0 current, "
                + "ignoring provided pressure, iota, and current profiles"
            )
            _ = inputs.pop("iota", None)
            _ = inputs.pop("current", None)
            _ = inputs.pop("pressure", None)

        # remove unused profile
        if iota_flag:
            _ = inputs.pop("current", None)
        else:
            _ = inputs.pop("iota", None)

        # sort axis array
        inputs["axis"] = inputs["axis"][inputs["axis"][:, 0].argsort()]

        # sort surface array
        inputs["surface"] = inputs["surface"][inputs["surface"][:, 2].argsort()]
        inputs["surface"] = inputs["surface"][
            inputs["surface"][:, 1].argsort(kind="mergesort")
        ]
        inputs["surface"] = inputs["surface"][
            inputs["surface"][:, 0].argsort(kind="mergesort")
        ]

        # array inputs
        arrs = [
            "L",
            "M",
            "N",
            "L_grid",
            "M_grid",
            "N_grid",
            "pres_ratio",
            "curr_ratio",
            "bdry_ratio",
            "pert_order",
            "ftol",
            "xtol",
            "gtol",
            "maxiter",
        ]
        arr_len = 0
        for a in arrs:
            arr_len = max(arr_len, len(inputs[a]))
        for a in arrs:
            if inputs[a].size < arr_len:
                inputs[a] = np.append(
                    inputs[a], np.tile(inputs[a][-1], arr_len - len(inputs[a]))
                )

        # unsupplied values
        if np.sum(inputs["M_grid"]) == 0:
            inputs["M_grid"] = (2 * inputs["M"]).astype(int)
        if np.sum(inputs["N_grid"]) == 0:
            inputs["N_grid"] = (2 * inputs["N"]).astype(int)
        if np.sum(inputs["axis"]) == 0:
            axis_idx = np.where(inputs["surface"][:, 1] == 0)[0]
            inputs["axis"] = inputs["surface"][axis_idx, 2:]
        if None in inputs["L"]:
            default_L = {
                "ansi": inputs["M"],
                "fringe": 2 * inputs["M"],
            }
            inputs["L"] = default_L[inputs["spectral_indexing"]]
        if None in inputs["L_grid"]:
            default_L_grid = {
                "ansi": inputs["M_grid"],
                "fringe": 2 * inputs["M_grid"],
            }
            inputs["L_grid"] = default_L_grid[inputs["spectral_indexing"]]

        # split into list of dicts
        inputs_list = []
        for ii in range(arr_len):
            inputs_ii = {}
            for key in inputs:
                if key in arrs:
                    inputs_ii[key] = inputs[key][ii]
                elif isinstance(inputs[key], np.ndarray):
                    inputs_ii[key] = inputs[key].copy()
                else:
                    inputs_ii[key] = inputs[key]
            # apply pressure ratio
            if "pressure" in inputs_ii and inputs_ii["pres_ratio"] is not None:
                inputs_ii["pressure"][:, 1] *= inputs_ii["pres_ratio"]
            # apply current ratio
            if "current" in inputs_ii and inputs_ii["curr_ratio"] is not None:
                inputs_ii["current"][:, 1] *= inputs_ii["curr_ratio"]
            else:
                del inputs_ii["curr_ratio"]
            # apply boundary ratio
            bdry_factor = np.where(
                inputs_ii["surface"][:, 2] != 0, inputs_ii["bdry_ratio"], 1.0
            )
            if inputs_ii["bdry_ratio"] is not None:
                inputs_ii["surface"][:, 3] *= bdry_factor
                inputs_ii["surface"][:, 4] *= bdry_factor
            inputs_list.append(inputs_ii)

        if splines:
            from desc.profiles import SplineProfile

            for inputs_ii in inputs_list:
                for name in ("pressure", "iota", "current"):
                    if name in splines and name in inputs_ii:
                        ratio = {
                            "pressure": inputs_ii.get("pres_ratio"),
                            "current": inputs_ii.get("curr_ratio"),
                        }.get(name)
                        values = splines[name] * (1.0 if ratio is None else ratio)
                        inputs_ii[name] = SplineProfile(
                            values, splines["rho"], name=name
                        )

        return inputs_list

    @staticmethod
    def write_desc_input(filename, inputs, header=""):  # noqa: C901 - fxn too complex
        """Generate a DESC input file from a dictionary of parameters.

        Parameters
        ----------
        filename : str or path-like
            name of the file to create
        inputs : dict or list of dict
            dictionary of input parameters
        header : str
            text to print at the top of the file

        """
        # open the file, unless its already open
        opened_here = False
        if not isinstance(filename, io.IOBase):
            filename = os.path.expanduser(filename)
            f = open(filename, "w+")
            opened_here = True
        else:
            f = filename
        f.seek(0)

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        f.write(header + "\n")

        f.write("# global parameters\n")
        f.write("sym = {:1d} \n".format(inputs[0]["sym"]))
        f.write("NFP = {:3d} \n".format(int(inputs[0]["NFP"])))
        f.write("Psi = {:.8e} \n".format(inputs[0]["Psi"]))

        f.write("\n# spectral resolution\n")
        for key, val in {
            "L_rad": "L",
            "M_pol": "M",
            "N_tor": "N",
            "L_grid": "L_grid",
            "M_grid": "M_grid",
            "N_grid": "N_grid",
        }.items():
            f.write(
                key + " = {}\n".format(", ".join([str(inp[val]) for inp in inputs]))
            )

        f.write("\n# continuation parameters\n")
        for key in ["bdry_ratio", "pres_ratio", "curr_ratio", "pert_order"]:
            inputs_not_None = []
            for inp in inputs:
                if inp.get(key) is not None:
                    inputs_not_None.append(inp)
            if not inputs_not_None:  # an  empty list evals to False
                continue  # don't write line if all input tolerance are None

            f.write(
                key
                + " = {} \n".format(
                    ", ".join([str(float(inp[key])) for inp in inputs_not_None])
                )
            )

        f.write("\n# solver tolerances\n")
        for key in ["ftol", "xtol", "gtol", "maxiter"]:
            inputs_not_None = []
            for inp in inputs:
                if inp[key] is not None:
                    inputs_not_None.append(inp)
            if not inputs_not_None:  # an  empty list evals to False
                continue  # don't write line if all input tolerance are None

            f.write(
                key
                + " = {}\n".format(
                    ", ".join([str(inp[key]) for inp in inputs_not_None])
                )
            )

        f.write("\n# solver methods\n")
        f.write("optimizer = {}\n".format(inputs[0]["optimizer"]))
        f.write("objective = {}\n".format(inputs[0]["objective"]))
        f.write("bdry_mode = {}\n".format(inputs[0]["bdry_mode"]))
        f.write("spectral_indexing = {}\n".format(inputs[0]["spectral_indexing"]))

        if inputs[-1].get("spline"):
            f.write("\n# profiles as cubic splines in rho\n")
            for key, values in inputs[-1]["spline"].items():
                f.write(
                    "spline_{} = {}\n".format(
                        key, ", ".join(repr(float(v)) for v in values)
                    )
                )

        f.write("\n# pressure and rotational transform/current profiles\n")
        if "iota" in inputs[-1].keys():
            char = "i"
            profile = inputs[-1]["iota"]
        elif "current" in inputs[-1].keys():
            char = "c"
            profile = inputs[-1]["current"]
        ls = np.unique(np.concatenate([inputs[-1]["pressure"][:, 0], profile[:, 0]]))
        for l in ls:
            idx = np.where(l == inputs[-1]["pressure"][:, 0])[0]
            if len(idx):
                p = inputs[-1]["pressure"][idx[0], 1]
            else:
                p = 0.0
            idx = np.where(l == profile[:, 0])[0]
            if len(idx):
                i = profile[idx[0], 1]
            else:
                i = 0.0
            f.write(
                "l: {:3d}\tp = {:16.8E}\t{} = {:16.8E}\n".format(int(l), p, char, i)
            )

        f.write("\n# fixed-boundary surface shape\n")
        for l, m, n, R1, Z1 in inputs[-1]["surface"]:
            f.write(
                "l: {:3d}\tm: {:3d}\tn: {:3d}\tR1 = {:16.8E}\tZ1 = {:16.8E}\n".format(
                    int(l), int(m), int(n), R1, Z1
                )
            )

        f.write("\n# magnetic axis initial guess\n")
        for n, R0, Z0 in inputs[0]["axis"]:
            f.write("n: {:3d}\tR0 = {:16.8E}\tZ0 = {:16.8E}\n".format(int(n), R0, Z0))

        # only close the file if we opened it here, as we also
        # can use this function to write to an already open file
        # or to an in-memory buffer
        if opened_here:
            f.close()

    @staticmethod
    def desc_output_to_input(  # noqa: C901 - fxn too complex
        outfile,
        infile,
        objective="force",
        optimizer="lsq-exact",
        header=None,
        ftol=1e-2,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=100,
        threshold=0,
    ):
        """Generate a DESC input file from a DESC output file.

        DESC will automatically choose continuation parameters

        Parameters
        ----------
        outfile : str or path-like
            name of the DESC input file to create
        infile : str or path-like
            path of the DESC output equilibrium file
        objective : str
            objective type used in the input file
        optimizer : str
            type of optimizer
        header : str
            text to print at the top of the file
        ftol : float
            relative tolerance of the objective function f
        xtol : float
            relative tolerance of the state vector x
        gtol : float
            absolute tolerance of the projected gradient g
        maxiter : int
            maximum number of optimizer iterations per continuation step
        threshold : float
            Fourier coefficients below this value will be set to 0.
        """
        from desc.grid import LinearGrid
        from desc.io.optimizable_io import load
        from desc.profiles import PowerSeriesProfile
        from desc.utils import copy_coeffs

        f = open(outfile, "w+")

        f.seek(0)

        fam = load(infile)
        try:
            eq = fam[-1]
        except TypeError:
            eq = fam

        if header is None:
            header = "# DESC input file generated from the output file:\n# " + infile
        f.write(header + "\n")

        f.write("\n# global parameters\n")
        f.write("sym = {:d}\n".format(eq.sym))
        f.write("NFP = {:d}\n".format(int(eq.NFP)))
        f.write("Psi = {:.8E}\n".format(eq.Psi))

        f.write("\n# spectral resolution\n")
        for key, val in {
            "L_rad": "L",
            "M_pol": "M",
            "N_tor": "N",
            "L_grid": "L_grid",
            "M_grid": "M_grid",
            "N_grid": "N_grid",
        }.items():
            f.write(f"{key} = {getattr(eq, val)}\n")

        f.write("\n# solver tolerances\n")
        f.write("ftol = {:.3E}\n".format(ftol))
        f.write("xtol = {:.3E}\n".format(xtol))
        f.write("gtol = {:.3E}\n".format(gtol))
        f.write("maxiter = {:d}\n".format(maxiter))

        f.write("\n# solver methods\n")
        f.write(f"optimizer = {optimizer}\n")
        f.write(f"objective = {objective}\n")
        f.write("spectral_indexing = {}\n".format(eq._spectral_indexing))

        # fit profiles to power series
        grid = LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        rho = grid.nodes[grid.unique_rho_idx, 0]
        if not isinstance(eq.pressure, PowerSeriesProfile):
            pressure = grid.compress(eq.compute("p", grid=grid)["p"])
            pres_profile = PowerSeriesProfile.from_values(
                rho, pressure, order=eq.L, sym=False
            )
        else:
            pres_profile = eq.pressure
        if not isinstance(eq.iota, PowerSeriesProfile):
            iota = grid.compress(eq.compute("iota", grid=grid)["iota"])
            iota_profile = PowerSeriesProfile.from_values(
                rho, iota, order=eq.L, sym=False
            )
        else:
            iota_profile = eq.iota
        if not isinstance(eq.current, PowerSeriesProfile):
            current = grid.compress(eq.compute("current", grid=grid)["current"])
            curr_profile = PowerSeriesProfile.from_values(
                rho, current, order=eq.L, sym=False
            )
        else:
            curr_profile = eq.current

        # ensure pressure and iota/current profiles are the same resolution
        if eq.iota:
            char = "i"
            profile = iota_profile
        else:
            char = "c"
            profile = curr_profile
        L_profile = max(pres_profile.basis.L, profile.basis.L)
        pres_profile.change_resolution(L=L_profile)
        profile.change_resolution(L=L_profile)

        prof_modes = np.zeros((L_profile + 1, 3))
        prof_modes[:, 0] = np.arange(L_profile + 1)
        p1 = copy_coeffs(pres_profile.params, pres_profile.basis.modes, prof_modes)
        p2 = copy_coeffs(profile.params, profile.basis.modes, prof_modes)
        f.write("\n# pressure and rotational transform/current profiles\n")
        for l in range(L_profile + 1):
            f.write(
                "l: {:3d}  p = {:15.8E}  {} = {:15.8E}\n".format(
                    int(l), p1[l], char, p2[l]
                )
            )

        f.write("\n# fixed-boundary surface shape\n")
        if eq.sym:
            for k, (l, m, n) in enumerate(eq.surface.R_basis.modes):
                if abs(eq.Rb_lmn[k]) > threshold:
                    f.write(
                        "l: {:3d}  m: {:3d}  n: {:3d}  ".format(int(0), m, n)
                        + "R1 = {:15.8E}  Z1 = {:15.8E}\n".format(eq.Rb_lmn[k], 0)
                    )
            for k, (l, m, n) in enumerate(eq.surface.Z_basis.modes):
                if abs(eq.Zb_lmn[k]) > threshold:
                    f.write(
                        "l: {:3d}  m: {:3d}  n: {:3d}  ".format(int(0), m, n)
                        + "R1 = {:15.8E}  Z1 = {:15.8E}\n".format(0, eq.Zb_lmn[k])
                    )
        else:
            for k, (l, m, n) in enumerate(eq.surface.R_basis.modes):
                if abs(eq.Rb_lmn[k]) > threshold or abs(eq.Zb_lmn[k]) > threshold:
                    f.write(
                        "l: {:3d}  m: {:3d}  n: {:3d}  ".format(int(0), m, n)
                        + "R1 = {:15.8E}  Z1 = {:15.8E}\n".format(
                            eq.Rb_lmn[k], eq.Zb_lmn[k]
                        )
                    )

        f.close()

    @staticmethod
    def vmec_to_desc_input(vmec_fname, desc_fname):
        """Convert a VMEC input file to an equivalent DESC input file.

        Parameters
        ----------
        vmec_fname : str or path-like
            filename of VMEC input file
        desc_fname : str or path-like
            filename of DESC input file. If it already exists it is overwritten.

        """
        now = datetime.now()
        date = now.strftime("%m/%d/%Y")
        time = now.strftime("%H:%M:%S")
        header = (
            "# This DESC input file was auto generated from the VMEC input file\n"
            + "# {}\n# on {} at {}.\n".format(vmec_fname, date, time)
            + "# For details on the various options see "
            + "https://desc-docs.readthedocs.io/en/stable/input.html\n"
        )
        inputs = InputReader.parse_vmec_inputs(vmec_fname)
        InputReader.write_desc_input(desc_fname, inputs, header)

    @staticmethod
    def parse_vmec_inputs(vmec_fname, threshold=0):  # noqa: C901
        """Parse a VMEC input file into a dictionary of DESC inputs.

        Parameters
        ----------
        vmec_fname : str or PathLike
            Path to VMEC input file.
        threshold : float
            Threshold value of boundary surface magnitudes to ignore.

        Returns
        -------
        inputs : dict
            Dictionary of inputs formatted for DESC.

        """
        if not isinstance(vmec_fname, io.IOBase):
            vmec_file = open(vmec_fname)
        else:
            vmec_file = vmec_fname
        vmec_file.seek(0)
        # f90nml fails if a namelist is terminated by both / and &END,
        # which some VMEC input files have, so replace all &END with /
        vmec_str = re.sub(r"(?i)[&$]end\b", "/", vmec_file.read())
        vmec_file.close()
        # only parse the INDATA namelist, since other namelists in the same file
        # (such as the STELLOPT &OPTIMUM) may have inputs that f90nml cannot parse
        start = re.search(r"(?im)^\s*[&$]indata\b", vmec_str)
        if start is None:
            raise ValueError(
                colored(f"No &INDATA namelist found in {vmec_fname}!", "red")
            )
        end = re.compile(r"(?m)^\s*[&$]").search(vmec_str, start.end())
        vmec_str = vmec_str[start.start() : end.start() if end else None] + "\n/\n"
        _warn_duplicate_vmec_inputs(vmec_str)
        parser = f90nml.Parser()
        # VMEC 1D arrays (profiles and axis) are indexed starting from 0
        parser.default_start_index = 0
        vmec_indata = parser.reads(vmec_str)["indata"]

        # default values
        inputs = {
            "sym": True,
            "NFP": 1,
            "Psi": 1.0,
            "L": None,
            "M": 0,
            "N": 0,
            "L_grid": None,
            "M_grid": 0,
            "N_grid": 0,
            "pres_ratio": None,
            "curr_ratio": None,
            "bdry_ratio": None,
            "pert_order": 2,
            "ftol": None,
            "xtol": None,
            "gtol": None,
            "maxiter": None,
            "objective": "force",
            "optimizer": "lsq-exact",
            "spectral_indexing": "ansi",
            "bdry_mode": "lcfs",
            "pressure": np.atleast_2d((0, 0.0)),
            "iota": np.atleast_2d((0, 0.0)),
            "current": np.atleast_2d((0, 0.0)),
            "surface": np.atleast_2d((0, 0, 0.0, 0.0)),
            "axis": np.atleast_2d((0, 0.0, 0.0)),
        }

        iota_flag = True
        pres_scale = 1.0
        curr_tor = None

        # VMEC defaults LFREEB to true and runs free-boundary whenever a grid is named
        mgrid_file = str(vmec_indata.get("MGRID_FILE", "")).strip().lower()
        if vmec_indata.get("LFREEB", True) and mgrid_file not in ("", "none"):
            warnings.warn(
                colored(
                    "Detected free-boundary mode in VMEC input file!"
                    " DESC will run as fixed-boundary, use a python script"
                    " to run DESC in free-boundary mode.",
                    "yellow",
                )
            )
        LRFP = vmec_indata.get("LRFP", False)
        if LRFP:
            warnings.warn(
                colored(
                    "Using poloidal flux instead of toroidal flux! "
                    + " DESC will read as toroidal flux.",
                    "yellow",
                )
            )
        inputs["sym"] = not vmec_indata.get("LASYM", False)
        inputs["NFP"] = vmec_indata.get("NFP", inputs["NFP"])
        inputs["Psi"] = vmec_indata.get("PHIEDGE", inputs["Psi"])

        MPOL = vmec_indata.get("MPOL", inputs["M"] + 1)
        M = MPOL - 1
        inputs["M"] = M
        inputs["L"] = M
        inputs["L_grid"] = 2 * M
        inputs["M_grid"] = 2 * M

        NTOR = vmec_indata.get("NTOR", inputs["N"])
        inputs["N"] = NTOR
        inputs["N_grid"] = 2 * NTOR

        # 0 means iota profile used, 1 means current (I'(s)) profile
        NCURR = vmec_indata.get("NCURR", 0)
        if NCURR != 0:
            iota_flag = False

        # profile types, converted to splines below unless they are power series
        ptype = vmec_indata.get("PMASS_TYPE", "power_series").strip().lower()
        itype = vmec_indata.get("PIOTA_TYPE", "power_series").strip().lower()
        ctype = vmec_indata.get("PCURR_TYPE", "power_series").strip().lower()
        if vmec_indata.get("GAMMA", 0.0) != 0:
            warnings.warn(
                colored("GAMMA is not 0.0, DESC will set this to 0.0.", "yellow")
            )
        bloat = float(vmec_indata.get("BLOAT", 1.0))
        spres_ped = abs(float(vmec_indata.get("SPRES_PED", 1.0)))
        # read pressure
        pres_scale = float(vmec_indata.get("PRES_SCALE", 1.0))
        AM = _get_vmec_vector(vmec_indata, "AM", np.zeros(1))
        # VMEC profile coefs are in increasing order of s powers, s=rho**2
        ls = np.arange(0, AM.size) * 2
        inputs["pressure"] = np.vstack([ls, AM]).T

        # read iota
        AI = _get_vmec_vector(vmec_indata, "AI", np.zeros(1))
        ls = np.arange(0, AI.size) * 2
        inputs["iota"] = np.vstack([ls, AI]).T

        # read current
        curr_tor = vmec_indata.get("CURTOR", None)
        AC = _get_vmec_vector(vmec_indata, "AC", np.zeros(1))
        # power_series is I'(s) = sum_i AC(i) s^i
        # power_series_I is I(s) = sum_i AC(i) s^(i+1)
        ls = (
            np.arange(0, AC.size) * 2
            if ctype.lower() == "power_series"
            else np.arange(1, AC.size + 1) * 2
        )
        inputs["current"] = np.vstack([ls, AC]).T

        # magnetic axis, RAXIS and ZAXIS are old names for RAXIS_CC and ZAXIS_CS
        RAXIS_CC = _get_vmec_vector(
            vmec_indata, "RAXIS", _get_vmec_vector(vmec_indata, "RAXIS_CC")
        )
        RAXIS_CS = _get_vmec_vector(vmec_indata, "RAXIS_CS")
        ZAXIS_CS = _get_vmec_vector(
            vmec_indata, "ZAXIS", _get_vmec_vector(vmec_indata, "ZAXIS_CS")
        )
        ZAXIS_CC = _get_vmec_vector(vmec_indata, "ZAXIS_CC")
        axis_coeffs = [
            x for x in [RAXIS_CC, RAXIS_CS, ZAXIS_CS, ZAXIS_CC] if x is not None
        ]
        if len(axis_coeffs):
            N_axis = max(x.size for x in axis_coeffs) - 1
            # DESC input axis array is (k,3) array of [n, R, Z],
            # where the row index of mode n is N_axis + n
            inputs["axis"] = np.zeros((2 * N_axis + 1, 3))
            inputs["axis"][:, 0] = np.arange(-N_axis, N_axis + 1)
            if RAXIS_CC is not None:
                inputs["axis"][N_axis : N_axis + RAXIS_CC.size, 1] = RAXIS_CC
            if ZAXIS_CC is not None:
                inputs["axis"][N_axis : N_axis + ZAXIS_CC.size, 2] = ZAXIS_CC
            # TODO: Why do we have to negate the sin modes here?
            # (ignore n=0 for sin modes)
            if RAXIS_CS is not None:
                inputs["axis"][N_axis - np.arange(1, RAXIS_CS.size), 1] = -RAXIS_CS[1:]
            if ZAXIS_CS is not None:
                inputs["axis"][N_axis - np.arange(1, ZAXIS_CS.size), 2] = -ZAXIS_CS[1:]

        # boundary surface
        # VMEC uses a series in cos/sin(m*t-n*p), while DESC uses products of
        # cos/sin(m*t) and cos/sin(n*p), with negative m,n denoting sin, so
        # RBC*cos(m*t-n*p) = RBC*cos(m*t)*cos(n*p) + RBC*sin(m*t)*sin(n*p)
        # RBS*sin(m*t-n*p) = RBS*sin(m*t)*cos(n*p) - RBS*cos(m*t)*sin(n*p)
        # and similarly for ZBC, ZBS
        surface = {(0, 0): [0.0, 0.0]}  # {(m, n): [R, Z]}
        for name in ["RBC", "RBS", "ZBC", "ZBS"]:
            col = 0 if name[0] == "R" else 1
            for (n, m), x in _get_vmec_boundary(vmec_indata, name).items():
                m_sgn = np.sign(m)
                n_sgn = np.sign(n)
                if m_sgn < 0:
                    warnings.warn(colored("m is negative!", "yellow"))
                m = abs(m)
                n = abs(n)
                terms = []
                if name[-1] == "C":
                    terms.append((m, n, x))
                    if m != 0 and n != 0:
                        terms.append((-m, -n, m_sgn * n_sgn * x))
                else:
                    if m != 0:
                        terms.append((-m, n, m_sgn * x))
                    if n != 0:
                        terms.append((m, -n, -n_sgn * x))
                for mm, nn, xx in terms:
                    surface.setdefault((mm, nn), [0.0, 0.0])[col] += xx
        # DESC input surface array is (k,4) array of [m, n, R, Z] sorted by m then n
        # (radial mode numbers are added below)
        inputs["surface"] = np.array(
            [[m, n, R, Z] for (m, n), (R, Z) in sorted(surface.items())]
        )
        # delete surface modes below threshold magnitude
        inputs["surface"] = np.delete(
            inputs["surface"],
            np.where(np.all(np.abs(inputs["surface"][:, -2:]) < threshold, axis=1))[0],
            axis=0,
        )
        # add radial mode numbers to surface array
        inputs["surface"] = np.pad(inputs["surface"], ((0, 0), (1, 0)), mode="constant")
        # scale pressure profile
        inputs["pressure"][:, 1] *= pres_scale
        # VMEC normalises the current to signgs * CURTOR (profil1d.f), and the
        # orientation of the boundary sets signgs
        signgs = _vmec_signgs(inputs["surface"])
        if not iota_flag:
            if ctype.lower() == "power_series":
                # integrate current derivative profile wrt s=rho^2
                inputs["current"] = np.pad(
                    np.vstack(
                        (
                            inputs["current"][:, 0] + 2,
                            inputs["current"][:, 1] * 2 / (inputs["current"][:, 0] + 2),
                        )
                    ).T,
                    ((1, 0), (0, 0)),
                )
            # scale current profile
            if curr_tor is not None:
                inputs["current"][:, 1] *= (
                    signgs * curr_tor / (np.sum(inputs["current"][:, 1]) or 1)
                )
        # delete unused profile
        if iota_flag:
            del inputs["current"]
        else:
            del inputs["iota"]

        if (
            ptype != "power_series"
            or (itype if iota_flag else ctype.removesuffix("_i")) != "power_series"
            or bloat != 1.0
            or spres_ped < 1.0
        ):
            from desc.vmec_utils import vmec_profile

            profile_data = {
                "pmass_type": ptype,
                "piota_type": itype,
                "pcurr_type": ctype,
                "bloat": bloat,
                "am": AM,
                "ai": AI,
                "ac": AC,
            }
            for name in ("AM", "AI", "AC"):
                for suffix in ("_AUX_S", "_AUX_F"):
                    values = vmec_indata.get(name + suffix, [])
                    values = values if isinstance(values, list) else [values]
                    # the knots are contiguous, f90nml pads to their first index
                    profile_data[(name + suffix).lower()] = [
                        v for v in values if v is not None
                    ]
            s_knots = 0.5 * (1 - np.cos(np.pi * np.arange(401) / 400))
            pressure = pres_scale * vmec_profile(profile_data, "pmass", s_knots)
            # VMEC holds the pressure at its pedestal value outside SPRES_PED
            pressure = np.where(
                s_knots > spres_ped, np.interp(spres_ped, s_knots, pressure), pressure
            )
            inputs["spline"] = {"rho": np.sqrt(s_knots), "pressure": pressure}
            if iota_flag:
                inputs["spline"]["iota"] = vmec_profile(profile_data, "piota", s_knots)
            else:
                current = vmec_profile(profile_data, "pcurr", s_knots)
                curtor = curr_tor or 0.0
                # VMEC drops the current when the profile vanishes at the edge
                scale = (
                    curtor / current[-1]
                    if abs(current[-1]) > abs(np.finfo(float).eps * curtor)
                    else 0.0
                )
                inputs["spline"]["current"] = signgs * scale * current

        inputs_arr = [inputs]

        return inputs_arr


def get_parser():
    """Get parser for command line arguments.

    Returns
    -------
    parser : argparse object
        argument parser

    """
    # NOTE: this has to be outside the class to work with autodoc
    parser = argparse.ArgumentParser(
        prog="desc",
        allow_abbrev=True,
        description="DESC computes equilibria by solving the force balance equations. "
        + "It can also be used for perturbation analysis and sensitivity studies "
        + "to see how the equilibria change as input parameters are varied.",
    )
    parser.add_argument("input_file", nargs="*", help="Path to input file")
    parser.add_argument(
        "-o",
        "--output",
        metavar="output_file",
        help="Path to output file. If not specified, defaults to <input_name>.output",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="count",
        default=0,
        help="Plot results after solver finishes. "
        + "Give once to show only final solution, "
        + "twice (eg -pp) to plot both initial and final, "
        + "and three times (-ppp) to show all iterations.",
    )
    parser.add_argument(
        "--guess",
        metavar="path",
        default=None,
        help="Path to DESC or VMEC equilibrium for initial guess.",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        action="store_true",
        help="Use GPU if available. If more than one are available, selects the "
        + "GPU with most available memory. ",
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Use numpy backend.Performance will be much slower, "
        + "and autodiff won't work but may be useful for debugging.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Display version number and exit."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not display any progress information.",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Display detailed progress information. "
        + "Once to include timing, twice to also show individual iterations.",
    )
    return parser


def _get_vmec_vector(indata, name, default=None):
    """Get a 1D array from a VMEC namelist, indexed starting from 0.

    Parameters
    ----------
    indata : f90nml.Namelist
        INDATA namelist read with f90nml, using ``default_start_index=0``.
    name : str
        Name of the array, case insensitive.
    default : ndarray, optional
        Value to return if the array is not in the namelist.

    Returns
    -------
    x : ndarray
        Values of the array, where x[i] is the namelist element NAME(i).
        Elements that are not specified are set to zero.

    """
    if name not in indata:
        return default
    values = indata[name]
    if not isinstance(values, list):
        values = [values]
    start = indata.start_index.get(name.lower(), [0])[0]
    if start is None:
        # whole array, ie X(:) = ..., which f90nml places starting from index 1
        # (padded with None) rather than from the VMEC lower bound of 0
        if values[0] is None:
            values = values[1:]
        start = 0
    x = np.zeros(start + len(values))
    # f90nml fills unspecified elements with None
    x[start:] = [0.0 if v is None else v for v in values]
    return x


def _get_vmec_boundary(indata, name):
    """Get the boundary Fourier coefficients RBC, RBS, ZBC or ZBS from a VMEC namelist.

    Parameters
    ----------
    indata : f90nml.Namelist
        INDATA namelist read with f90nml.
    name : str
        Name of the boundary array, case insensitive.

    Returns
    -------
    coeffs : dict
        Dictionary of {(n, m): value} for each element NAME(n,m) that is specified.

    """
    if name not in indata:
        return {}
    values = indata[name]
    start = indata.start_index.get(name.lower())
    if (
        start is None
        or len(start) != 2
        or None in start
        or not all(isinstance(row, list) or row is None for row in values)
    ):
        raise ValueError(
            colored(
                f"Cannot read VMEC boundary {name}, each coefficient must be given "
                + f"with explicit (n,m) indices, ie {name}(n,m) = value",
                "red",
            )
        )
    n0, m0 = start
    coeffs = {}
    # f90nml stores the Fortran array X(n,m) as (possibly ragged) nested lists
    # X[m - m0][n - n0], where elements that are not specified are None
    for i, row in enumerate(values):
        for j, x in enumerate(row or []):
            if x is not None:
                coeffs[(n0 + j, m0 + i)] = float(x)
    return coeffs


def _warn_duplicate_vmec_inputs(namelist):
    """Warn if any input in a VMEC namelist string is assigned more than once.

    f90nml (like VMEC) silently uses the last value assigned to each element.
    """
    already_read_names = set()
    warned_names = set()
    for line in namelist.splitlines():
        # remove strings, then comments, so only assignments are left
        line = re.sub(r"'[^']*'|\"[^\"]*\"", "''", line).split("!")[0]
        for match in re.finditer(r"([A-Za-z]\w*)\s*(?:\(([^)]*)\))?\s*=", line):
            input_name = match.group(1).upper()
            if match.group(2) is not None:
                # normalize indices so that ie RBC(000, 001) is the same as RBC(0,1)
                idx = [i.strip() for i in match.group(2).split(",")]
                idx = [str(int(i)) if re.fullmatch(r"[-+]?\d+", i) else i for i in idx]
                input_name += "(" + ",".join(idx) + ")"
            if input_name in already_read_names and input_name not in warned_names:
                warnings.warn(
                    colored(
                        f"Detected multiple inputs for {input_name}! "
                        + "DESC will default to reading the last one.",
                        "yellow",
                    )
                )
                warned_names.add(input_name)
            already_read_names.add(input_name)
