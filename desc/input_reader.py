"""Class for reading and writing DESC and VMEC input files."""

import argparse
import io
import os
import pathlib
import re
import warnings
from datetime import datetime

import numpy as np
from termcolor import colored

from desc import set_device

# shouldn't import anything else from DESC here, since that will initialize JAX
# before we have a chance to parse user inputs to see if they want to use GPU
# instead, do local imports where needed.


class InputReader:
    """Reads command line arguments and parses input files.

    Parameters
    ----------
    cl_args : None, str, or list (Default = None)
        command line arguments to parse

    """

    def __init__(self, cl_args=None):
        """Initialize InputReader instance."""
        self._args = None
        self._inputs = None
        self._input_path = None
        self._output_path = None

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
            "pressure": np.atleast_2d((0, 0.0)),
            "iota": np.atleast_2d((0, 0.0)),
            "current": np.atleast_2d((0, 0.0)),
            "surface": np.atleast_2d((0, 0, 0, 0.0, 0.0)),
            "axis": np.atleast_2d((0, 0.0, 0.0)),
            "xsection": None,
        }

        iota_flag = False
        pres_flag = False
        curr_flag = False
        vac_flag = False
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
                path = self.input_path + "_desc"
                InputReader.vmec_to_desc_input(self.input_path, path)
                print("Generated DESC input file {}:".format(path))
                return self.parse_inputs(path)

            # extract numbers & words
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
            command = (line.strip() + " ")[0:comment]
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
        if not isinstance(filename, io.IOBase):
            f = open(filename, "w+")
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
        f.write("spectral_indexing = {}\n".format(inputs[0]["spectral_indexing"]))

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

        prof_modes = np.zeros((L_profile, 3))
        prof_modes[:, 0] = np.arange(L_profile)
        p1 = copy_coeffs(pres_profile.params, pres_profile.basis.modes, prof_modes)
        p2 = copy_coeffs(profile.params, profile.basis.modes, prof_modes)
        f.write("\n# pressure and rotational transform/current profiles\n")
        for l in range(L_profile):
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
        num_form = r"[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?"

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
            "pressure": np.atleast_2d((0, 0.0)),
            "iota": np.atleast_2d((0, 0.0)),
            "current": np.atleast_2d((0, 0.0)),
            "surface": np.atleast_2d((0, 0, 0.0, 0.0)),
            "axis": np.atleast_2d((0, 0.0, 0.0)),
        }

        iota_flag = True
        pres_scale = 1.0
        curr_tor = None

        # find start of namelist (&INDATA)
        vmeclines = vmec_file.readlines()
        for i, line in enumerate(vmeclines):
            if line.upper().find("&INDATA") != -1:
                start_ind = i
                continue
            elif line.find("&") != -1:  # only care about the INDATA section
                end_ind = i
                break
            elif i == len(vmeclines) - 1:
                end_ind = i
        vmeclines = vmeclines[start_ind + 1 : end_ind]

        # Loop which makes multi-line inputs a single line
        vmeclines_no_multiline = []
        for line in vmeclines:
            comment = line.find("!")
            line = (line.strip() + " ")[0:comment].strip()
            equal_ind = line.find("=")

            # ignore blank lines or leftover comment !'s
            if line.isspace() or line == "" or line == r"!":
                continue
            if equal_ind != -1:
                vmeclines_no_multiline.append(line)
            else:  # is a multi-line input,append the line to the previous
                vmeclines_no_multiline[-1] += " " + line

        # remove duplicate lines
        vmec_no_multiline_no_duplicates = []
        already_read_names = []
        already_read = False

        # it is easiest to find first instances of something, but
        # we want to keep only the last instance of each duplicate
        # as that is the behavior VMEC has,
        # so we will read through the lines reversed
        vmeclines_no_multiline.reverse()

        for line in vmeclines_no_multiline:
            comment = line.find("!")
            line = (line.strip() + " ")[0:comment].strip()
            equal_ind = line.find("=")

            # ignore blank lines or leftover comment !'s
            if line.isspace() or line == "" or line == r"!":
                continue
            if equal_ind != -1:
                # add name of variable to already_read_names
                input_name = line[0:equal_ind].strip()
                if input_name not in already_read_names:
                    already_read_names.append(input_name)
                    vmec_no_multiline_no_duplicates.append(line)
                    already_read = False
                else:
                    warnings.warn(
                        colored(
                            f"Detected multiple inputs for {input_name}! "
                            + "DESC will default to using the last one.",
                            "yellow",
                        )
                    )
                    already_read = True
            # make sure if it is a duplicate line,
            # to skip the multi-line associated with it
            elif not already_read:
                # is a multi-line input,append the line to the previous
                vmec_no_multiline_no_duplicates[-1] += " " + line

        # undo the reverse so the file we write looks as expected
        vmec_no_multiline_no_duplicates.reverse()

        # read the inputs for use in DESC
        for line in vmec_no_multiline_no_duplicates:
            comment = line.find("!")
            command = (line.strip() + " ")[0:comment]
            # global parameters
            if re.search(r"LFREEB\s*=\s*T", command, re.IGNORECASE):
                warnings.warn(
                    colored(
                        "Using free-boundary mode! DESC will run as fixed-boundary.",
                        "yellow",
                    )
                )
            if re.search(r"LRFP\s*=\s*T", command, re.IGNORECASE):
                warnings.warn(
                    colored(
                        "Using poloidal flux instead of toroidal flux! "
                        + " DESC will read as toroidal flux.",
                        "yellow",
                    )
                )
            match = re.search(r"LASYM\s*=\s*[TF]", command, re.IGNORECASE)
            if match:
                if re.search(r"T", match.group(0), re.IGNORECASE):
                    inputs["sym"] = False
                else:
                    inputs["sym"] = True
            match = re.search(r"NFP\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                inputs["NFP"] = numbers[0]
            match = re.search(r"PHIEDGE\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                inputs["Psi"] = numbers[0]
            match = re.search(r"MPOL\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                inputs["M"] = numbers[0] - 1
                inputs["L"] = numbers[0] - 1
                inputs["L_grid"] = 2 * (numbers[0] - 1)
                inputs["M_grid"] = 2 * (numbers[0] - 1)
            match = re.search(r"NTOR\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                inputs["N"] = numbers[0]
                inputs["N_grid"] = 2 * numbers[0]
            match = re.search(
                r"NCURR\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                if numbers[0] != 0:
                    iota_flag = False

            # pressure profile
            match = re.search(r"bPMASS_TYPE\s*=\s*\w*", command, re.IGNORECASE)
            if match:
                if not re.search(r"\bpower_series\b", match.group(0), re.IGNORECASE):
                    warnings.warn(colored("Pressure is not a power series!", "yellow"))
            match = re.search(r"GAMMA\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                if numbers[0] != 0:
                    warnings.warn(
                        colored(
                            "GAMMA is not 0.0, DESC will set this to 0.0.", "yellow"
                        )
                    )
            match = re.search(r"BLOAT\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                if numbers[0] != 1:
                    warnings.warn(colored("BLOAT is not 1.0", "yellow"))
            match = re.search(r"SPRES_PED\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                if numbers[0] != 1:
                    warnings.warn(colored("SPRES_PED is not 1.0", "yellow"))
            match = re.search(r"PRES_SCALE\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                pres_scale = numbers[0]
            match = re.search(
                r"AM\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    l = 2 * k
                    if len(inputs["pressure"]) < k + 1:
                        inputs["pressure"] = np.pad(
                            inputs["pressure"],
                            ((0, k + 1 - len(inputs["pressure"])), (0, 0)),
                            mode="constant",
                        )
                    inputs["pressure"][k, 1] = numbers[k]
                    inputs["pressure"][k, 0] = l

            # rotational transform
            if re.search(r"\bPIOTA_TYPE\b", command, re.IGNORECASE):
                if not re.search(r"\bpower_series\b", command, re.IGNORECASE):
                    warnings.warn(colored("Iota is not a power series!", "yellow"))
            match = re.search(
                r"AI\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    l = 2 * k
                    if len(inputs["iota"]) < k + 1:
                        inputs["iota"] = np.pad(
                            inputs["iota"],
                            ((0, k + 1 - len(inputs["iota"])), (0, 0)),
                            mode="constant",
                        )
                    inputs["iota"][k, 1] = numbers[k]
                    inputs["iota"][k, 0] = l

            # current
            if re.search(r"\bPCURR_TYPE\b", command, re.IGNORECASE):
                if not re.search(r"\bpower_series\b", command, re.IGNORECASE):
                    warnings.warn(colored("Current is not a power series!", "yellow"))
            match = re.search(r"CURTOR\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                curr_tor = numbers[0]
            match = re.search(
                r"AC\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    l = 2 * k
                    if len(inputs["current"]) < k + 1:
                        inputs["current"] = np.pad(
                            inputs["current"],
                            ((0, k + 1 - len(inputs["current"])), (0, 0)),
                            mode="constant",
                        )
                    inputs["current"][k, 1] = numbers[k]
                    inputs["current"][k, 0] = l

            # magnetic axis
            match = re.search(
                r"RAXIS(_CC)?\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    n = k
                    idx = np.where(inputs["axis"][:, 0] == n)[0]
                    if np.size(idx):
                        inputs["axis"][idx[0], 1] = numbers[k]
                    else:
                        inputs["axis"] = np.pad(
                            inputs["axis"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["axis"][-1, :] = np.array([n, numbers[k], 0.0])
            match = re.search(
                r"RAXIS_CS\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                # ignore the n=0 since it should always be zero for sin terms
                for k in range(1, len(numbers)):
                    n = -k
                    idx = np.where(inputs["axis"][:, 0] == n)[0]
                    if np.size(idx):
                        inputs["axis"][idx[0], 1] = -numbers[k]
                    else:
                        inputs["axis"] = np.pad(
                            inputs["axis"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["axis"][-1, :] = np.array([n, -numbers[k], 0.0])
            match = re.search(
                r"ZAXIS(_CS)?\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                # ignore the n=0 since it should always be zero for sin terms
                for k in range(1, len(numbers)):
                    n = -k
                    idx = np.where(inputs["axis"][:, 0] == n)[0]
                    if np.size(idx):
                        inputs["axis"][idx[0], 2] = -numbers[k]
                    else:
                        inputs["axis"] = np.pad(
                            inputs["axis"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["axis"][-1, :] = np.array([n, 0.0, -numbers[k]])
            match = re.search(
                r"ZAXIS_CC\s*=(\s*" + num_form + r"\s*,?)*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    n = k
                    idx = np.where(inputs["axis"][:, 0] == n)[0]
                    if np.size(idx):
                        inputs["axis"][idx[0], 2] = numbers[k]
                    else:
                        inputs["axis"] = np.pad(
                            inputs["axis"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["axis"][-1, :] = np.array([n, 0.0, numbers[k]])

            # boundary shape
            # RBS*sin(m*t-n*p) = RBS*sin(m*t)*cos(n*p) - RBS*cos(m*t)*sin(n*p)
            match = re.search(
                r"RBS\(\s*"
                + num_form
                + r"\s*,\s*"
                + num_form
                + r"\s*\)\s*=\s*"
                + num_form,
                command,
                re.IGNORECASE,
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                n = int(numbers[0])
                m = int(numbers[1])
                m_sgn = np.sign(np.array([m]))[0]
                n_sgn = np.sign(np.array([n]))[0]
                n = abs(n)
                if m_sgn < 0:
                    warnings.warn(colored("m is negative!", "yellow"))
                    m = abs(m)
                RBS = numbers[2]
                if m != 0:  # RBS*sin(m*t)*cos(n*p)
                    m_idx = np.where(inputs["surface"][:, 0] == -m)[0]
                    n_idx = np.where(inputs["surface"][:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        inputs["surface"][m_idx[idx[0]], 2] += m_sgn * RBS
                    else:
                        inputs["surface"] = np.pad(
                            inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["surface"][-1, :] = np.array([-m, n, m_sgn * RBS, 0.0])
                if n != 0:  # -RBS*cos(m*t)*sin(n*p)
                    m_idx = np.where(inputs["surface"][:, 0] == m)[0]
                    n_idx = np.where(inputs["surface"][:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        inputs["surface"][m_idx[idx[0]], 2] += -n_sgn * RBS
                    else:
                        inputs["surface"] = np.pad(
                            inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["surface"][-1, :] = np.array([m, -n, -n_sgn * RBS, 0.0])
            # RBC*cos(m*t-n*p) = RBC*cos(m*t)*cos(n*p) + RBC*sin(m*t)*sin(n*p)
            match = re.search(
                r"RBC\(\s*"
                + num_form
                + r"\s*,\s*"
                + num_form
                + r"\s*\)\s*=\s*"
                + num_form,
                command,
                re.IGNORECASE,
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                n = int(numbers[0])
                m = int(numbers[1])
                m_sgn = np.sign(np.array([m]))[0]
                n_sgn = np.sign(np.array([n]))[0]
                n = abs(n)
                if m_sgn < 0:
                    warnings.warn(colored("m is negative!", "yellow"))
                    m = abs(m)
                RBC = numbers[2]
                # RBC*cos(m*t)*cos(n*p)  # noqa: E800
                m_idx = np.where(inputs["surface"][:, 0] == m)[0]
                n_idx = np.where(inputs["surface"][:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx):
                    inputs["surface"][m_idx[idx[0]], 2] += RBC
                else:
                    inputs["surface"] = np.pad(
                        inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["surface"][-1, :] = np.array([m, n, RBC, 0.0])
                if m != 0 and n != 0:  # RBC*sin(m*t)*sin(n*p)
                    m_idx = np.where(inputs["surface"][:, 0] == -m)[0]
                    n_idx = np.where(inputs["surface"][:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        inputs["surface"][m_idx[idx[0]], 2] += m_sgn * n_sgn * RBC
                    else:
                        inputs["surface"] = np.pad(
                            inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["surface"][-1, :] = np.array(
                            [-m, -n, m_sgn * n_sgn * RBC, 0.0]
                        )
            # ZBS*sin(m*t-n*p) = ZBS*sin(m*t)*cos(n*p) - ZBS*cos(m*t)*sin(n*p)
            match = re.search(
                r"ZBS\(\s*"
                + num_form
                + r"\s*,\s*"
                + num_form
                + r"\s*\)\s*=\s*"
                + num_form,
                command,
                re.IGNORECASE,
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                n = int(numbers[0])
                m = int(numbers[1])
                m_sgn = np.sign(np.array([m]))[0]
                n_sgn = np.sign(np.array([n]))[0]
                n = abs(n)
                if m_sgn < 0:
                    warnings.warn(colored("m is negative!", "yellow"))
                    m = abs(m)
                ZBS = numbers[2]
                if m != 0:  # ZBS*sin(m*t)*cos(n*p)
                    m_idx = np.where(inputs["surface"][:, 0] == -m)[0]
                    n_idx = np.where(inputs["surface"][:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        inputs["surface"][m_idx[idx[0]], 3] += m_sgn * ZBS
                    else:
                        inputs["surface"] = np.pad(
                            inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["surface"][-1, :] = np.array([-m, n, 0.0, m_sgn * ZBS])
                if n != 0:  # -ZBS*cos(m*t)*sin(n*p)
                    m_idx = np.where(inputs["surface"][:, 0] == m)[0]
                    n_idx = np.where(inputs["surface"][:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        inputs["surface"][m_idx[idx[0]], 3] += -n_sgn * ZBS
                    else:
                        inputs["surface"] = np.pad(
                            inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["surface"][-1, :] = np.array([m, -n, 0.0, -n_sgn * ZBS])
            # ZBC*cos(m*t-n*p) = ZBC*cos(m*t)*cos(n*p) + ZBC*sin(m*t)*sin(n*p)
            match = re.search(
                r"ZBC\(\s*"
                + num_form
                + r"\s*,\s*"
                + num_form
                + r"\s*\)\s*=\s*"
                + num_form,
                command,
                re.IGNORECASE,
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                n = int(numbers[0])
                m = int(numbers[1])
                m_sgn = np.sign(np.array([m]))[0]
                n_sgn = np.sign(np.array([n]))[0]
                n = abs(n)
                if m_sgn < 0:
                    warnings.warn(colored("m is negative!", "yellow"))
                    m = abs(m)
                ZBC = numbers[2]
                # ZBC*cos(m*t)*cos(n*p)  # noqa: E800
                m_idx = np.where(inputs["surface"][:, 0] == m)[0]
                n_idx = np.where(inputs["surface"][:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx):
                    inputs["surface"][m_idx[idx[0]], 3] += ZBC
                else:
                    inputs["surface"] = np.pad(
                        inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["surface"][-1, :] = np.array([m, n, 0.0, ZBC])
                if m != 0 and n != 0:  # ZBC*sin(m*t)*sin(n*p)
                    m_idx = np.where(inputs["surface"][:, 0] == -m)[0]
                    n_idx = np.where(inputs["surface"][:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        inputs["surface"][m_idx[idx[0]], 3] += m_sgn * n_sgn * ZBC
                    else:
                        inputs["surface"] = np.pad(
                            inputs["surface"], ((0, 1), (0, 0)), mode="constant"
                        )
                        inputs["surface"][-1, :] = np.array(
                            [-m, -n, 0.0, m_sgn * n_sgn * ZBC]
                        )

            # catch multi-line inputs
            match = re.search(r"=", command)
            if not match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, command)
                    if re.search(r"\d", x)
                ]
                if len(numbers) > 0:
                    raise OSError(
                        colored("Cannot handle multi-line VMEC inputs!", "red")
                    )

        # sort axis array
        inputs["axis"] = inputs["axis"][inputs["axis"][:, 0].argsort()]
        # sort surface array
        inputs["surface"] = inputs["surface"][inputs["surface"][:, 1].argsort()]
        inputs["surface"] = inputs["surface"][
            inputs["surface"][:, 0].argsort(kind="mergesort")
        ]
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
        # integrate current profile wrt s=rho^2
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
            inputs["current"][:, 1] *= curr_tor / (np.sum(inputs["current"][:, 1]) or 1)
        # delete unused profile
        if iota_flag:
            del inputs["current"]
        else:
            del inputs["iota"]

        vmec_file.close()

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
