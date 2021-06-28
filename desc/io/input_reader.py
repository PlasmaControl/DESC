import argparse
import pathlib
import warnings
import os
import re
import numpy as np
from datetime import datetime
from termcolor import colored
from desc import set_device


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

    def parse_inputs(self, fname=None):
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
            "pres_ratio": np.atleast_1d(1.0),
            "bdry_ratio": np.atleast_1d(1.0),
            "pert_order": np.atleast_1d(1),
            "ftol": np.atleast_1d(1e-2),
            "xtol": np.atleast_1d(1e-6),
            "gtol": np.atleast_1d(1e-6),
            "nfev": np.atleast_1d(None),
            "objective": "force",
            "optimizer": "lsq-exact",
            "spectral_indexing": "fringe",
            "node_pattern": "jacobi",
            "bdry_mode": "lcfs",
            "profiles": np.atleast_2d((0, 0.0, 0.0)),
            "boundary": np.atleast_2d((0, 0, 0, 0.0, 0.0)),
            "axis": np.atleast_2d((0, 0.0, 0.0)),
        }

        inputs["output_path"] = self.output_path

        if self.args is not None and self.args.quiet:
            inputs["verbose"] = 0
        elif self.args is not None:
            inputs["verbose"] = self.args.verbose
        else:
            inputs["verbose"] = 1

        file = open(fname, "r")
        num_form = r"[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?"

        for line in file:

            # check if VMEC input file format
            isVMEC = re.search(r"&INDATA", line)
            if isVMEC:
                print("Converting VMEC input to DESC input")
                path = self.input_path + "_desc"
                self.vmec_to_desc_input(self.input_path, path)
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
            flag = False

            # global parameters
            match = re.search(r"sym", argument, re.IGNORECASE)
            if match:
                inputs["sym"] = int(numbers[0])
                flag = True
            match = re.search(r"NFP", argument, re.IGNORECASE)
            if match:
                inputs["NFP"] = numbers[0]
                if len(numbers) > 1:
                    inputs["NFP"] /= numbers[1]
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
                inputs["nfev"] = np.array([None if i == 0 else int(i) for i in numbers])
                flag = True

            # solver methods
            match = re.search(r"objective", argument, re.IGNORECASE)
            if match:
                inputs["objective"] = words[0]
                flag = True
            match = re.search(r"optimizer", argument, re.IGNORECASE)
            if match:
                inputs["optimizer"] = words[0]
                flag = True
            match = re.search(r"spectral_indexing", argument, re.IGNORECASE)
            if match:
                inputs["spectral_indexing"] = words[0]
                flag = True
            match = re.search(r"node_pattern", argument, re.IGNORECASE)
            if match:
                inputs["node_pattern"] = words[0]
                flag = True
            match = re.search(r"bdry_mode", argument, re.IGNORECASE)
            if match:
                inputs["bdry_mode"] = words[0]
                flag = True
                # TODO: set bdry_mode automatically based on bdry coeffs

            # coefficient indicies
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
                p_l = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
                prof_idx = np.where(inputs["profiles"][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs["profiles"].shape[0])
                    inputs["profiles"] = np.pad(
                        inputs["profiles"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["profiles"][prof_idx[0], 0] = l
                inputs["profiles"][prof_idx[0], 1] = p_l
                flag = True
            match = re.search(r"\si\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                i_l = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][0]
                prof_idx = np.where(inputs["profiles"][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs["profiles"].shape[0])
                    inputs["profiles"] = np.pad(
                        inputs["profiles"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["profiles"][prof_idx[0], 0] = l
                inputs["profiles"][prof_idx[0], 2] = i_l
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
                    (inputs["boundary"][:, :3] == [l, m, n]).all(axis=1)
                )[0]
                if bdry_idx.size == 0:
                    bdry_idx = np.atleast_1d(inputs["boundary"].shape[0])
                    inputs["boundary"] = np.pad(
                        inputs["boundary"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["boundary"][bdry_idx[0], 0] = l
                    inputs["boundary"][bdry_idx[0], 1] = m
                    inputs["boundary"][bdry_idx[0], 2] = n
                inputs["boundary"][bdry_idx[0], 3] = R1
                flag = True
            match = re.search(r"Z1\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                Z1 = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ][1]
                bdry_idx = np.where(
                    (inputs["boundary"][:, :3] == [l, m, n]).all(axis=1)
                )[0]
                if bdry_idx.size == 0:
                    bdry_idx = np.atleast_1d(inputs["boundary"].shape[0])
                    inputs["boundary"] = np.pad(
                        inputs["boundary"], ((0, 1), (0, 0)), mode="constant"
                    )
                    inputs["boundary"][bdry_idx[0], 0] = l
                    inputs["boundary"][bdry_idx[0], 1] = m
                    inputs["boundary"][bdry_idx[0], 2] = n
                inputs["boundary"][bdry_idx[0], 4] = Z1
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
                raise IOError(
                    colored(
                        "The following line is not a valid input:\n" + command, "red"
                    )
                )

        # error handling
        if np.any(inputs["M"] == 0):
            raise IOError(colored("M_pol is not assigned", "red"))
        if np.sum(inputs["boundary"]) == 0:
            raise IOError(colored("Fixed-boundary surface is not assigned", "red"))
        arrs = [
            "L",
            "M",
            "N",
            "L_grid",
            "M_grid",
            "N_grid",
            "pres_ratio",
            "bdry_ratio",
            "pert_order",
            "ftol",
            "xtol",
            "gtol",
            "nfev",
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
            inputs["M_grid"] = np.rint(1.5 * inputs["M"]).astype(int)
        if np.sum(inputs["N_grid"]) == 0:
            inputs["N_grid"] = np.rint(1.5 * inputs["N"]).astype(int)
        if np.sum(inputs["axis"]) == 0:
            axis_idx = np.where(inputs["boundary"][:, 1] == 0)[0]
            inputs["axis"] = inputs["boundary"][axis_idx, 2:]
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
            inputs_ii["profiles"][:, 1] *= inputs_ii["pres_ratio"]
            # apply boundary ratio
            bdry_factor = np.where(
                inputs_ii["boundary"][:, 2] != 0, inputs_ii["bdry_ratio"], 1
            )
            inputs_ii["boundary"][:, 3] *= bdry_factor
            inputs_ii["boundary"][:, 4] *= bdry_factor
            inputs_list.append(inputs_ii)

        return inputs_list

    def write_desc_input(self, filename, inputs=None):
        """Generate a DESC input file from a dictionary of parameters.

        Parameters
        ----------
        filename : str or path-like
            name of the file to create
        inputs : dict
            dictionary of input parameters

        """
        # default to use self.inputs
        if inputs is None:
            inputs = self.inputs

        f = open(filename, "w+")

        f.write("# global parameters \n")
        f.write("sym = {} \n".format(inputs[0]["sym"]))
        f.write("NFP = {} \n".format(inputs[0]["NFP"]))
        f.write("Psi = {} \n".format(inputs[0]["Psi"]))

        f.write("\n# spectral resolution \n")
        for key, val in {
            "L_rad": "L",
            "M_pol": "M",
            "N_tor": "N",
            "L_grid": "L_grid",
            "M_grid": "M_grid",
            "N_grid": "N_grid",
        }.items():
            f.write(
                key + " = {} \n".format(", ".join([str(inp[val]) for inp in inputs]))
            )

        f.write("\n# continuation parameters \n")
        for key in ["bdry_ratio", "pres_ratio", "pert_order"]:
            f.write(
                key + " = {} \n".format(", ".join([str(inp[key]) for inp in inputs]))
            )

        f.write("\n# solver tolerances \n")
        for key in ["ftol", "xtol", "gtol", "nfev"]:
            f.write(
                key
                + " = {} \n".format(
                    ", ".join(
                        [
                            str(inp[key]) if inp[key] is not None else str(0)
                            for inp in inputs
                        ]
                    )
                )
            )

        f.write("\n# solver methods \n")
        f.write("optimizer = {} \n".format(inputs[0]["optimizer"]))
        f.write("objective = {} \n".format(inputs[0]["objective"]))
        f.write("bdry_mode = {} \n".format(inputs[0]["bdry_mode"]))
        f.write("spectral_indexing = {} \n".format(inputs[0]["spectral_indexing"]))
        f.write("node_pattern = {} \n".format(inputs[0]["node_pattern"]))

        f.write("\n# pressure and rotational transform profiles \n")
        for (l, p, i) in inputs[0]["profiles"]:
            f.write("l: {:3d}\tp = {:16.8E}\ti = {:16.8E}\n".format(int(l), p, i))

        f.write("\n# fixed-boundary surface shape \n")
        for (l, m, n, R1, Z1) in inputs[0]["boundary"]:
            f.write(
                "l: {:3d}\tm: {:3d}\tn: {:3d}\tR1 = {:16.8E}\tZ1 = {:16.8E}\n".format(
                    int(l), int(m), int(n), R1, Z1
                )
            )

        f.write("\n# magnetic axis initial guess \n")
        for (n, R0, Z0) in inputs[0]["axis"]:
            f.write("n: {:3d}\tR0 = {:16.8E}\tZ0 = {:16.8E}\n".format(int(n), R0, Z0))

        f.close()

    def vmec_to_desc_input(self, vmec_fname, desc_fname):
        """Convert a VMEC input file to an equivalent DESC input file.

        Parameters
        ----------
        vmec_fname : str or path-like
            filename of VMEC input file
        desc_fname : str or path-like
            filename of DESC input file. If it already exists it is overwritten.

        """
        # file objects
        vmec_file = open(vmec_fname, "r")
        desc_file = open(desc_fname, "w")

        desc_file.seek(0)
        now = datetime.now()
        date = now.strftime("%m/%d/%Y")
        time = now.strftime("%H:%M:%S")
        desc_file.write(
            "# This DESC input file was auto generated from the VMEC input file\n"
            + "# {}\n# on {} at {}.\n\n".format(vmec_fname, date, time)
        )

        num_form = r"[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?"
        Ntor = 99

        pres_scale = 1.0
        p_l = np.array([0.0])
        i_l = np.array([0.0])
        axis = np.array([[0, 0, 0.0]])
        bdry = np.array([[0, 0, 0.0, 0.0]])

        for line in vmec_file:
            comment = line.find("!")
            command = (line.strip() + " ")[0:comment]

            # global parameters
            if re.search(r"LRFP\s*=\s*T", command, re.IGNORECASE):
                warnings.warn(
                    colored("Using poloidal flux instead of toroidal flux!", "yellow")
                )
            match = re.search("LASYM\s*=\s*[TF]", command, re.IGNORECASE)
            if match:
                if re.search(r"T", match.group(0), re.IGNORECASE):
                    desc_file.write("sym = 0\n")
                else:
                    desc_file.write("sym = 1\n")
            match = re.search(r"NFP\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                desc_file.write("NFP = {:3d}\n".format(numbers[0]))
            match = re.search(r"PHIEDGE\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                desc_file.write("Psi = {:16.8E}\n".format(numbers[0]))
            match = re.search(r"MPOL\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                desc_file.write("M_pol = {:3d}\n".format(numbers[0]))
            match = re.search(r"NTOR\s*=\s*" + num_form, command, re.IGNORECASE)
            if match:
                numbers = [
                    int(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                desc_file.write("N_tor = {:3d}\n".format(numbers[0]))
                Ntor = numbers[0]

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
                    warnings.warn(colored("GAMMA is not 0.0", "yellow"))
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
            match = re.search(r"AM\s*=(\s*" + num_form + ")*", command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    l = 2 * k
                    if p_l.size < l + 1:
                        p_l = np.pad(p_l, (0, l + 1 - p_l.size), mode="constant")
                    p_l[l] = numbers[k]

            # rotational transform
            match = re.search(
                r"NCURR\s*=(\s*" + num_form + ")*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                if numbers[0] != 0:
                    warnings.warn(colored("Not using rotational transform!", "yellow"))
            if re.search(r"\bPIOTA_TYPE\b", command, re.IGNORECASE):
                if not re.search(r"\bpower_series\b", command, re.IGNORECASE):
                    warnings.warn(colored("Iota is not a power series!", "yellow"))
            match = re.search(r"AI\s*=(\s*" + num_form + ")*", command, re.IGNORECASE)
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    l = 2 * k
                    if i_l.size < l + 1:
                        i_l = np.pad(i_l, (0, l + 1 - i_l.size), mode="constant")
                    i_l[l] = numbers[k]

            # magnetic axis
            match = re.search(
                r"RAXIS\s*=(\s*" + num_form + ")*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    if k > Ntor:
                        l = -k + Ntor + 1
                    else:
                        l = k
                    idx = np.where(axis[:, 0] == l)[0]
                    if np.size(idx):
                        axis[idx[0], 1] += numbers[k]
                    else:
                        axis = np.pad(axis, ((0, 1), (0, 0)), mode="constant")
                        axis[-1, :] = np.array([l, numbers[k], 0.0])
            match = re.search(
                r"ZAXIS\s*=(\s*" + num_form + ")*", command, re.IGNORECASE
            )
            if match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, match.group(0))
                    if re.search(r"\d", x)
                ]
                for k in range(len(numbers)):
                    if k > Ntor:
                        l = k - Ntor - 1
                    else:
                        l = -k
                    idx = np.where(axis[:, 0] == l)[0]
                    if np.size(idx) > 0:
                        axis[idx[0], 2] += numbers[k]
                    else:
                        axis = np.pad(axis, ((0, 1), (0, 0)), mode="constant")
                        axis[-1, :] = np.array([l, 0.0, numbers[k]])

            # boundary shape
            # RBS*sin(m*t-n*p) = RBS*sin(m*t)*cos(n*p) - RBS*cos(m*t)*sin(n*p)
            match = re.search(
                r"RBS\(\s*"
                + num_form
                + "\s*,\s*"
                + num_form
                + "\s*\)\s*=\s*"
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
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        bdry[m_idx[idx[0]], 2] += m_sgn * RBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                        bdry[-1, :] = np.array([-m, n, m_sgn * RBS, 0.0])
                if n != 0:  # -RBS*cos(m*t)*sin(n*p)
                    m_idx = np.where(bdry[:, 0] == m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        bdry[m_idx[idx[0]], 2] += -n_sgn * RBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                        bdry[-1, :] = np.array([m, -n, -n_sgn * RBS, 0.0])
            # RBC*cos(m*t-n*p) = RBC*cos(m*t)*cos(n*p) + RBC*sin(m*t)*sin(n*p)
            match = re.search(
                r"RBC\(\s*"
                + num_form
                + "\s*,\s*"
                + num_form
                + "\s*\)\s*=\s*"
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
                # RBC*cos(m*t)*cos(n*p)
                m_idx = np.where(bdry[:, 0] == m)[0]
                n_idx = np.where(bdry[:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx):
                    bdry[m_idx[idx[0]], 2] += RBC
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                    bdry[-1, :] = np.array([m, n, RBC, 0.0])
                if m != 0 and n != 0:  # RBC*sin(m*t)*sin(n*p)
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        bdry[m_idx[idx[0]], 2] += m_sgn * n_sgn * RBC
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                        bdry[-1, :] = np.array([-m, -n, m_sgn * n_sgn * RBC, 0.0])
            # ZBS*sin(m*t-n*p) = ZBS*sin(m*t)*cos(n*p) - ZBS*cos(m*t)*sin(n*p)
            match = re.search(
                r"ZBS\(\s*"
                + num_form
                + "\s*,\s*"
                + num_form
                + "\s*\)\s*=\s*"
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
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        bdry[m_idx[idx[0]], 3] += m_sgn * ZBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                        bdry[-1, :] = np.array([-m, n, 0.0, m_sgn * ZBS])
                if n != 0:  # -ZBS*cos(m*t)*sin(n*p)
                    m_idx = np.where(bdry[:, 0] == m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        bdry[m_idx[idx[0]], 3] += -n_sgn * ZBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                        bdry[-1, :] = np.array([m, -n, 0.0, -n_sgn * ZBS])
            # ZBC*cos(m*t-n*p) = ZBC*cos(m*t)*cos(n*p) + ZBC*sin(m*t)*sin(n*p)
            match = re.search(
                r"ZBC\(\s*"
                + num_form
                + "\s*,\s*"
                + num_form
                + "\s*\)\s*=\s*"
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
                # ZBC*cos(m*t)*cos(n*p)
                m_idx = np.where(bdry[:, 0] == m)[0]
                n_idx = np.where(bdry[:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx):
                    bdry[m_idx[idx[0]], 3] += ZBC
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                    bdry[-1, :] = np.array([m, n, 0.0, ZBC])
                if m != 0 and n != 0:  # ZBC*sin(m*t)*sin(n*p)
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx):
                        bdry[m_idx[idx[0]], 3] += m_sgn * n_sgn * ZBC
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode="constant")
                        bdry[-1, :] = np.array([-m, -n, 0.0, m_sgn * n_sgn * ZBC])

            # catch multi-line inputs
            match = re.search(r"=", command)
            if not match:
                numbers = [
                    float(x)
                    for x in re.findall(num_form, command)
                    if re.search(r"\d", x)
                ]
                if len(numbers) > 0:
                    raise IOError(
                        colored("Cannot handle multi-line VMEC inputs!", "red")
                    )

        p_l *= pres_scale
        desc_file.write("\n")
        desc_file.write("# pressure and rotational transform profiles\n")
        for k in range(max(p_l.size, i_l.size)):
            if k >= p_l.size:
                desc_file.write(
                    "l: {:3d}  p = {:16.8E}  i = {:16.8E}\n".format(k, 0.0, i_l[k])
                )
            elif k >= i_l.size:
                desc_file.write(
                    "l: {:3d}  p = {:16.8E}  i = {:16.8E}\n".format(k, p_l[k], 0.0)
                )
            else:
                desc_file.write(
                    "l: {:3d}  p = {:16.8E}  i = {:16.8E}\n".format(k, p_l[k], i_l[k])
                )

        desc_file.write("\n")
        desc_file.write("# magnetic axis initial guess\n")
        for k in range(np.shape(axis)[0]):
            desc_file.write(
                "n: {:3d}  R0 = {:16.8E}  Z0 = {:16.8E}\n".format(
                    int(axis[k, 0]), axis[k, 1], axis[k, 2]
                )
            )

        desc_file.write("\n")
        desc_file.write("# fixed-boundary surface shape\n")
        for k in range(np.shape(bdry)[0]):
            desc_file.write(
                "m: {:3d}  n: {:3d}  R1 = {:16.8E}  Z1 = {:16.8E}\n".format(
                    int(bdry[k, 0]), int(bdry[k, 1]), bdry[k, 2], bdry[k, 3]
                )
            )

        desc_file.truncate()

        # close files
        vmec_file.close()
        desc_file.close()


# NOTE: this has to be outside the class to work with autodoc


def get_parser():
    """Get parser for command line arguments.

    Returns
    -------
    parser : argparse object
        argument parser

    """
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
        "--vmec",
        metavar="vmec_path",
        default=None,
        help="Path to VMEC data for initial guess.",
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
