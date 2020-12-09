from desc.input_reader import InputReader
from desc.continuation import solve_eq_continuation

ir = InputReader(cl_args=['examples/DESC/HELIOTRON'])
iterations, timer = solve_eq_continuation(ir.inputs)
