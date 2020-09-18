==========
Inputs
==========

This page explains the input file format.


File Format
************

This is a python code block:

.. code-block:: python
		
    # comment
    filename = 'HELIOTRON'
    in_fname = 'benchmarks/DESC/'+filename+'.input'
    inputs = read_input(in_fname)
    out_fname = inputs['out_fname']
    
    # solve equilibrium
    equil_init,equil = solve_eq_continuation(inputs)