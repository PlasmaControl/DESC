These are inputs that are adjusted to match the VMEC inputs in ../../../iota/edu-vmec/input-current.

Note that:
* sym is set to 0 (important for grid bug)
* NFP is set to 1 (important for grid bug)

## fail list
On some inputs the solver just exits early saying no nested surfaces.
* Heliotron - flux surfaces no longer nested
* Estell, QAS, Wistell-A - stops at computing df with message "killed"
This can either mean
* the input current profile does not lend itself to nested surfaces
