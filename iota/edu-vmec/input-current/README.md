These are inputs that are adjusted to match the DESC inputs in ../../../examples/DESC/input-current.

## Fail list:
The inputs that exit early / don't converge are:
* Heliotron

## useful links that attempt to document VMEC
* https://princetonuniversity.github.io/STELLOPT/Tutorial%20VMEC%20Input%20Namelist.html
* https://github.com/jonathanschilling/educational_VMEC/blob/e6df9ac224b2ec8bbd0a5e085489c6d1bb74defa/src/data/vmec_main.f90
* https://github.com/jonathanschilling/educational_VMEC/blob/0999e1e66d0ba8407e587084787cbf0d92c8dab2/src/profile_functions.f

## Changes required for edu-vmec to run normal vmec file
* change NCURR = 0 to 1
* change CURTOR = 0 to some value like 1E+5
    - too large and all plots will look like a generic quadratic curve because iota ~ current
* add PCURR_TYPE = "power_series_I"
* add AC = pick numbers
* change RAXIS to RAXIS_CC
* change ZAXIS to ZAXIS_CS
* remove NITER when NITER_ARRAY is already specified
    - or use NITER_ARRAY instead
* comment out with !
    - LOLDOUT = F
    - LWOUTTXT = F
    - LDIAGNO = T
    - PRECON_TYPE = 'NONE'
    - PREC2D_THRESHOLD = 1.000000E-30

## Note
The QAS input was solved with regular VMEC. So it's input file does not have the changes above.

## edu-vmec run command
../../../../educational_VMEC/build/bin/xvmec input.NAME
