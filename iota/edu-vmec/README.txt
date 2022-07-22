useful links
https://princetonuniversity.github.io/STELLOPT/Tutorial%20VMEC%20Input%20Namelist.html
https://github.com/jonathanschilling/educational_VMEC/blob/e6df9ac224b2ec8bbd0a5e085489c6d1bb74defa/src/data/vmec_main.f90

run command
../../../../educational_VMEC/build/bin/xvmec input.NAME

Couldn't get heliotron to converge on DESC or VMEC. Suggestions for current profile? I looked it up and it says order of magnitude for Heliotron was kiloamps.

changes
change NCURR = 0 to 1
change CURTOR = 0 to 1E+3 to 1E+6
    too large and all plots will look like a generic quadratic curve because iota ~ current
add PCURR_TYPE = "power_series"
add AC = pick numbers
change RAXIS to RAXIS_CC
change ZAXIS to ZAXIS_CC
remove NITER when NITER_ARRAY is already specified
