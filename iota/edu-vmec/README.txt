https://princetonuniversity.github.io/STELLOPT/Tutorial%20VMEC%20Input%20Namelist.html
../../../../educational_VMEC/build/bin/xvmec input.NAME

change NCURR = 0 to 1
change CURTOR = 0 to 1E+4 to 1E+6
    too large and all plots will look like a generic quadratic curve because iota ~ current
add PCURR_TYPE = "power_series"
add AC = pick numbers
change RAXIS to RAXIS_CC
change ZAXIS to ZAXIS_CC
remove NITER when NITER_ARRAY is already specified
