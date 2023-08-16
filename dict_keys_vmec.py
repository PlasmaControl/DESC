"""Try vmec utility."""

from simsopt.mhd.vmec import Vmec

file_vmec = "../Python/maxJqi/alan_qis/N1/wout_QI_nfp1.nc"
vmec = Vmec(file_vmec)
temp = {}

print(vmec.wout.__dir__())
print(vmec.wout.lasym)
print(vmec.wout.betatotal)
