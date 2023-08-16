"""Try vmec utility."""
from database_converter_vmec import vmec_to_csv

file_vmec = "../Python/maxJqi/alan_qis/N1/wout_QI_nfp1.nc"
vmec_to_csv(
    file_vmec,
    current=True,
    name=None,
    provenance=None,
    description=None,
    inputfilename=None,
    initialization_method="surface",
)
