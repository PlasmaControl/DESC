"""try using the device and concept table creation util."""
from desc.io.equilibrium_io import device_or_concept_to_csv

device_or_concept_to_csv(
    name="Wendelstein 7-X",
    NFP=5,
    description="Stellarator optimized for particle confinement"
    + " through isodynamicity, largest in the world.",
    stell_sym=True,
    deviceid="W7-X",
    device_class="QI",
)

device_or_concept_to_csv(
    name="Advanced Toroidal Facility",
    NFP=12,
    description="Torsatron-type stellarator, operated in ORNL in 80s.",
    stell_sym=True,
    deviceid="ATF",
)

device_or_concept_to_csv(
    name="National Compact Stellarator Experiment",
    NFP=3,
    description="Low aspect ratio QA stellarator partially"
    + " constructed at PPPL in early 2000s.",
    stell_sym=True,
    deviceid="NCSX",
    device_class="QAS",
)
