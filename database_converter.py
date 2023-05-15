"""Functions to convert DESC or VMEC output files into .csv files for the Datbase."""

from desc.equilibrium import EquilibriaFamily
from desc.io import load


def DESC_hash(eq):  # or take in the data from a DESC eq?
    """Take in a DESC equilibrium and return a unique hash.

    Args:
    ----
        eq (_type_): _description_
    """


def desc_to_csv(
    eq,
    filename=None,
    provenance=None,
    description=None,
    publications=None,
    version=None,
):
    """Save DESC as a csv with relevant information.

    Args
    ----
        eq (Equilibrium): DESC equilibrium to save
        filename: name of csv file to save to

    Returns
    -------
        None
    """
    if isinstance(eq, str):
        eq = load(eq)
    if isinstance(eq, EquilibriaFamily):
        eq = eq[-1]  # just use last equilibrium

    data = {}
    data["desc_run_ID"] = None  # what should this be? how to hash?

    return None


# TODO: other fields?
# provenance could be used to include if the file was based off a
# NAE equilibrium, but maybe if this is supposed to be more
# common we could also include the NAE parameters...

# Aza would ideally like one CSV per input
# can take multiple inputs at a time
# then just output to a few csvs
