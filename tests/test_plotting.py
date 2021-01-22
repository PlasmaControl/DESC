import pytest
import unittest

from desc.plotting import Plot, plot_logo


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.names = [
            "B",
            "|B|",
            "B^zeta",
            "B_zeta",
            "B_r",
            "B^zeta_r",
            "B_zeta_r",
            "B**2",
            "B_r**2",
            "B^zeta**2",
            "B_zeta**2",
            "B^zeta_r**2",
            "B_zeta_r**2",
        ]
        self.bases = ["B", "|B|", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
        self.sups = ["", "", "zeta", "", "", "zeta", "", "", "", "zeta", "", "zeta", ""]
        self.subs = ["", "", "", "zeta", "", "", "zeta", "", "", "", "zeta", "", "zeta"]
        self.ds = ["", "", "", "", "r", "r", "r", "", "r", "", "", "r", "r"]
        self.pows = ["", "", "", "", "", "", "", "2", "2", "2", "2", "2", "2"]
        self.name_dicts = []
        self.plot = Plot()
        for name in self.names:
            self.name_dicts.append(self.plot.format_name(name))

    def test_name_dict(self):
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["base"] == self.bases[i]
                    for i in range(len(self.names))
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["sups"] == self.sups[i]
                    for i in range(len(self.names))
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["subs"] == self.subs[i]
                    for i in range(len(self.names))
                ]
            )
        )
        self.assertTrue(
            all([self.name_dicts[i]["d"] == self.ds[i] for i in range(len(self.names))])
        )
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["power"] == self.pows[i]
                    for i in range(len(self.names))
                ]
            )
        )

    def test_name_label(self):
        labels = [self.plot.name_label(nd) for nd in self.name_dicts]
        print(labels)
        self.assertTrue(all([label[0] == "$" and label[-1] == "$" for label in labels]))
        self.assertTrue(
            all(
                [
                    "/dr" in labels[i]
                    for i in range(len(labels))
                    if self.name_dicts[i]["d"] != ""
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    "^{" not in labels[i]
                    for i in range(len(labels))
                    if self.name_dicts[i]["sups"] == ""
                    and self.name_dicts[i]["power"] == ""
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    "_{" not in labels[i]
                    for i in range(len(labels))
                    if self.name_dicts[i]["subs"] == ""
                ]
            )
        )


@pytest.mark.mpl_image_compare
def test_1d_p(plot_eq):
    fig, ax = Plot().plot_1d(plot_eq, "p")
    return fig


@pytest.mark.mpl_image_compare
def test_1d_iota(plot_eq):
    fig, ax = Plot().plot_1d(plot_eq, "iota")
    return fig


@pytest.mark.mpl_image_compare
def test_1d_psi(plot_eq):
    fig, ax = Plot().plot_1d(plot_eq, "psi")
    return fig


@pytest.mark.mpl_image_compare
def test_2d_B(plot_eq):
    fig, ax = Plot().plot_2d(plot_eq, "|B|")
    return fig


@pytest.mark.mpl_image_compare
def test_2d_g(plot_eq):
    fig, ax = Plot().plot_2d(plot_eq, "g")
    return fig


@pytest.mark.mpl_image_compare
def test_2d_lambda(plot_eq):
    fig, ax = Plot().plot_2d(plot_eq, "lambda")
    return fig


@pytest.mark.mpl_image_compare
def test_section_J(plot_eq):
    fig, ax = Plot().plot_section(plot_eq, "J^rho")
    return fig


@pytest.mark.mpl_image_compare
def test_section_Z(plot_eq):
    fig, ax = Plot().plot_section(plot_eq, "Z")
    return fig


@pytest.mark.mpl_image_compare
def test_section_R(plot_eq):
    fig, ax = Plot().plot_section(plot_eq, "R")
    return fig


@pytest.mark.mpl_image_compare
def test_section_F(plot_eq):
    fig, ax = Plot().plot_section(plot_eq, "F_rho")
    return fig


@pytest.mark.mpl_image_compare
def test_section_logF(plot_eq):
    fig, ax = Plot().plot_section(plot_eq, "|F|", log=True)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_surfaces(plot_eq):
    fig, ax = Plot().plot_surfaces(plot_eq)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_logo():
    fig, ax = plot_logo()
    return fig
