import versioneer
from setuptools import setup, find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, "requirements_optional.txt"), encoding="utf-8") as f:
    requirements_optional = f.read().splitlines()

with open(os.path.join(here, "docs/requirements.txt"), encoding="utf-8") as f:
    docs_requirements = f.read().splitlines()

with open(os.path.join(here, "tests/requirements.txt"), encoding="utf-8") as f:
    test_requirements = f.read().splitlines()

setup(
    name="desc-opt",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Computes, analyzes and optimizes 3D MHD equilibria for stellarators and tokamaks",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/PlasmaControl/DESC/",
    author="Daniel Dudt, Rory Conlin, Dario Panici, Egemen Kolemen",
    author_email="PlasmaControl@princeton.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="stellarator tokamak equilibrium perturbation mhd "
    + "magnetohydrodynamics stability confinement plasma physics "
    + "optimization design fusion",
    packages=find_packages(exclude=["docs", "tests", "local", "report"]),
    install_requires=requirements,
    extras_require={
        "docs": docs_requirements,
        "test": test_requirements,
        "opt": requirements_optional,
    },
    python_requires="~=3.6",
    entry_points={"console_scripts": ["desc=desc.__main__:main"]},
    project_urls={
        "Issues Tracker": "https://github.com/PlasmaControl/DESC/issues",
        "Contributing": "https://github.com/PlasmaControl/DESC/blob/master/CONTRIBUTING.rst",
        "Source Code": "https://github.com/PlasmaControl/DESC/",
        "Documentation": "https://desc-docs.readthedocs.io/",
    },
)
