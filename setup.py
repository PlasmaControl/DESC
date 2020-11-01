from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))


def read(rel_path):
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, 'docs/requirements.txt'), encoding='utf-8') as f:
    docs_requirements = f.read().splitlines()

with open(os.path.join(here, 'tests/requirements.txt'), encoding='utf-8') as f:
    test_requirements = f.read().splitlines()

setup(name='desc',
      version=get_version('desc/__init__.py'),
      description='Computes, analyzes and optimizes 3D MHD equilibria for stellarators and tokamaks',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='https://github.com/ddudt/DESC/',
      author='Daniel Dudt, Rory Conlin, Dario Panici, Egemen Kolemen',
      author_email='ddudt@princeton.edu',
      license='MIT',
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Environment :: GPU :: NVIDIA CUDA',
                   'Intended Audience :: Manufacturing',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Scientific/Engineering :: Visualization'],
      keywords='stellarator tokamak equilibrium perturbation mhd '
      + 'magnetohydrodynamics stability confinement plasma physics '
      + 'optimization design fusion',
      packages=find_packages(exclude=['docs', 'tests', 'local', ]),
      install_requires=requirements,
      extras_require={'docs': docs_requirements,
                      'test': test_requirements},
      python_requires='~=3.6',
      entry_points={'console_scripts': ['desc=desc.__main__:main']},
      project_urls={'Issues Tracker': 'https://github.com/ddudt/DESC/issues',
                    'Contributing': 'https://github.com/ddudt/DESC/blob/master/CONTRIBUTING.rst',
                    'Source Code': 'https://github.com/ddudt/DESC/',
                    'Documentation': 'https://desc-docs.readthedocs.io/'},
      )
