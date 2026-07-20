# Copyright (C) 2024 Kaya Unalmis
# SPDX-License-Identifier: LGPL-3.0

"""Setup/build/install script for adv-jax-math."""

import os

import versioneer
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="adv-jax-math",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=("Advanced automatic-differentiation and batching utilities for JAX"),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/unalmis/adv-jax-math",
    author="Kaya Unalmis",
    author_email="kunalmis@stanford.edu",
    license="LGPL-3.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Typing :: Typed",
    ],
    keywords="jax automatic-differentiation batching mathematics",
    packages=find_packages(exclude=["docs", "tests", "local", "report"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    project_urls={
        "Issues Tracker": "https://github.com/unalmis/adv-jax-math/issues",
        "Contributing": "https://github.com/unalmis/adv-jax-math/blob/main/.github/CONTRIBUTING.rst",  # noqa: E501
        "Source Code": "https://github.com/unalmis/adv-jax-math/",
        "Documentation": "https://unalmis.github.io/adv-jax-math/",
    },
)
