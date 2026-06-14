from setuptools import setup, find_packages

setup(
    name="compound-hazard",
    version="1.0.0",
    description=(
        "Compound orbital hazard analysis: ionizing radiation + space debris "
        "Pareto optimization using NASA OSDR RadLab ISS dosimetry data."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Minh Nguyen",
    author_email="Mnguyen@xorbita.com",
    url="https://github.com/minhnguyen/space-debris-compound-hazard",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "scipy>=1.11",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
    ],
)
