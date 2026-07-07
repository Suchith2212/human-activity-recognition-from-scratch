"""Setup script for the Human Activity Recognition package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="har-recognition",
    version="1.0.0",
    author="Suchith",
    description=(
        "Human Activity Recognition using wearable sensor data — "
        "from-scratch ML implementations with production-grade engineering"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Suchith2212/human-activity-recognition-from-scratch",
    packages=find_packages(exclude=["notebooks", "notebooks.*", "tests", "tests.*", "scripts", "scripts.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
