from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

setup(
    name="urnng",
    version="0.1.0",
    description="Third party wrapper of original URNNG repo.",
    long_description=readme,
    author="Benjamin Lipkin",
    author_email="lipkinb@mit.edu",
    license="MIT",
    packages=find_packages(where="urnng"),
    install_requires=requirements,
    python_requires="3.6",
)
