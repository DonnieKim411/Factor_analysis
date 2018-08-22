import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Factor_analysis_pkg",
    version="0.0.1",
    author="Donnie Kim",
    author_email="kdk411@gmail.com",
    description="perform factor analysis with EM method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kdk411/EM_factor_analysis",
    packages=setuptools.find_packages(),
    install_requires=[
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)