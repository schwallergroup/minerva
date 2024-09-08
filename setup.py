from setuptools import setup, find_packages

setup(
    name="minerva",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT",
    install_requires=[
        "botorch==0.6.0",
        "gpytorch==1.6.0",
        "ipykernel==6.29.4",
        "jupyter==1.0.0",
        "matplotlib==3.7.4",
        "pandas==2.0.3",
        "pykeops==1.5",
        "pytorch-lightning==2.1.2",
        "scikit-learn==1.3.2",
        "scipy==1.10.1",
        "torch==2.0.1",
        "tqdm==4.66.1",
    ],
)
