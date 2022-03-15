from setuptools import setup

setup(
    name="partglot",
    version=0.1,
    description="PartGlot: Learning Shape Part Segmentation from Language Reference Games",
    url="https://github.com/63days/PartGlot",
    author="Juil Koo",
    author_email="63days@kaist.ac.kr",
    packages=["partglot"],
    install_requires=[
        "pandas",
        "torch",
        "h5py",
        "Pillow",
        "numpy",
        "matplotlib",
        "six",
        "nltk",
        "pytorch-lightning==1.5.5",
        "hydra-core==1.1.0",
        "omegaconf==2.1.1"
    ],
    zip_safe=False,
)
