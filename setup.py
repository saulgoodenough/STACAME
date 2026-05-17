from setuptools import Command, find_packages, setup

__lib_name__ = "STACAME"
__lib_version__ = "1.1.0"
__description__ = "Integrating spatial transcriptomics data across species"
__url__ = "https://github.com/zhanglabtools/STACAME"
__author__ = "Biao Zhang"
__author_email__ = "biaozhang@ysu.edu.cn"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Cross-species", "Brain", "data integration", "Graph attention auto-encoder", "spatial domain", "three-dimensional reconstruction"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['STACAME'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)