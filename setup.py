import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'BIOT'
AUTHOR = 'Rebecca Marion'
AUTHOR_EMAIL = 'rebecca.marion@unamur.be'
URL = 'https://github.com/rebeccamarion/BIOT_Python_Package'

LICENSE = 'MIT'
DESCRIPTION = 'Implementation of Best Interpretable Orthogonal Transformation (BIOT)'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'sklearn',
      'math'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )