import pathlib
from setuptools import setup


version = "0.0.1"


here = pathlib.Path(__file__).parent.resolve()


setup(
    name="ml-report",
    version=version,
    description="Automated reporting for training Machine Learning models",
    long_description=(here / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    author="Wally Wissner",
    author_email="wally.wissner.MS@gmail.com",
    url="https://github.com/wally-wissner/ml_report",
    license=(here / 'LICENSE.txt').read_text(),
    install_requires=[
        "setuptools",
    ],
    classiiers=[
        'Development Status :: 3 - Alpha',
    ],
    package_dir={'': "ml_report"},
)

# test.pypi.org
