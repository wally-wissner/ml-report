import pathlib
from setuptools import setup


__version__ = "0.0.1"


here = pathlib.Path(__file__).parent.resolve()


setup(
    name="ml-report",
    version=__version__,
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
        'Development Status :: 2 - Pre-Alpha'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
    ],
    package_dir={'': "ml_report"},
    python_requires=">=3.7",
)

# test.pypi.org
