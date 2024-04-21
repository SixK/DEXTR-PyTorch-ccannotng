
  
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='dextr',
    version='0.0.1',
    description='Deep cut python package',
    url='https://github.com/jsbroks/dextr-keras',
    author='Sergi Caelles',
    author_email='scaelles@vision.ee.ethz.ch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=[
        'numpy',
        'opencv-python>=3',
        'torch',
        'pillow',
        'scikit-learn',
        'scikit-image'
    ],
    # packages=find_packages(include=('dextr', 'dataloaders', 'networks')),
    packages=['dextr', 'dataloaders', 'networks' ],
    python_requires='>=3',
    zip_safe=False
)
