'''
Created on Jun 11, 2012

@author: fmertens
'''

import glob
from setuptools import setup, find_packages


setup(
    name='wisetool',
    version='0.4.7',
    description='Wavelet Image Segmentation and Evaluation',
    url='https://github.com/flomertens/wise',
    author='Florent Mertens',
    author_email='flomertens@gmail.com',
    license='GPL2',

    include_package_data=True,
    packages=find_packages(),
    scripts=glob.glob('scripts/*'),
    install_requires=[
        'libwise',
        'numpy',
        'scipy',
        'scikit-image',
        'astropy',
        'matplotlib',
        'pyregion',
        'uncertainties',
        'pymorph',
        'jsonpickle']

)
