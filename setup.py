'''
Created on Jun 11, 2012

@author: fmertens
'''

from setuptools import setup, find_packages


setup(
    name = 'wise',
    version = '0.1',
    description = 'Wavelet Image Segmentation and Evaluation',
    url = 'https://github.com/flomertens/wise',
    author = 'Florent Mertens',
    author_email = 'flomertens@gmail.com',
    license='GPL2',

    include_package_data=True,
    packages=find_packages(),
)

