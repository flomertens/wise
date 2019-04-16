WISE
====

WISE is the Wavelet Image Segmentation and Evaluation tool, developed to address the issue of detecting significant features in radio interferometric images and obtaining reliable velocity field from cross-correlation of these regions in multi-epoch observations.

Please check https://flomertens.github.io/wise/ for more information, documentation and tutorials.

Installation
------------

You should be able to install WISE using pip or conda:

    pip install wise

or

    conda install -c flomertens wise

Alternatively, to install WISE globally:

    python setup.py install

or to install it locally:

    python setup.py install --user

Requirements
------------

You need to install (if not done by pip/conda) the following packages for WISE to work:

- libwise (https://github.com/flomertens/libwise)
- numpy (>= 1.5)
- scipy (>= 0.10)
- skimage (>= 0.5)
- astropy (>= 0.4)
- matplotlib (>= 1.0)
- pyqt (>= 4.8)
- pandas
- pyregion (https://pypi.python.org/pypi/pyregion)
- uncertainties (https://pypi.python.org/pypi/uncertainties)
- pymorph (https://pypi.python.org/pypi/pymorph)
- jsonpickle (https://pypi.org/project/jsonpickle/)

