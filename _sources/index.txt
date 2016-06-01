WISE: Wavelet Image Segmentation and Evaluation
===============================================

The WISE package has been developed to address the issue of detecting
significant features in radio interferometric images and obtaining reliable
velocity field from cross-correlation of these regions in multi-epoch
observations.

It comprises three main constituents:

- Detection of structural information is performed using the segmented wavelet
  decomposition (SWD) method. The SWD algorithm provides a structural
  representation of astronomical images with exceptional sensitivity for
  identifying compact and marginally resolved features as well as large scale
  structural patterns. It delivers a set of two-dimensional significant structural
  patterns (SSP), which are identified at each scale of the wavelet decomposition.

- Tracking of these SSP detected in multiple-epoch images is performed with a
  multiscale cross-correlation (MCC) algorithm. It combines structural information
  on different scales of the wavelet decomposition and provides a robust and
  reliable cross-identification of related SSP.

- A stacked cross correlation (SCC) is introduced to recover multiple velocity
  components from partially overlapping emitting regions.

.. image:: imgs/m87_combined.png
    :width: 800px
    :align: center

This software is based on a method introduced and described in:

| *"Wavelet-based decomposition and analysis of structural patterns in astronomical image"*
| Mertens, F., Lobanov, A.P. 2015, Astronomy & Astrophysics, 574, 67 `ADS <http://adsabs.harvard.edu/abs/2015A%26A...574A..67M>`_

| *"Detection of multiple velocity components in partially overlapping emitting regions"*
| Mertens, F., Lobanov, A.P. 2016, Astronomy & Astrophysics, 587, 52 `ADS <http://adsabs.harvard.edu/abs/2016A%26A...587A..52M>`_

Installation
-------------

The WISE package can be installed using the conda package management system. You
can install conda easily by downloading the installer corresponding to your OS
and for Python 2.7 at:

http://conda.pydata.org/miniconda.html

and following the installation instruction at:

http://conda.pydata.org/docs/install/quick.html

Once done, you can install WISE in just one command line:

::

    conda install --channel https://conda.anaconda.org/flomertens wise


Alternative manual installation:
--------------------------------

Latest version of WISE and libwise package can be download at:

https://github.com/flomertens/libwise/releases/

https://github.com/flomertens/wise/releases/

Each package can be installed with the command line:

::

    python setup.py install

Requirements:

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

Using WISE
----------

WISE can be used either from the command line or from an ipython notebook. Check
the tutorials below to see how to use it.


Tutorials
----------

.. toctree::
   :maxdepth: 2

   tutorials_cmd/index
   tutorials_notebook/index

Full API documentation:
-----------------------

.. toctree::
   :maxdepth: 2

   docs/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

