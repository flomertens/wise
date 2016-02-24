import os
import re
import glob
import logging
import datetime

import wds
import matcher
import wiseutils
import features as wfeatures

import numpy as np

from libwise import plotutils, nputils, imgutils
from libwise.nputils import validator_is, is_callable, validator_in_range, validator_list, str2bool, str2floatlist

import jsonpickle as jp

import astropy.units as u

logger = logging.getLogger(__name__)


def quantity_decode(s):
    try:
        value, unit = re.match('(\d+\.*\d*)\s*([a-zA-Z]+)', s).group(1,2)
    except:
        raise ValueError("Quantity '%s' is not in the right format." % s)
    return float(value) * u.Unit(unit)


class DataConfiguration(nputils.BaseConfiguration):
    """Data configuration object."""

    def __init__(self):
        data = [
        ["data_dir", None, "Base data directory", validator_is(str), str, str, 0],
        ["fits_extension", 0, "Extension index", validator_is(int), int, str, 0],
        ["stack_image_filename", "full_stack_image.fits", "Stack Image filename", nputils.validator_is(str), str, str, 2],
        ["ref_image_filename", "reference_image", "Reference image filename", validator_is(str), str, str, 0],
        ["mask_filename", "mask.fits", "Mask filename", validator_is(str), str, str, 0],
        ["bg_fct", None, "Background extraction fct", is_callable, None, None, 2],
        ["bg_coords", None, "Background region in coordinates [Xa,Ya,Xb,Yb]", validator_list(4, (int, float)), 
            str2floatlist, jp.encode, 0],
        ["bg_use_ksigma_method", False, "Use ksigma method to estimate the background level", validator_is(bool), str2bool, str, 0],
        ["roi_coords", None, "Region of interest in coordinates [Xa,Ya,Xb,Yb]", validator_list(4, (int, float)), 
            str2floatlist, jp.encode, 0],            
        ["core_offset_filename", "core.dat", "Core offset filename", validator_is(str), str, str, 0],
        ["core_offset_fct", None, "Core offset generation fct", is_callable, None, None, 2],
        ["pre_bg_process_fct", None, "Initial processing before bg extraction", is_callable, None, None, 2],
        ["pre_process_fct", None, "Pre detection processing", is_callable, None, None, 2],
        ["post_process_fct", None, "Post detection processing", is_callable, None, None, 2],
        ["crval", None, "CRVAL", validator_is(list), jp.decode, jp.encode, 1],
        ["crpix", None, "CRPIX", validator_is(list), jp.decode, jp.encode, 1],
        ["projection_unit", u.mas, "Unit used for the projection", validator_is(u.Unit), u.Unit, str, 0],
        ["projection_relative", True, "Use relative projection", validator_is(bool), str2bool, str, 0],
        ["projection_center", "pix_ref", "Method used to get the center", validator_is(str), str, str, 0],
        ["object_distance", None, "Object distance", validator_is(u.Quantity), quantity_decode, str, 0],
        ["object_z", 0, "Object z", validator_in_range(0, 5), float, str, 0],
        ]

        # nputils.BaseConfiguration.__init__(self, data, title="Finder configuration")
        super(DataConfiguration, self).__init__(data, title="Data configuration")


class AnalysisConfiguration(nputils.ConfigurationsContainer):
    """Analysis configuration container.
    
    Attributes
    ----------
    data : :class:`DataConfiguration`
        Data configuration
    finder : :class:`FinderConfiguration`
        Detection configuration
    matcher : :class:`MatcherConfiguration`
        Matcher configuration
    """
    def __init__(self):
        self.data = DataConfiguration()
        self.finder = wds.FinderConfiguration()
        self.matcher = matcher.MatcherConfiguration()
        nputils.ConfigurationsContainer.__init__(self, [self.data, self.finder, self.matcher])

    # def to_file(self, name, path):
    #     base = os.path.join(path, name)
    #     self.data.to_file(base + ".data.conf")
    #     print "Saved data configuration @ %s" % base + ".data.conf"
    #     self.finder.to_file(base + ".finder.conf")
    #     print "Saved finder configuration @ %s" % base + ".finder.conf"
    #     self.matcher.to_file(base + ".matcher.conf")
    #     print "Saved matcher configuration @ %s" % base + ".matcher.conf"


class AnalysisResult(object):
    """Analysis result container.
    
    Parameters
    ----------
    config : :class:`AnalysisConfiguration`
        The project configuration used.

    Attributes
    ----------
    config : :class:`AnalysisConfiguration`
        Configuration used during processing
    detection : :class:`wise.wds.MultiScaleImageSet`
        Detection result
    image_set : :class:`libwise.imgutils.ImageSet`
        Set of images
    link_builder : :class:`wise.matcher.MultiScaleFeaturesLinkBuilder`
        Matching result (as FetauresLinks)
    ms_match_results : :class:`wise.matcher.MultiScaleMatchResultSet`
        Matching result (as FeaturesMatch and DeltaInformation)
    """
    def __init__(self, config):
        self.detection = wds.MultiScaleImageSet()
        self.ms_match_results = matcher.MultiScaleMatchResultSet()
        self.link_builder = matcher.MultiScaleFeaturesLinkBuilder()
        self.image_set = imgutils.ImageSet()
        self.config = config

    def has_detection_result(self):
        """Return True if this object contains any detection result."""
        return len(self.detection) > 0

    def has_match_result(self):
        """Return True if this object contains any matching result."""
        return len(self.ms_match_results) > 0

    def get_scales(self):
        """Return the list of scales in the results"""
        return self.detection.get_scales()

    def add_detection_result(self, img, res):
        """Add a detection result"""
        self.image_set.add_img(img)
        self.detection.append(res)

    def add_match_result(self, match_res):
        """Add a matching result"""
        self.ms_match_results.append(match_res)
        self.link_builder.add_match_result(match_res)

    def get_match_result(self):
        """Get the matching result as a tuple of :class:`wise.matcher.MultiScaleMatchResultSet`
        and :class:`wise.matcher.MultiScaleFeaturesLinkBuilder` """
        if self.ms_match_results is None:
            raise Exception("No match result found.")
        return self.ms_match_results, self.link_builder

    def get_detection_result(self):
        """Get the detection result as a :class:`wise.wds.MultiScaleImageSet`"""
        if self.detection is None:
            raise Exception("No detection result found.")
        return self.detection


class AnalysisContext(object):
    """An analysis context encapsulates all the configuration and the results
    of a project.

    Example
    --------
    >>> ctx = wise.AnalysisContext()
    >>> ctx.config.data.data_dir = os.path.expanduser("~/project/data")

    >>> ctx.config.finder.min_scale = 1
    >>> ctx.config.finder.max_scale = 3

    >>> ctx.config.matcher.method_klass = wise.ScaleMatcherMSCSC2

    >>> ctx.select_files(os.path.expanduser("~/project/files/*"))
    

    Parameters
    ----------
    config : :class:`AnalysisConfiguration`, optional
        The project configuration. If not set, a default configuration will be used.


    Attributes
    ----------
    config : :class:`AnalysisConfiguration`
        The project configuration.
    files : list
        The project files.
    result : :class:`AnalysisResult`
        the project results.

    """
    def __init__(self, config=None):
        if config is None:
            config = AnalysisConfiguration()
        self.config = config
        self.result = AnalysisResult(self.config)
        self.files = []

        self._cache_mask_filter = None
        self._cache_core_offset = None

    def get_data_dir(self):
        """Return the project data directory as configured by config.data.data_dir. 
        If the directory does not exist, it will be created. """
        path = self.config.data.data_dir
        if self.config.data.data_dir is None:
            return os.getcwd()
        if not os.path.exists(path):
            print "Creating %s" % path
            os.makedirs(path)
        return path

    def get_projection(self, img=None):
        """ Return a :class:`libwise.imgutils.Projection` corresponding to `img` and the settings 
        defined in config.data. If `img` is not set, the reference image will be used instead.

        Parameters
        ----------
        img : :class:`libwise.imgutils.Image`
        """
        if img is None:
            img = self.get_ref_image()
        return img.get_projection(relative=self.config.data.projection_relative, 
                                  center=self.config.data.projection_center, 
                                  unit=self.config.data.projection_unit, 
                                  distance=self.config.data.object_distance, 
                                  z=self.config.data.object_z)

    def get_core_offset_filename(self):
        path = self.get_data_dir()
        if self.config.data.core_offset_filename is None:
            return None
        return os.path.join(path, self.config.data.core_offset_filename)

    def get_core_offset(self):
        """ Return a :class:`CoreOffsetPositions`  based on the core position 
        defined in the file self.config.data.core_offset_filename.
        """
        filename = self.get_core_offset_filename()
        if not os.path.isfile(filename):
            return None
        mtime = os.path.getmtime(filename)
        if self._cache_core_offset is None or self._cache_core_offset[0] != (mtime, filename):
            core_offset_pos = wiseutils.CoreOffsetPositions.new_from_file(filename)
            self._cache_core_offset = [(mtime, filename), core_offset_pos]
        return self._cache_core_offset[1]

    def get_mask_filename(self):
        path = self.get_data_dir()
        if self.config.data.mask_filename is None:
            return None
        return os.path.join(path, self.config.data.mask_filename)

    def get_mask(self):
        """ Return a mask (:class:`libwise.imgutils.Image`) from self.config.data.mask_filename.
        """
        filename = self.get_mask_filename()
        if not os.path.isfile(filename):
            return None
        mtime = os.path.getmtime(filename)
        if self._cache_mask_filter is None or self._cache_mask_filter[0] != (mtime, filename):
            mask = imgutils.FitsImage(filename)
            mask.data = mask.data.astype(bool)
            self._cache_mask_filter = [(mtime, filename), mask]
        return self._cache_mask_filter[1]

    def get_stack_image_filename(self):
        path = self.get_data_dir()
        if self.config.data.stack_image_filename is None:
            return None
        return os.path.join(path, self.config.data.stack_image_filename)

    def get_ref_image(self, preprocess=True):
        """Return the reference image (:class:`libwise.imgutils.Image`) of the project, 
        used for the projection defintion and several plotting tasks.

        A reference image can be set using self.config.data.ref_image_filename. 
        Alternatively the first file of the project is used.
        
        Parameters
        ----------
        preprocess : bool, optional
            If True, the reference image is pre processed .
        """
        ref_file =  os.path.join(self.get_data_dir(), self.config.data.ref_image_filename)
        if not os.path.isfile(ref_file):
            if len(self.files) == 0:
                raise Exception("No files selected")
            ref_file = self.files[0]

        img = imgutils.guess_and_open(ref_file, check_stack_img=True)

        if preprocess:
            self.pre_process(img)

        return img

    def set_ref_image(self, img):
        """Set the reference image.
        
        Parameters
        ----------
        img : :class:`libwise.imgutils.Image`
            The new reference image.
        """
        ref_file =  os.path.join(self.get_data_dir(), self.config.data.ref_image_filename)
        img.save(ref_file)

    def get_result(self):
        """Return the project result (:class:`AnalysisResult`)
        """
        return self.result

    def get_match_result(self):
        """Return the matching result (tuple of :class:`wise.matcher.MultiScaleMatchResultSet`
        and :class:`wise.matcher.MultiScaleFeaturesLinkBuilder`).
        """
        return self.result.get_match_result()

    def get_detection_result(self):
        """Return the detection result (:class:`wise.wds.MultiScaleImageSet`).
        """
        return self.result.get_detection_result()

    def align(self, img):
        """Align image using core position defined in self.config.data.core_offset_filename.
        
        Parameters
        ----------
        img : :class:`libwise.imgutils.Image`
            The image to align.
        """
        core_offset = self.get_core_offset()
        if core_offset is not None:
            print "Aligning:", img.get_epoch()
            core_offset.align_img(img, projection=self.get_projection(img))

    def build_stack_image(self, preprocess=False, nsigma=0, nsigma_connected=False):
        """Create a stacked image (:class:`libwise.imgutils.StackedImage` of all 
           the project images, aligning them if necessary.
        
        Parameters
        ----------
        preprocess : bool, optional
            If True, the images are pre processed .
        nsigma : int, optional
            Clip bg below nsigma level
        nsigma_connected : bool, optional
            If True, keep only the brightest connected structure
        """
        stack_builder = imgutils.StackedImageBuilder()
        stack_bg_builder = imgutils.StackedImageBuilder()
        for file in self.files:
            img = self.open_file(file)
            self.pre_bg_process(img)
            bg = imgutils.Image(self.get_bg(img))
            stack_bg_builder.add(bg)
            if preprocess:
                self.pre_process(img)
            self.align(img)
            stack_builder.add(img)

        stack_img = stack_builder.get()
        stack_bg = stack_bg_builder.get()

        if nsigma > 0:
            stack_img.data[stack_img.data < nsigma * stack_bg.data.std()] = 0
            if nsigma_connected:
                segments = wds.SegmentedImages(stack_img)
                segments.connected_structure()
                stack_img.data = segments.sorted_list()[-1].get_segment_image()
        return stack_img

    def get_stack_image(self, nsigma=0, nsigma_connected=False, preprocess=False):
        filename = self.get_stack_image_filename()
        if filename is None or not os.path.isfile(filename):
            raise Exception("A stack image need to be generated")

        stack_image = imgutils.StackedImage.from_file(filename)
        if nsigma > 0:
            bg = self.get_bg(stack_image)
        if preprocess:
            self.pre_process(stack_image)
        if nsigma > 0:
            stack_image.data[stack_image.data < nsigma * bg.std()] = 0
            if nsigma_connected:
                segments = wds.SegmentedImages(stack_image)
                segments.connected_structure()
                stack_image.data = segments.sorted_list()[-1].get_segment_image()

        return stack_image

    def open_file(self, file):
        """Open `file` and return an image (:class:`libwise.imgutils.Image`).
        """
        img = imgutils.guess_and_open(file, fits_extension=self.config.data.fits_extension)
        if self.config.data.crval is not None:
            img.set_crval(self.config.data.crval)
        if self.config.data.crpix is not None:
            img.set_pix_ref(self.config.data.crpix)
        return img

    def get_bg(self, img):
        """Return either the noise level or a background map.
        """
        if self.config.data.bg_use_ksigma_method:
            return nputils.k_sigma_noise_estimation(img.data)
        if self.config.data.bg_coords is not None:
            x1, y1, x2, y2 = self.config.data.bg_coords
            prj = self.get_projection(img)
            xy_p1, xy_p2 = np.round(prj.s2p([(x1, y1), (x2, y2)])).astype(int)
            ex = [0, img.data.shape[1]]
            ey = [0, img.data.shape[0]]
            xlim1, xlim2 = sorted([nputils.clamp(xy_p1[0], *ex), nputils.clamp(xy_p2[0], *ex)])
            ylim1, ylim2 = sorted([nputils.clamp(xy_p1[1], *ey), nputils.clamp(xy_p2[1], *ey)])
            return img.data[ylim1:ylim2, xlim1:xlim2].copy()
        elif self.config.data.bg_fct is not None:
            return self.config.data.bg_fct(self, img)
        raise Exception("Bg extraction method need to be set")

    def pre_bg_process(self, img):
        if self.config.data.pre_bg_process_fct is not None:
            self.config.data.pre_bg_process_fct(self, img)

    def pre_process(self, img):
        """Run self.config.data.pre_process_fct on `img`
        """
        if self.config.data.pre_process_fct is not None:
            self.config.data.pre_process_fct(self, img)
        if self.config.data.roi_coords is not None:
            x1, y1, x2, y2 = self.config.data.roi_coords
            img.crop((x1, y1), (x2, y2), projection=self.get_projection(img))

    def post_process(self, img, res):
        if self.config.data.post_process_fct is not None:
            self.config.data.post_process_fct(self, img, res)

    def save_core_offset_pos_file(self):
        """ Create a core position definition object based on self.config.data.core_offset_fct
        and save the result on disk using path defined in self.config.data.core_offset_filename.
        """
        if self.config.data.core_offset_fct is None:
            print "Warning: No core offset fct defined"
            return
        filename = self.get_core_offset_filename()
        core_offset_pos = wiseutils.CoreOffsetPositions()

        for file in self.files:
            img = self.open_file(file)
            core = self.config.data.core_offset_fct(self, img)
            core_offset_pos.set(img.get_epoch(), core)

        self._cache_core_offset = None
        core_offset_pos.save(filename)

    def save_mask_file(self, mask_fct):
        """Create a mask image based on `mask_fct` and save the result on disk using
        path defined in self.config.data.mask_filename. `mask_fct` must be a function
        accepting an :class:`AnalysisContext` as argument and returning a corresponding 
        mask as :class:`libwise.imgutils.Image`.
        """
        filename = self.get_mask_filename()
        if os.path.isfile(filename):
            os.remove(filename)
        mask = mask_fct(self)
        mask.data = mask.data.astype(bool).astype(float)

        self._cache_mask_filter = None
        mask.save_to_fits(filename)

    def save_stack_image(self, preprocess=False):
        stack = self.build_stack_image(preprocess=preprocess)
        stack.save_to_fits(self.get_stack_image_filename())

    def detection(self, img, config=None, filter=None, verbose=True):
        """Run detection on `img` (:class:`libwise.imgutils.Image`).
        """
        if verbose:
            print "Start detection on: %s" % img
            if filter is not None:
                print "  with filter: %s" % filter
        self.pre_bg_process(img)
        bg = self.get_bg(img)
        detection_filter = wfeatures.MaskFilter(self.get_mask())
        if filter is not None:
            detection_filter = detection_filter & filter

        self.align(img)
        self.pre_process(img)

        if config is None:
            config = self.config.finder

        finder = wds.FeaturesFinder(img, bg, config=config, filter=detection_filter)

        res = finder.execute()

        self.post_process(img, res)

        return res

    def select_files(self, files, start_date=None, end_date=None, filter_dates=None, step=1):
        """Set the images files of the projects. The `files` parameter accept shell like wildcards, 
        and it is possible to filter files by dates.

        Examples
        ---------

        >>> ctx.select_files('/project/files/*.fits')

        >>> ctx.select_files('/project/files/*.fits', start_date=datetime.datetime(2000, 1, 1))

        >>> ctx.select_files('/project/files/*.fits', step=2)
        

        Parameters
        ----------
        files : str
            Path to the file(s). Accept shell like wildcards.
        start_date : :class:`datetime.datetime`, optional
            Reject files with date < `start_date`.
        end_date : :class:`datetime.datetime`, optional
            Reject files with date > `end_date`.
        filter_dates : a list of :class:`datetime.datetime`, optional
            Reject files with date in `filter_dates`
        step : int, optional
        """
        if isinstance(files, str):
            files = glob.glob(files)

        self.files = imgutils.fast_sorted_fits(files, start_date=start_date, 
                            end_date=end_date, filter_dates=filter_dates, step=step)

        print "Number of files selected:", len(self.files)

    def match(self, find_res1, find_res2, verbose=True):
        """Run match on `find_res1` and `find_res2` (both :class:`wise.wds.SegmentedScale`)
        """
        m = matcher.ImageMatcher(self.config.finder, self.config.matcher)

        return m.get_match(find_res1, find_res2, verbose=verbose)

