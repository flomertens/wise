import os
import logging
import datetime

import numpy as np
from scipy.spatial import KDTree

import astropy.units as u

from libwise import nputils, imgutils

p2i = imgutils.p2i
logger = logging.getLogger(__name__)


class Feature(object):
    '''Base class defining a feature. Features coordinate are stored as array index.

    Parameters
    ----------
    coord : list of 2 elements
        Coordinate of the features in array index.
    intensity : float
        Intensity of the feature
    '''

    __slots__ = ['coord', 'intensity', 'initial_coord', '__hash']

    def __init__(self, coord, intensity):
        self.set_coord(coord)
        self.intensity = intensity
        self.initial_coord = self.get_coord()

        self.__hash = hash(tuple(self.initial_coord.tolist() + [self.intensity]))

    def __str__(self):
        return "c:%s,i:%s" % (self.get_coord(), self.intensity)

    def __eq__(x, y):
        return x.__hash == y.__hash

    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return str(self)

    def __cmp__(self, other):
        res = cmp(self.initial_coord[0], other.initial_coord[0])
        if res == 0:
            res = cmp(self.initial_coord[1], other.initial_coord[1])
        if res == 0:
            res = cmp(self.intensity, other.intensity)
        return res

    def set_coord(self, coord):
        """Set the coordinate of the feature."""
        self.coord = nputils.get_pair(coord, dtype=np.float32)

    def get_coord(self, mode=None):
        """Get the coordinate of the feature."""
        return self.coord.copy()

    def set_intensity(self, intensity):
        """Set the intensity of he feature."""
        self.intensity = intensity

    def get_intensity(self):
        '''Get the intensity of the feature.
        '''
        return self.intensity

    def move(self, delta):
        """Move the feature by an offset `delta` (in array index)."""
        coord = self.get_coord()
        self.set_coord(coord + delta)

    def move_back_to_initial(self):
        """Reset the coordinate to the initial one."""
        self.set_coord(self.initial_coord)

    def copy(self):
        new = Feature(self.get_coord(), self.get_intensity())
        new.initial_coord = self.initial_coord
        return new

    def distance(self, other, fct=None):
        """Return the euclidean distance between this feature and `other`."""
        if fct is None:
            fct = lambda f: f.get_coord()
        return np.linalg.norm(fct(self) - fct(other))


class ImageFeature(Feature):
    """Class defining a feature in an image.

    Parameters
    ----------
    coord : list of 2 elements
        Coordinate of the features in array index.
    meta : :class:`libwise.imgutils.ImageMeta`
        Image meta information.
    intensity : float
        Intensity of the feature
    snr : float
        Signal over noise ratio.
    id : str, optional
        A feature identifier. Must be unique.
    
    Attributes
    ----------
    id : str
        The feature identifier
    meta : :class:`libwise.imgutils.ImageMeta`
        Image meta information. 
    snr : float
        Signal over noise ratio of the feature.
    """
    __slots__ = ['coord', 'intensity', 'initial_coord', 'meta', 'snr', 'id', '__hash']

    def __init__(self, coord, meta, intensity, snr, id=None):
        Feature.__init__(self, coord, intensity)
        self.meta = meta
        self.snr = snr
        if id is None:
            id = ''
            if meta is not None:
                if isinstance(meta.get_epoch(), datetime.date):
                    id += meta.get_epoch().strftime("%Y%m%d")
                else:
                    id += str(meta.get_epoch())
                id += "@"
            id += 'x'.join([str(np.round(k, decimals=2)) for k in coord])

        self.id = id

    def get_epoch(self):
        """Return the epoch of the feature."""
        return self.meta.get_epoch()

    def get_id(self):
        ''' Return the identifier of the feature.'''
        return self.id

    def get_meta(self):
        ''' Return the feature image meta information (:class:`libwise.imgutils.ImageMeta`).'''
        return self.meta

    def get_coord_error(self, min_snr=0, max_snr=10):
        ''' Return the standard error on the coordinate position_angle.'''
        beam = self.meta.get_beam()
        width = np.array([1, 1])
        if isinstance(beam , imgutils.GaussianBeam):
            width_x = nputils.get_ellipse_radius(beam.bmin, beam.bmaj, beam.bpa)
            width_y = nputils.get_ellipse_radius(beam.bmin, beam.bmaj, np.pi / 2. + beam.bpa)
            width = [width_y, width_x]

        return width / (np.sqrt(2) * max(min(self.get_snr(), max_snr), min_snr))

    def get_rms_noise(self):
        ''' Return the rms noise of the image.'''
        return self.get_intensity() / self.get_snr()

    def get_snr(self):
        ''' Return the feature SNR.'''
        return self.snr

    def get_coordinate_system(self):
        ''' Return the image coordinate system (:class:`libwise.imgutils.AbstractCoordinateSystem`).'''
        return self.meta.get_coordinate_system()

    def copy(self):
        new = ImageFeature(self.get_coord(), self.meta, self.intensity, self.snr)
        new.initial_coord = self.initial_coord
        return new


class FeaturesGroup(object):
    '''A group of features.
    
    Attributes
    ----------
    features : list
    '''

    def __init__(self, features=None):
        if features is None:
            features = []
        self.features = list(features)

    def __str__(self):
        return str([str(k) for k in self.features])

    def __iter__(self):
        return iter(self.features)

    def __len__(self):
        return self.size()

    def set_features(self, features):
        '''Set the list of features.'''
        self.features = features

    def merge(self, other):
        ''' Merge this list of features with the features from `other`.'''
        self.features.extend(other.features)

    @classmethod
    def from_img_peaks(Klass, img, width, threashold, feature_filter=None, 
                       fit_gaussian=False, exclude_border_dist=1):
        '''Detect local maxima in an image and return a corresponding :class:`FeaturesGroup`.
        
        Parameters
        ----------
        img : :class:`libwise.imgutils.Image`
            The image to analyse.
        width : int
            The typical width of the features to detect.
        threashold : float
            Threshold above which a local maxima is considered a feature.
        feature_filter : :class:`FeatureFilter`, optional
            Filter the features.
        fit_gaussian : bool, optional
            If true, a sub pixel coordinate position is estimated by fitting a 2D 
            gaussian profile on the feature.
        exclude_border_dist : int, optional
            Exclude features which are at distance < `exclude_border_dist` from 
            teh image border.
        '''
        width = max(width, 2)
        img_meta = img.get_meta()
        threashold_pics_coord = nputils.find_peaks(img.data, width, threashold, exclude_border=False,
                                                   fit_gaussian=fit_gaussian,
                                                   exclude_border_dist=exclude_border_dist)

        result = Klass()

        for coord in threashold_pics_coord:
            pic_max = img.data[tuple(coord)]
            snr = pic_max / float(threashold)
            feature = ImageFeature(coord, img_meta, pic_max, snr)
            if feature_filter is not None:
                res = feature_filter.filter(feature)
                if res is False:
                    continue
            result.add_feature(feature)

        return result

    def intersect(self, other):
        ''' Return the intersection between this list of features and `other`.'''
        return FeaturesGroup(set(self.features).intersection(set(other.features)))

    def has_feature(self, feature):
        ''' Return True if the length of this list of features is not null.'''
        return feature in self.features

    def add_feature(self, feature, test_exist=False):
        ''' Append a feature to this list of feature.'''
        assert isinstance(feature, Feature)

        if test_exist and self.has_feature(feature):
            return

        self.features.append(feature)

    def remove_feature(self, feature):
        ''' Remove a feature to this list of features.'''
        self.features.remove(feature)

    def add_features_group(self, group):
        ''' Append a list of features to this list of features.'''
        assert isinstance(group, FeaturesGroup)

        for feature in group.get_features():
            self.add_feature(feature)

    def get_features(self):
        ''' Return the list of features as a set.'''
        return set(self.features)

    def filter(self, feature_filter):
        ''' In place filtering of this list of features. '''
        for feature in list(self.features):
            if not feature_filter.filter(feature):
                self.remove_feature(feature)

    def get_filtered(self, feature_filter):
        ''' Return a new filtered list of features.'''
        new = self.copy()
        new.filter(feature_filter)
        return new

    def move_features(self, delta_fct, time_delta=1):
        ''' Move all features of an offset determined by a function `delta_fct`
        which take a feature as argument and return a delta offset.'''
        deltas = DeltaInformation(self)
        delta_fct = nputils.make_callable(delta_fct)
        for feature in sorted(self.features):
            delta = np.round(delta_fct(feature.get_coord()))
            deltas.add_delta(feature, delta, time_delta)
            feature.move(delta)
        return deltas

    def move_features_from_delta_info(self, delta_info):
        ''' Move all features using information from a :class:`DeltaInformation`.'''
        for feature in self.features:
            delta = delta_info.get_delta(feature)
            if delta is not None:
                feature.move(delta)

    def move_back_to_initial(self):
        ''' Move all features to there initial coordinates.'''
        for feature in self.features:
            feature.move_back_to_initial()

    def get_coords(self, mode='lm'):
        ''' Return a list of all features coordinates.'''
        coords = np.zeros((len(self.features), 2))
        for i, feature in enumerate(self.features):
            coords[i] = feature.get_coord(mode=mode)
        return coords

    def get_intensities(self):
        ''' Return a list of all features intensities.'''
        return [k.get_intensity() for k in self.features]

    def find(self, feature, tol=0, mode=None):
        ''' Return a list of this group features which are at a distance < tol of `feature`.
        The returned list is sorted by increasing distance.'''
        founds = []
        for f in self.features:
            dist = feature.distance(f, mode)
            if tol is None or dist < tol:
                founds.append([f, dist])
        # sort the founds matching feature by distance
        founds = sorted(founds, cmp=lambda x, y: cmp(x[1], y[1]))
        return [k[0] for k in founds]

    def find_at_coord(self, coord, tol=0, coord_fct=None):
        ''' Return a list of this group features which are at a distance < tol of `coord`.
        The returned list is sorted by increasing distance.'''
        founds = []
        if coord_fct is None:
            coord_fct = lambda f: f.get_coord()
        for feature in self.features:
            dist = np.linalg.norm(coord_fct(feature) - coord)
            if tol is None or dist < tol:
                founds.append([feature, dist])
        # sort the founds matching feature by distance
        founds = sorted(founds, cmp=lambda x, y: cmp(x[1], y[1]))
        return [k[0] for k in founds]

    def sorted_list(self, cmp=None, key=None):
        ''' Sort this list of features using `cmp` or `key`. 
        See Python list documentation for more information.'''
        if cmp is None and key is None:
            key = lambda f: f.get_intensity()
        l = list(self.get_features())
        return sorted(l, cmp=cmp, key=key)

    def get_match(self, features_group2, tol=0, pop=True):
        f1_no_match = FeaturesGroup()
        f2_no_match = FeaturesGroup()
        match = FeaturesMatch()
        features_group2 = features_group2.copy()

        for feature in self.get_features():
            test = features_group2.find(feature, tol=tol)
            if len(test) > 0:
                match.add_feature_match(feature, test[0])
                if pop:
                    features_group2.remove_feature(test[0])
            else:
                f1_no_match.add_feature(feature)

        f2_match = match.get_twos()
        for feature in features_group2.get_features():
            if not f2_match.has_feature(feature):
                f2_no_match.add_feature(feature)

        return match, f1_no_match, f2_no_match

    def get_match2(self, features_group2, tol=0):
        q1 = FeaturesQuery(self)
        return q1.get_match(features_group2, tol=tol)

        # match = dict()
        # changed = True

        # while changed:
        #     changed = False
        #     for feature in self.get_features() - set(match.values()):
        #         test = features_group2.find(feature, tol=tol)
        #         if len(test) > 0:
        #             if test[0] in match and \
        #                     test[0].distance(feature) >= test[0].distance(match[test[0]]):
        #                 # already match and not better
        #                 continue
        #             elif test[0] in match:
        #                 # already match but better
        #                 match[test[0]] = feature
        #                 changed = True
        #             else:
        #                 match[test[0]] = feature

        # match_obj = FeaturesMatch()
        # for f2, f1 in match.items():
        #     match_obj.add_feature_match(f1, f2)

        # return match_obj

    def get_minimum_separation(self):
        min_sep = 50000
        for feature1 in self.get_features():
            for feature2 in self.get_features():
                if feature1 == feature2:
                    continue
                sep = np.linalg.norm(feature1.get_coord() - feature2.get_coord())
                if sep < min_sep:
                    min_sep = sep
        return min_sep

    def copy(self):
        return FeaturesGroup([k.copy() for k in self.features])

    def size(self):
        return len(self.features)


class FeaturesQuery(object):

    def __init__(self, fgroup, coord_modes=['lm']):
        self.features1 = fgroup.features
        self.nf1 = len(self.features1) 
        coords1 = np.array([f.get_coord(mode=mode) for mode in coord_modes for f in self.features1])
        if len(self.features1) > 0:
            self.kdtree = KDTree(coords1)

    def get_features(self):
        return set(self.features1)

    def find(self, feature2, tol=np.inf):
        if len(self.features1) == 0:
            return []
        d, i = self.kdtree.query(feature2.get_coord(), k=len(self.features1),
                                 distance_upper_bound=tol)
        if len(self.features1) == 1:
            if d < tol:
                return self.features1
            return []
        i = i[d < tol]
        return nputils.uniq([self.features1[k % self.nf1] for k in i])

    def get_match(self, fgroup2, tol=2):
        if len(self.features1) == 0:
            return FeaturesMatch()
        coords2 = np.array([f.get_coord() for f in fgroup2.features])
        d, i = self.kdtree.query(coords2, k=1, distance_upper_bound=tol)
        i1 = i[d < tol]
        i2 = np.arange(len(coords2))[d < tol]
        features1 = [self.features1[k % self.nf1] for k in i1]
        features2 = [fgroup2.features[k] for k in i2]

        return FeaturesMatch(features1=features1, features2=features2)



class DatedFeaturesGroup(FeaturesGroup):

    def __init__(self, features=None, epoch=0):
        FeaturesGroup.__init__(self, features)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    def copy(self):
        return DatedFeaturesGroup([k.copy() for k in self.features], epoch=self.epoch)


class FeaturesMatch(object):

    def __init__(self, features1=None, features2=None):
        self.one_two = dict()
        self.two_one = dict()
        if features1 is not None and features2 is not None:
            self.one_two = dict(zip(features1, features2))
            self.two_one = dict(zip(features2, features1))

    def add_feature_match(self, feature1, feature2):
        self.one_two[feature1] = feature2
        self.two_one[feature2] = feature1

    def get_peer_of_one(self, feature1):
        return self.one_two.get(feature1)

    def get_peer_of_two(self, feature2):
        return self.two_one.get(feature2)

    def get_ones(self):
        return self.one_two.keys()

    def get_twos(self):
        return self.one_two.values()

    def get_pairs(self):
        return self.one_two.items()

    def remove(self, feature1):
        feature2 = self.get_peer_of_one(feature1)
        del self.one_two[feature1]
        del self.two_one[feature2]

    def filter(self, feature_filter):
        ''' In place filter '''
        for feature in self.get_ones():
            if not feature_filter.filter(feature):
                self.remove(feature)

    def get_filtered(self, feature_filter):
        new = self.copy()
        new.filter(feature_filter)
        return new

    def size(self):
        return len(self.one_two)

    def merge(self, other):
        self.one_two.update(other.one_two)
        self.two_one.update(other.two_one)

    def reverse(self):
        new = FeaturesMatch()
        new.two_one = self.one_two.copy()
        new.one_two = self.two_one.copy()
        return new

    def copy(self):
        new = FeaturesMatch()
        new.one_two = self.one_two.copy()
        new.two_one = self.two_one.copy()

        return new


class Delta(object):

    __slots__ = ['feature', 'delta', 'time']

    def __init__(self, feature, delta, time):
        self.feature = feature
        self.delta = np.array(delta, dtype=np.float32)
        self.time = time

    def get_delta(self):
        return self.delta

    def get_time(self):
        return self.time

    def get_feature(self):
        return self.feature

    def get_angular_velocity(self, projection):
        coord = self.feature.get_coord()
        return projection.angular_velocity(p2i(coord), p2i(coord + self.delta), self.time)

    def get_angular_velocity_vector(self, projection):
        import time
        coord = self.feature.get_coord()
        # d1 = projection.mean_pixel_scale() * np.array(p2i(self.delta)) * projection.unit
        # t = time.time()
        d = (projection.p2s(p2i(coord + self.delta)) - projection.p2s(p2i(coord))) * projection.unit
        # print projection.p2s(p2i(coord + self.delta)), projection.p2s(p2i(coord))
        # t1 = time.time() - t
        # d = projection.pixel_scales() * np.array(p2i(self.delta)) * projection.unit
        # t2 = time.time() - (t1 + t)
        # d3 = projection.angular_separation_vector(p2i(coord), p2i(coord + self.delta))
        # t3 = time.time() - (t1 + t2 + t)
        # print  projection.pixel_scales(), np.array(p2i(self.delta))
        # print "%s vs %s vs %s (%s vs %s, vs %s)" % (d1.to(u.mas), d.to(u.mas), d3.to(u.mas), t1, t2, t3)
        return d / (self.time.total_seconds() * u.second)

    def get_proper_velocity(self, projection):
        coord = self.feature.get_coord()
        return projection.proper_velocity(coord, coord + self.delta, self.time)


class DeltaInformation(object):

    NO_DELTA = 1 << 0
    DELTA_MATCH = 1 << 1
    DELTA_COMPUTED = 1 << 2

    def __init__(self, features=[], average_tol=10):
        # MEM ISSUE: using dict here does not scale well with increasing 
        # number of features. 
        self.deltas = dict()
        self.flags = dict.fromkeys(features, self.NO_DELTA)
        self.average_tol = average_tol

    def merge(self, other):
        self.deltas.update(other.deltas)
        self.flags.update(other.flags)

    def dump(self):
        for feature in self.get_features():
            delta = self.get_delta(feature)
            flag = self.get_flag(feature)
            print feature, delta, flag

    def add_match(self, match, delta_fct=None):
        if delta_fct is None:
            delta_fct = lambda x, y: y.get_coord() - x.get_coord()
        own_f1s = dict(zip(self.flags.keys(), self.flags.keys()))
        for f1, f2 in match.get_pairs():
            time_delta = f2.get_epoch() - f1.get_epoch()
            # f1 from match might have been moved. We want to get our own f1 to calculate delta correctly
            f1 = own_f1s[f1]
            self.add_delta(f1, delta_fct(f1, f2), time_delta, flag=self.DELTA_MATCH)

    def add_delta(self, feature, delta, time_delta, flag=DELTA_MATCH):
        assert delta is not None
        assert not np.isnan(delta[0]) and not np.isnan(delta[1]), feature
        # print "Add delta:", feature, delta
        self.add_full_delta(Delta(feature, delta, time_delta), flag=flag)

    def add_full_delta(self, delta, flag=DELTA_MATCH):
        self.deltas[delta.get_feature()] = delta
        self.flags[delta.get_feature()] = flag

    def remove_delta(self, feature):
        if feature in self.deltas:
            del self.deltas[feature]
        if feature in self.flags:
            self.flags[feature] = self.NO_DELTA

    def remove_feature(self, feature):
        if feature in self.deltas:
            del self.deltas[feature]
            del self.flags[feature]

    def filter(self, feature_filter):
        ''' In place filter '''
        for feature in self.get_features():
            if not feature_filter.filter(feature):
                self.remove_feature(feature)

    def get_filtered(self, feature_filter):
        new = self.copy()
        new.filter(feature_filter)
        return new

    def discard_diff_outliers(self, nsigma):
        from scipy.signal import detrend

        deltax, deltay = zip(*[k.get_delta() for k in self.deltas.values()])
        for delta in deltax, deltay:
            delta = detrend(delta)
            sigma = np.std(delta, ddof=1)
            
            for feature, delta in self.deltas.copy().items():
                if diff > alldiff.mean() + (nsigma * alldiff.std()) \
                        or diff < alldiff.mean() - (nsigma * alldiff.std()):
                    logger.info("Remove %s, %s, %s" % (feature, delta, diff))
                    self.remove_delta(feature)
                    yield feature

    def complete_with_average_delta(self, distance_mode=None):
        with_delta_query = FeaturesQuery(self.get_features(~self.NO_DELTA))
        for feature in self.flags.keys():
            if feature not in self.deltas:
                delta = self.get_average_delta_information(feature, with_delta_query, distance_mode=distance_mode)
                logger.debug("Complete with average delta: %s -> %s", feature, delta)
                if delta is not None:
                    self.add_delta(feature, np.round(delta), None, flag=self.DELTA_COMPUTED)

    def get_average_delta_information(self, feature, group_or_query, distance_mode=None):
        to_average = group_or_query.find(feature, self.average_tol)
        c = max((self.average_tol / 10.), 10)
        weight_fct = lambda distance: np.exp((-distance ** 2) / (2 * c ** 2))
        if len(to_average) > 0:
            weights = weight_fct(np.array([k.distance(feature, mode=distance_mode) for k in to_average]))
            if weights.sum() > 0:
                deltas = self.get_deltas(to_average)
                return np.average(deltas, axis=0, weights=weights)
        return None

    def get_features(self, flag=None):
        if flag is None:
            return FeaturesGroup(self.flags.keys())
        return FeaturesGroup([f for f, fflag in self.flags.items() if fflag & flag == fflag])

    def get_deltas(self, features=None):
        return np.array([self.get_delta(k) for k in features])

    def get_items(self):
        return self.deltas.items()

    def get_delta(self, feature):
        if feature not in self.deltas:
            return [0, 0]
        return self.deltas[feature].get_delta()

    def get_full_delta(self, feature):
        return self.deltas.get(feature, None)

    def get_full_deltas(self, flag=None):
        if flag is None:
            return self.deltas.values()
        return [self.deltas[f] for f, fflag in self.flags.items() if fflag & flag == fflag]

    def get_flag(self, feature):
        return self.flags.get(feature, self.NO_DELTA)

    def set_flag(self, feature, flag):
        self.flags[feature] = flag

    def size(self, flag=None):
        if flag is not None:
            return self.flags.values().count(flag)
        return len(self.deltas)

    def copy(self):
        new = DeltaInformation([], self.average_tol)
        new.deltas = self.deltas.copy()
        new.flags = self.flags.copy()
        return new


class FeatureFilter(nputils.AbstractFilter):

    def filter(self, feature):
        raise NotImplementedError()


class MaskFilter(FeatureFilter):

    def __init__(self, img, coord_mode='com', prj_settings=None):
        self.img = img
        self.coord_mode = coord_mode
        if prj_settings is None:
            prj_settings = imgutils.ProjectionSettings(relative=True, unit=u.mas)
        self.prj_settings = prj_settings

    def __str__(self):
        return "MaskFilter(%s)" % (self.img)

    def filter(self, feature):
        if self.img is None:
            return True
        
        feature_prj = feature.get_coordinate_system().get_projection(self.prj_settings)
        mask_prj = self.img.get_projection(self.prj_settings)
        coord = feature.get_coord(mode=self.coord_mode)
        feature_sky_coord = feature_prj.p2s(p2i(coord))
        feature_pixel_mask_coord = np.round(p2i(mask_prj.s2p(feature_sky_coord)))

        if not nputils.check_index(self.img.data, *feature_pixel_mask_coord):
            return False

        return bool(self.img.data[tuple(feature_pixel_mask_coord.astype(int))])


class DateFilter(FeatureFilter):

    def __init__(self, start_date=None, end_date=None, filter_dates=None):
        self.filter_fct = nputils.date_filter(start_date=start_date, end_date=end_date, 
            filter_dates=filter_dates)

    @staticmethod
    def from_filter_fct(self, filter_fct):
        new = DateFilter()
        new.filter_fct = filter_fct
        return new

    def filter(self, feature):
        return self.filter_fct(feature.get_epoch())


class DfcFilter(FeatureFilter):

    def __init__(self, dfc_min, dfc_max, unit, coord_mode='com'):
        self.dfc_min = dfc_min
        self.dfc_max = dfc_max
        self.prj_settings = imgutils.ProjectionSettings(relative=True, unit=unit)
        self.coord_mode = coord_mode

    def __str__(self):
        return "DfcFilter(%s, %s)" % (self.dfc_min, self.dfc_max)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['prj_settings']
        return state

    def filter(self, feature):
        feature_prj = feature.get_coordinate_system().get_projection(self.prj_settings)
        dfc = feature_prj.dfc(p2i(feature.get_coord(mode=self.coord_mode)))
        res = bool(dfc >= self.dfc_min and dfc <= self.dfc_max)
        logger.debug("DfcFilter: feature=%s, dfc=%s, res=%s" % (feature, dfc, res))
        return res


class PaFilter(FeatureFilter):

    def __init__(self, pa_min, pa_max, pa_fct=None, coord_mode='com'):
        self.pa_min = pa_min
        self.pa_max = pa_max
        self.prj_settings = imgutils.ProjectionSettings(relative=True)
        self.pa_fct = pa_fct
        self.coord_mode = coord_mode

    def __str__(self):
        return "PaFilter(%s, %s)" % (self.pa_min, self.pa_max)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['prj_settings']
        return state

    def filter(self, feature):
        feature_prj = feature.get_coordinate_system().get_projection(self.prj_settings)
        pa = feature_prj.pa(p2i(feature.get_coord(mode=self.coord_mode)))
        if self.pa_fct is not None:
            pa = self.pa_fct(feature, pa)
        res = bool(pa >= self.pa_min and pa <= self.pa_max)
        logger.debug("DfcFilter: feature=%s, pa=%s, res=%s" % (feature, pa, res))
        return res


class RegionFilter(FeatureFilter):

    def __init__(self, region, coord_mode='com'):
        self.region = region
        self.coord_mode = coord_mode
        self.cache = nputils.LimitedSizeDict(size_limit=100)

    def __str__(self):
        return "RegionFilter(%s)" % (os.path.basename(self.region.get_filename()))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['cache']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.cache = nputils.LimitedSizeDict(size_limit=100)

    def filter(self, feature):
        if feature not in self.cache:
            feature_cs = feature.get_coordinate_system()
            region = self.region.get_pyregion(feature_cs)
            y, x = feature.get_coord(mode=self.coord_mode)
            self.cache[feature] = bool(region.get_filter().inside1(x, y))
        res = self.cache[feature]
        logger.debug("RegionFilter: feature=%s, region=%s, res=%s" % (feature, self.region.get_name(), res))
        return res


class RegionsList(list):

    def __init__(self, all_regions):
        self.region_filters = dict()
        for region in all_regions:
            if not isinstance(region, imgutils.Region):
                filt = reduce(lambda x, y: x | y, [RegionFilter(k) for k in region])
            else:
                filt = RegionFilter(region)
            self.append(region)
            self.region_filters[region] = filt

    def get_region(self, feature):
        for region in self:
            region_filter = self.region_filters[region]
            if region_filter.filter(feature):
                return region
        return None


class IntensityFilter(nputils.AbstractFilter):

    def __init__(self, intensity_min=-np.inf, intensity_max=np.inf):
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def filter(self, feature):
        return bool(feature.get_intensity() > self.intensity_min and feature.get_intensity() < self.intensity_max)


class AbstractDeltaRangeFilter(nputils.AbstractFilter):

    def filter(self, delta):
        raise NotImplementedError()


class DeltaRangeFilter(AbstractDeltaRangeFilter):

    def __init__(self, vxrange=None, vyrange=None, normrange=None, unit=None, pix_limit=2, x_dir=[1, 0]):
        self.vxrange = vxrange
        self.vyrange = vyrange
        self.normrange = normrange
        self.x_dir = x_dir
        self.unit = unit
        self.pix_limit = pix_limit
        self.prj_settings = imgutils.ProjectionSettings(relative=True, unit=u.mas)

    def __str__(self):
        d = (self.vxrange, self.vyrange, self.normrange, self.unit, self.pix_limit, self.x_dir)
        return "DeltaRangeFilter(vx=%s, vy=%s, v=%s, u=%s, pix=%s, dir=%s)" % d

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['prj_settings']
        state['unit'] = str(self.unit)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.unit = u.Unit(self.unit)
        self.prj_settings = imgutils.ProjectionSettings(relative=True, unit=u.mas)

    def filter(self, delta):
        feature = delta.get_feature()
        prj = feature.get_coordinate_system().get_projection(self.prj_settings)
        v = [k.to(self.unit).value for k in delta.get_angular_velocity_vector(prj)]
        x_dir = self.x_dir
        if x_dir == 'position_angle':
            x_dir = prj.p2s(p2i(feature.get_coord()))
        vx, vy = nputils.vector_projection(v, x_dir)
        dy, dx = delta.get_delta()
        vn = np.linalg.norm([vx, vy])
        dn = np.linalg.norm([dx, dy])
        res = True
        for v_range, value, v_pix in zip([self.vxrange, self.vyrange, self.normrange], [vx, vy, vn], [dx, dy, dn]):
            if np.abs(v_pix) < self.pix_limit:
                continue
            if v_range is not None and not nputils.in_range(value, v_range):
                res = False
                break
        logger.debug("DeltaRangeFilter: delta=%s, v=%s, vp=%s, res=%s" % (delta.get_delta(), v, [vx, vy], res))
        return res


class DeltaRegionFilter(AbstractDeltaRangeFilter):

    def __init__(self, region_filter, range_filter, coord_mode='com'):
        self.region_filter = region_filter
        self.range_filter = range_filter

    def __str__(self):
        return "(%s: %s)" % (self.region_filter, self.range_filter)


    def filter(self, delta):
        feature = delta.get_feature()
        if self.region_filter.filter(feature):
            return self.range_filter.filter(delta)
        return True

