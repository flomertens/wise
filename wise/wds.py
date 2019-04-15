import numpy as np
from skimage.morphology import watershed
from scipy.ndimage import measurements, gaussian_filter

from libwise import nputils, imgutils, wtutils, wavelets
from libwise.nputils import validator_is, is_callable, validator_in_range, str2jsonfunction
from libwise.nputils import validator_list, validator_is_class, str2bool, str2jsonclass

import jsonpickle as jp

from features import *

p2i = imgutils.p2i
logger = logging.getLogger(__name__)


def build_image(segments, delta_info=None):
    builder = imgutils.ImageBuilder()
    for segment in segments:
        region = segment.get_image_region()
        if delta_info is not None:
            shift = delta_info.get_delta(segment)
            region.set_shift(shift)
        builder.add(region)
    return builder.get()


class Segment(ImageFeature):

    def __init__(self, coord, intensity, segmentid, segmented_image, inner_features=None):
        snr = intensity / segmented_image.get_rms_noise()
        meta = segmented_image.get_img_meta()
        ImageFeature.__init__(self, coord, meta, intensity, snr)
        self.segmentid = segmentid
        self.inner_features = inner_features
        self.segmented_image = segmented_image

        # cache
        # self.center_of_shape = None
        # self.center_of_mass = None
        self.area = None
        self.crop_index = None
        self.image_region = None
        self.interface = None

        # if isinstance(self.segmented_image, SegmentedScale):
        #     k = self.initial_coord.tolist() + [self.segmentid, self.segmented_image.get_scale() + self.get_epoch()]
        # else:
        #     k = self.initial_coord.tolist() + [self.segmentid, id(self.segmented_image)]
        self.__hash = hash(tuple(self.initial_coord.tolist() + [self.segmentid, id(self.segmented_image)]))

    def __str__(self):
        return "Segment(id:%s, c:%s, i:%s, snr:%s)" % (self.segmentid, self.coord, self.get_intensity(), self.get_snr())

    def get_epoch(self):
        return self.get_segmented_image().get_epoch()

    def get_id(self):
        epoch = self.get_epoch()
        if isinstance(epoch, datetime.date):
            epoch = epoch.strftime("%Y%m%d")
        if self.inner_features is not None and self.inner_features.size() > 0:
            segid = '+'.join([str(k.get_segmentid()) for k in self.inner_features.get_features()])
        else:
            segid = self.get_segmentid()
        return '%s_%s' % (epoch, segid)

    def add_inner_feature(self, feature):
        if self.inner_features is None:
            self.inner_features = FeaturesGroup()
        self.inner_features.add_feature(feature)

    def get_inner_features(self):
        if self.inner_features is None:
            return []
        return self.inner_features

    def has_inner_features(self):
        if self.inner_features is None:
            return False
        return self.inner_features.size() > 0

    def get_segmentid(self):
        return self.segmentid

    def get_segmented_image(self):
        return self.segmented_image

    def get_mask(self):
        return self.segmented_image.get_mask(self)

    def get_segment_image(self):
        image = self.segmented_image.get_img().data.copy()
        labels = self.segmented_image.get_labels()
        image[labels != self.get_segmentid()] = 0
        return image

    def get_cropped_segment_image(self):
        s = nputils.index2slice(self.get_cropped_index())
        image = self.segmented_image.get_img().data[s].copy()
        labels = self.segmented_image.get_labels()
        image[labels[s] != self.get_segmentid()] = 0
        return image

    def get_image_region(self):
        if self.image_region is None:
            shape = self.segmented_image.get_img().data.shape
            self.image_region = imgutils.ImageRegion(self.get_cropped_segment_image(),
                                                     self.get_cropped_index(), cropped=True, shape=shape)
        return self.image_region

    def get_center_of_shape(self):
        return self.segmented_image.get_center_of_shape(self)

    def get_area(self):
        if self.area is None:
            self.area = self.get_mask().sum()
        return self.area

    def get_center_of_mass(self):
        return self.segmented_image.get_center_of_mass(self)

    def get_coord(self, mode=None):
        if mode is None or mode is 'lm':
            coord = Feature.get_coord(self)
        elif mode is 'com':
            coord = self.get_center_of_mass()
        elif mode is 'cos':
            coord = self.get_center_of_shape()
        return coord

    def get_total_intensity(self):
        labels = self.segmented_image.get_labels()
        img = self.segmented_image.get_img().data
        return measurements.sum(img, labels, self.segmentid)

    def get_cropped_index(self):
        if self.crop_index is None:
            nu, self.crop_index = nputils.crop_threshold(self.get_mask(), output_index=True)
        return self.crop_index

    def get_interface(self):
        if self.interface is None:
            segments = self.segmented_image
            i = self.get_cropped_index()
            # crop the labels, adding just a small border to get the interface
            index = [max(0, i[0] - 1), max(0, i[1] - 1), i[2] + 1, i[3] + 1]
            labels = segments.get_labels()[nputils.index2slice(index)]
            interface = nputils.get_interface(labels, self.segmentid)
            self.interface = dict([(segments.get_segment_from_id(k), v) for k, v in interface.items()])
        return self.interface

    def get_connected_segments(self):
        return FeaturesGroup(self.get_interface().keys())

    def is_connected(self, segment):
        return segment in self.get_connected_segments()

    def distance(self, other, mode='local_max'):
        self_delta_to_initial = self.get_coord() - self.initial_coord
        other_delta_to_initial = other.get_coord() - other.initial_coord
        if mode == 'local_max' or mode is None:
            delta = self.get_coord() - other.get_coord()
        elif mode == 'center_of_mass':
            delta = (self.get_center_of_mass() + self_delta_to_initial) \
                - (other.get_center_of_mass() + other_delta_to_initial)
        elif mode == 'center_of_shape':
            delta = (self.get_center_of_shape() + self_delta_to_initial) \
                - (other.get_center_of_shape() + other_delta_to_initial)
        elif mode == 'min':
            delta_coord = np.linalg.norm(self.get_coord() - other.get_coord())
            delta_com = np.linalg.norm(
                (self.get_center_of_mass() + self_delta_to_initial) - (other.get_center_of_mass() + other_delta_to_initial))
            delta_cos = np.linalg.norm(
                (self.get_center_of_shape() + self_delta_to_initial) - (other.get_center_of_shape() + other_delta_to_initial))
            return min([delta_coord, delta_com, delta_cos])
        return np.linalg.norm(delta)

    def in_region(self, region):
        return (self.get_mask() * region.get_mask()).sum() > 0

    def discard_cache(self):
        self.area = None
        self.crop_index = None
        self.image_region = None
        self.interface = None

    def copy(self):
        new = Segment(self.coord, self.intensity,
                      self.segmentid, self.segmented_image)
        if self.has_inner_features():
            new.inner_features = self.inner_features.copy()
        new.initial_coord = self.initial_coord
        return new


class SegmentedImages(DatedFeaturesGroup):

    def __init__(self, img, features=[], labels=None, rms_noise=0):
        self.img = img
        self.img_meta = img.get_meta()
        self.rms_noise = rms_noise
        self.labels = labels
        self.ids = dict(zip([f.get_segmentid() for f in features], features))
        DatedFeaturesGroup.__init__(self, features, self.get_img().get_epoch())

        self.center_of_shape = None
        self.center_of_mass = None

    def add_feature(self, segment):
        FeaturesGroup.add_feature(self, segment)
        self.ids[segment.get_segmentid()] = segment

    def watershed_segmentation(self, features, mask, feature_filter=None):
        markers = np.zeros_like(self.img.data, dtype=np.int16)

        cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())
        features = features.sorted_list(cmp=cmp_intensity)[::-1]

        for i, feature in enumerate(features):
            markers[tuple(feature.get_coord().astype(int))] = i + 2

        self.labels = watershed(- self.img.data, markers, mask=mask)
        self.labels[self.labels == 1] = 0

        for i, feature in enumerate(features):
            id = i + 2

            segment = Segment(feature.get_coord(), feature.get_intensity(), id, self)
            if feature_filter is not None and feature_filter.filter(segment) is False:
                self.labels[self.labels == id] = 0
                continue

            self.add_feature(segment)

    def connected_structure(self, structure=None, mask=None, feature_filter=None):
        img = self.img.data
        if mask is not None:
            img = img * mask
        self.labels, n = measurements.label(img, structure=structure)
        self.labels += 1

        for id in range(2, n + 2):
            coord = measurements.maximum_position(img, self.labels, id)
            intensity = img[tuple(coord)]
            segment = Segment(coord, intensity, id, self)

            if feature_filter is not None and feature_filter.filter(segment) is False:
                self.labels[self.labels == id] = 0
                continue

            self.add_feature(segment)

    def get_rms_noise(self):
        return self.rms_noise

    def get_all_inner_features(self):
        group = FeaturesGroup()
        for feature in self.get_features():
            if feature.has_inner_features():
                group.add_features_group(feature.get_inner_features())
        return group

    def merge_segments(self, segment1, segment2):
        if not self.has_feature(segment1) or not self.has_feature(segment2):
            logger.warning("Merge issue: current:%s, one:%s, two:%s" % (self, segment1, segments2))
        logger.info("Merging %s and %s" % (segment1, segment2))

        id1 = segment1.get_segmentid()
        id2 = segment2.get_segmentid()

        # id1, id2 = sorted([id1, id2])

        # get our own version of f1 and f2
        segment1 = self.get_segment_from_id(id1)
        segment2 = self.get_segment_from_id(id2)

        # change labels
        self.labels[self.labels == id2] = id1

        # merge inner features from each segments
        if segment2.has_inner_features():
            for feature in segment2.get_inner_features():
                segment1.add_inner_feature(feature.copy())
        else:
            segment1.add_inner_feature(segment2)
        if not segment1 in segment1.get_inner_features():
            segment1.add_inner_feature(segment1.copy())

        # get new local max
        feature_max_intensity = max(segment1, segment2, key=lambda k: k.get_intensity())

        # TODO: intensity

        # keep corrdinate of the first segment
        segment1.set_coord(feature_max_intensity.get_coord())
        segment1.initial_coord = feature_max_intensity.initial_coord

        self.discard_cache(segment1)

        # remove segment2
        self.remove_feature(segment2)

        return segment1

    def get_segments_inside(self, segment):
        result = FeaturesGroup()
        # print "Finding segments in", segment, id(segment)
        for feature in self.get_features():
            upper_segment = segment.get_segmented_image().get_segment_from_feature(feature)
            # print feature, ", upper:", upper_segment, id(upper_segment)
            if upper_segment is not None and upper_segment == segment:
                result.add_feature(feature)
        return result

    def get_overlapping_segment(self, segment, delta=None, min_ratio=0.4):
        mask1 = segment.get_segmented_image().get_mask(segment)
        if delta is not None:
            mask1 = nputils.shift2d(mask1, delta)
        count = np.bincount((self.labels * mask1).flatten().astype(np.int))
        count[0] = 0
        imax = np.argmax(count)
        ratio = count.flatten()[imax] / float(mask1.sum())

        if ratio >= min_ratio:
            return self.get_segment_from_id(imax)
        return None

    def get_segments_inside2(self, segment, delta=None, ratio=None):
        mask = segment.get_mask()
        if delta is not None:
            mask = nputils.shift2d(mask, delta)
        result = FeaturesGroup()
        # print "Look for segments inside", segment
        for id, count in nputils.count((self.labels * mask).flatten()):
            segment_inside = self.get_segment_from_id(id)
            if segment_inside is not None:
                # print "->", segment_inside, count / float(segment_inside.get_area()), count / float(segment.get_area())
                if ratio is not None:
                    ratio_inside = max(count / float(segment_inside.get_area()), count / float(segment.get_area()))
                    if ratio_inside < ratio:
                        continue
                result.add_feature(segment_inside)
        return result

    def remove_feature(self, feature, remove_label=False):
        FeaturesGroup.remove_feature(self, feature)
        if remove_label is True:
            id = feature.get_segmentid()
            self.img.data[self.labels == id] = 0
            self.labels[self.labels == id] = 0

    def get_img(self):
        return self.img

    def get_img_meta(self):
        return self.img_meta

    def get_segments_img(self):
        img = self.img.data.copy()
        img[self.labels <= 0] = 0
        return img

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        self.labels = labels

    def get_mask(self, segment):
        return self.labels == segment.get_segmentid()

    def get_segment_from_id(self, segmentid):
        return self.ids.get(segmentid, None)

    def get_segment_from_feature(self, feature, mode='lm'):
        x, y = feature.get_coord(mode=mode)
        return self.get_segment_from_coord(x, y)

    def get_segment_from_coord(self, x, y):
        if not nputils.check_index(self.labels, x, y):
            return None
        return self.get_segment_from_id(self.labels[int(x), int(y)])

    def region_filter(self, region):
        for feature in self.features[:]:
            # if not feature.in_region(region):
            x, y = feature.get_coord()
            if region.get_mask()[x, y] == 0:
                self.remove_feature(feature)

    def filter_out_distance_from_border(self, d):
        for feature in self.get_features():
            if min(nputils.distance_from_border(feature.get_coord(), self.img.data.shape)) <= d:
                self.remove_feature(feature)

    def filter_out_not_in(self, set):
        for feature in self.get_features():
            if feature not in set:
                self.remove_feature(feature)

    def filter_out_snr(self, min_snr):
        for feature in self.get_features():
            if feature.get_snr() < min_snr:
                self.remove_feature(feature)

    def get_values(self, coords):
        xs, ys = np.array(coords).T
        return self.img.data[xs, ys]

    def get_segmentids(self):
        return self.ids.keys()

    def get_center_of_mass(self, segment):
        if self.center_of_mass is None:
            labels = self.get_labels()
            img = self.get_img()
            # ids = self.get_segmentids()
            ids = np.unique(labels)
            coms = measurements.center_of_mass(img.data, labels, ids)
            self.center_of_mass = dict(zip(ids, np.array(coms)))
        return self.center_of_mass.get(segment.get_segmentid(), segment.get_coord(mode="lm"))

    def get_center_of_shape(self, segment):
        if self.center_of_shape is None:
            labels = self.get_labels()
            # ids = self.get_segmentids()
            ids = np.unique(labels)
            coss = measurements.center_of_mass(labels, labels, ids)
            self.center_of_shape = dict(zip(ids, np.array(coss)))
        return self.center_of_shape.get(segment.get_segmentid(), segment.get_coord(mode="lm"))

    def discard_cache(self, segment=None):
        if segment is None:
            for segment in self.get_features():
                segment.discard_cache()
        else:
            segment.discard_cache()
        self.center_of_shape = None
        self.center_of_mass = None

    def copy(self):
        # MEM ISSUE: we should avoid having to copy the labels
        new = SegmentedImages(self.img, [k.copy() for k in self.features], self.labels.copy(), self.rms_noise)
        for feature in new.get_features():
            feature.segmented_image = new
        for feature in new.get_all_inner_features():
            feature.segmented_image = new
        return new


class AbstractScale(object):

    def __init__(self, scale):
        self.scale = scale

    def set_scale(self, scale):
        self.scale = scale

    def get_scale(self, projection=None):
        if projection is not None:
            return projection.mean_pixel_scale() * self.scale
        return self.scale


class SegmentedScale(AbstractScale, SegmentedImages):

    def __init__(self, img, features, labels, rms_noise, original_image, scale):
        SegmentedImages.__init__(self, img, features, labels, rms_noise)
        AbstractScale.__init__(self, scale)
        self.original_image = original_image

    def get_original_image(self):
        return self.original_image

    def get_feature_snr(self, segment):
        return measurements.maximum(self.original_image.data, self.labels, segment.get_segmentid()) / self.rms_noise

    def copy(self):
        new = SegmentedScale(self.img, [k.copy() for k in self.features], self.labels.copy(), self.rms_noise,
                             self.original_image, self.scale)
        for feature in new.get_features():
            feature.segmented_image = new
        return new


class DatedFeaturesGroupScale(AbstractScale, DatedFeaturesGroup):

    def __init__(self, scale, features=None, epoch=0):
        DatedFeaturesGroup.__init__(self, features, epoch=epoch)
        AbstractScale.__init__(self, scale)

    def copy(self):
        return DatedFeaturesGroupScale(self.scale, [k.copy() for k in self.features], epoch=self.epoch)


class FinderConfiguration(nputils.BaseConfiguration):

    def __init__(self):
        data = [
            ["alpha_threashold", 3, "Significance threshold", validator_in_range(0.1, 20), float, str, 0],
            ["alpha_detection", 4, "Detection threshold", validator_in_range(0.1, 20), float, str, 0],
            ["min_scale", 1, "Minimum Wavelet scale", validator_in_range(0, 10, instance=int), int, str, 0],
            ["max_scale", 4, "Maximum Wavelet scale", validator_in_range(1, 10, instance=int), int, str, 0],
            ["scales_snr_filter", None, "Per scales detection threshold", validator_is(dict), jp.decode, jp.encode, 1],
            ["ms_dec_klass", WaveletMultiscaleDecomposition, "Multiscale decompostion class",
             validator_is_class(AbstractMultiScaleDecomposition), lambda s: jp.decode(str2jsonclass(s)), jp.encode, 1],
            ["use_iwd", False, "Use Intermediate Wavelet Decomposition", validator_is(bool), str2bool, str, 0],
            ["dec", wtutils.uiwt, "Multiscale decompostion class", is_callable,
                lambda s: jp.decode(str2jsonfunction(s)), jp.encode, 1],
            ["wd_wavelet", 'b1', "Wavelet to use for the Wavelet Decomposition", validator_is(str), str, str, 1],
            ["iwd_wavelet", 'b3', "Wavelet to use for the Intermediate Wavelet Decomposition",
                validator_is(str), str, str, 1],
            ["dog_step", True, "DOG", validator_is(int), None, None, 2],
            ["dog_angle", True, "DOG", validator_is((int, float)), None, None, 2],
            ["dog_ellipticity", True, "DOG", validator_is((int, float)), None, None, 2],
            ["exclude_border_dist", 1, "Number of pixel from border to exclude", validator_is(int), int, str, 0],
            ["exclude_noise", True, "Include coefficients below threshold in resulting image",
                validator_is(bool), str2bool, str, 1],
        ]

        super(FinderConfiguration, self).__init__(data, title="Finder configuration")


class Node(object):

    def __init__(self, obj, parent, childs=[]):
        self.parent = parent
        self.childs = []
        self.obj = obj

    def get_id(self):
        if self.is_root():
            return "root"
        return "%s:%s" % (self.obj.get_segmented_image().get_scale(), self.obj.get_segmentid())

    def __str__(self):
        return "Node(%s)" % self.get_id()

    def show(self, level):
        print " " * level + "\-- %s" % self.get_id()
        for child in self.childs:
            child.show(level + 1)

    def get(self):
        return self.obj

    def is_root(self):
        return self.parent == None

    def add_child(self, child):
        self.childs.append(child)

    def get_parent(self):
        return self.parent

    def has_parent(self):
        return self.parent is not None

    def get_childs(self):
        return self.childs

    def walk_up(self):
        yield self.parent
        if self.parent is not None:
            self.parent.walk_up()


class MultiScaleNode(list):

    def __init__(self, first_node):
        list.__init__(self)
        self.append(first_node)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return "MSNode(" + ", ".join([node.get_id() for node in self]) + ")"

    def get_scales(self):
        return [node.get().get_segmented_image().get_scale() for node in self]


class MultiScaleTree(object):

    def __init__(self):
        self.root = Node(None, None, [])
        self.nodes = dict()
        self.ms_nodes = dict()

    def show(self):
        for node in self.root.get_childs():
            print "\n"
            node.show(0)
            print "\n"

    def add(self, obj, parent=None):
        if parent is None:
            parent = self.root
        node = Node(obj, parent)
        parent.add_child(node)
        self.nodes[obj] = node
        if parent == self.root:
            self.add_ms_node(node)
        return node

    def add_ms_node(self, node):
        self.ms_nodes[node] = MultiScaleNode(node)
        return self.ms_nodes[node]

    def get_node(self, obj):
        return self.nodes[obj]

    def get_ms_node(self, node):
        return self.ms_nodes.get(node, None)


class MultiScaleRelation(object):

    def __init__(self, ms_image):
        self.ms_image = ms_image
        self.relations = dict([(k, dict()) for k in self.ms_image])
        self.build()

    def build(self):
        for (segments1, segments2) in nputils.pairwise(self.ms_image[::-1]):
            dict_relation = self.relations[segments1]
            for feature1 in segments1:
                dict_relation[feature1] = []
                for feature2 in segments2:
                    seg1_f2 = segments1.get_segment_from_feature(feature2)
                    if seg1_f2 is not None and seg1_f2 == feature1:
                        dict_relation[feature1].append(feature2)

    def get_relations(self, segment):
        segmented_image = segment.get_segmented_image()
        if len(self.relations[segmented_image].keys()) == 0:
            return []
        return self.relations[segmented_image][segment]


class AbstractKeyList(list):

    def get_item_key(self, item):
        return NotImplemented()

    def get_keys(self):
        return map(self.get_item_key, self)

    def get_key(self, key):
        for item in self:
            if self.get_item_key(item) == key:
                return item
        return None

    def merge(self, other):
        this_keys = self.get_keys()
        other_keys = other.get_keys()
        keys = set(this_keys + other_keys)

        for key in keys:
            if key not in this_keys:
                self.append(other.get_key(key))
            elif key in other_keys:
                self.get_key(key).merge(other.get_key(key))


class BaseMultiScaleImage(AbstractKeyList):

    def __init__(self, epoch):
        self.epoch = epoch

    def get_item_key(self, item):
        return item.get_scale()

    def get_scales(self):
        return self.get_keys()

    def get_scale(self, scale):
        return self.get_key(scale)

    def features_iter(self):
        for segments in self:
            for segment in segments:
                yield segment

    def get_epoch(self):
        return self.epoch


class MultiScaleImage(BaseMultiScaleImage):

    def __init__(self, original_img, rms_noise, reversable=False, approx=None):
        self.original_img = original_img
        if isinstance(rms_noise, np.ndarray):
            rms_noise = rms_noise.std()
        self.rms_noise = rms_noise
        self.reversable = reversable
        self.approx = approx
        BaseMultiScaleImage.__init__(self, self.original_img.get_epoch())

    def get_shape(self):
        return self.original_img.data.shape

    def get_rms_noise(self):
        return self.rms_noise

    def get_original_img(self):
        return self.original_img

    def build_tree(self):
        tree = MultiScaleTree()

        # start from coarser scale and go down
        for segment in self[-1]:
            tree.add(segment)

        for (parent_segments, child_segments) in nputils.pairwise(self[::-1]):
            for segments in child_segments:
                parent = parent_segments.get_segment_from_feature(segments)
                if parent is None:
                    parent_node = None
                else:
                    parent_node = tree.get_node(parent)
                tree.add(segments, parent=parent_node)

            # second pass to fill ms_nodes
            for segments in child_segments:
                node = tree.get_node(segments)
                parent_node = node.get_parent()
                n_parent_childs = len(parent_node.get_childs())
                if n_parent_childs == 1 and not parent_node.is_root():
                    parent_ms_node = tree.get_ms_node(parent_node)
                    parent_ms_node.append(node)
                    tree.ms_nodes[node] = parent_ms_node
                else:
                    tree.add_ms_node(node)

        return tree

    def get_segmented_scale(self, scale):
        ''' Not optimized'''
        for segments in self:
            if segments.get_scale() == scale:
                return segments

    def get_combined_representation(self):
        # Experimental
        # TODO: make use of MSTree instead
        ms_relation = MultiScaleRelation(self)
        img = imgutils.Image.from_image(self.original_img)
        labels = np.zeros_like(self.original_img.data)

        best = SegmentedImages(img, None, labels, 1)

        i = 2
        for segments in self:
            segments_img = segments.get_img()
            for segment in segments:
                if len(ms_relation.get_relations(segment)) == 0:
                    mask = segments.get_mask(segment)
                    segment_img = segments_img.data[mask == 1] / segments.get_rms_noise()
                    img.data[mask == 1] += segment_img
                    labels[mask == 1] = i
                    new_segment = Segment(segment.get_coord(), segment.get_intensity(), i, best)
                    best.add_feature(new_segment)
                    new_segment.scale = segments.get_scale()

                    i += 1

        return best

    def recompose(self, delta_fct=None):
        '''This assume decomposition was done using WaveletMultiscaleDecomposition'''
        img = np.zeros_like(self.original_img.data)
        for segments in self:
            if delta_fct is not None:
                builder = imgutils.ImageBuilder()
                for segment in segments:
                    region = segment.get_image_region()
                    if delta_fct is not None:
                        shift = np.round(delta_fct(segment.get_coord()))
                        region.set_shift(shift)
                    builder.add(region)
                    del region
                segs_img = builder.get().get_data()
                img += segs_img
            else:
                # print segments.img.data
                img += segments.img.data

        if self.approx is not None:
            img += self.approx

        return imgutils.Image.from_image(self.original_img, img)

    def save_to_fits(self, filename):
        pass

    def load_from_fits(self, filename):
        pass

    def copy(self):
        new = MultiScaleImage(self.original_img, self.rms_noise)
        for k in self:
            new.append(k.copy())
        return new


class MultiScaleImageSet(AbstractKeyList):

    def get_item_key(self, item):
        return item.get_epoch()

    def get_epochs(self):
        return self.get_keys()

    def get_epoch(self, epoch):
        return self.get_key(epoch)

    def get_scales(self):
        if len(self) == 0:
            return []
        return sorted(self[0].get_scales())

    def features_iter(self):
        for ms_segments in self:
            for segments in ms_segments:
                for segment in segments:
                    yield segment

    def is_full_wds(self):
        try:
            first_segment = self.features_iter().next()
        except StopIteration:
            return False
        return isinstance(first_segment, Segment)

    def to_file_full(self, projection, image_set):
        pass

    def to_file(self, filename, projection, coord_mode='com'):
        '''Format is: epoch, x, y, intensity, snr, scale'''
        l = []
        for ms_segments in self:
            epoch = float(nputils.datetime_to_epoch(ms_segments.get_epoch()))
            for segments in ms_segments:
                scale = projection.mean_pixel_scale() * segments.get_scale()
                for feature in segments:
                    x, y = projection.p2s(p2i(feature.get_coord(mode=coord_mode)))
                    intensity = feature.get_intensity()
                    snr = feature.get_snr()
                    l.append([epoch, x, y, intensity, snr, scale])

        unit = projection.unit
        header = 'WISE Features lists\n'
        header += 'Epoch, X (%s), Y (%s), Intensity, SNR, Scale (%s)\n' % (unit, unit, unit)

        np.savetxt(filename, l, ["%f", "%.5f", "%.5f", "%.6f", "%.6f", "%f"],
                   delimiter=' ', header=header)
        print "Saved MultiScaleImageSet @ %s" % filename

    @staticmethod
    def from_file_full(self, projection, image_set):
        pass

    @staticmethod
    def from_file(file, projection, image_set, feature_filter=None):
        '''Format is: epoch, x, y, intensity, snr, scale'''
        new = MultiScaleImageSet()
        array = np.loadtxt(file, dtype=str, delimiter=' ')
        epochs = dict()
        img_metas = dict()
        cs = projection.get_coordinate_system()
        for line in array:
            date = nputils.epoch_to_datetime(line[0])
            x, y = projection.s2p(map(float, line[1:3]))
            intensity = float(line[3])
            snr = float(line[4])
            if date not in img_metas:
                img_metas[date] = imgutils.ImageMeta(date, cs, image_set.get_beam(date))
            feature = ImageFeature([y, x], img_metas[date], intensity, snr)
            if feature_filter is not None and not feature_filter(feature):
                continue
            scale = np.round(float(line[5]) / projection.mean_pixel_scale())
            if not date in epochs:
                epochs[date] = dict()
            ms_features = epochs[date]
            if not scale in ms_features:
                ms_features[scale] = DatedFeaturesGroupScale(scale, epoch=date)
            ms_features[scale].add_feature(feature)

        for epoch, scales in epochs.items():
            ms_features = BaseMultiScaleImage(epoch)
            ms_features.extend(scales.values())
            new.append(ms_features)

        print "Loaded MultiScaleImageSet from %s" % file
        return new


class AbstractMultiScaleDecomposition(object):

    reversable = False

    def __init__(self, img, bg, config):
        self.img = img
        self.bg = bg
        self.config = config
        self.approx = None

    def decompose(self):
        pass


class WaveletMultiscaleDecomposition(AbstractMultiScaleDecomposition):

    reversable = True

    def decompose(self, wavelet_fct=None, img=None, bg=None, min_scale=None, max_scale=None):
        if wavelet_fct is None:
            wavelet_fct = self.config.get("wd_wavelet")
        if img is None:
            img = self.img
        if bg is None:
            bg = self.bg
        if min_scale is None:
            min_scale = self.config.get("min_scale")
        if max_scale is None:
            max_scale = self.config.get("max_scale")
        wt_dec = self.config.get("dec")
        if max_scale is None:
            max_scale = wavelets.get_wavelet(wavelet_fct).get_max_level(img.data)

        scales = wtutils.wavedec(img.data, wavelet_fct, max_scale, dec=wt_dec)
        self.approx = scales[-1]
        scales = scales[min_scale:-1]
        scales = [nputils.resize_like(s, img.data) for s in scales]

        scales_noise = wtutils.wave_noise_factor(bg, wavelet_fct, max_scale, wt_dec, beam=img.get_beam())
        scales_noise = scales_noise[min_scale:]

        if wavelet_fct in ['b3', 'triangle2']:
            scales_width = [max(1.5, 3 * min(1, j) * pow(2, max(0, j - 1))) for j in range(min_scale, max_scale)]
        else:
            scales_width = [max(1, 2 * min(1, j) * pow(2, max(0, j - 1))) for j in range(min_scale, max_scale)]

        return zip(scales, scales_noise, scales_width)


class InterscalesWaveletMultiscaleDecomposition(WaveletMultiscaleDecomposition):

    reversable = False

    def decompose(self):
        b1 = WaveletMultiscaleDecomposition.decompose(self, self.config.get("wd_wavelet"))
        b3 = WaveletMultiscaleDecomposition.decompose(self, self.config.get("iwd_wavelet"))

        return nputils.roundrobin(b1, b3)


class WavePacketMultiscaleDecomposition(WaveletMultiscaleDecomposition):

    reversable = False

    def decompose(self):
        level1 = WaveletMultiscaleDecomposition.decompose(self)
        levels = level1
        for scale, scale_noise, width in level1:
            level2 = WaveletMultiscaleDecomposition.decompose(self, img=imgutils.Image(scale), bg=scale_noise,
                                                              min_scale=1, max_scale=2)
            levels = nputils.roundrobin(levels, level2)

        return levels


class DoGMultiscaleDecomposition(WaveletMultiscaleDecomposition):

    reversable = False

    def decompose(self):
        min_scale = self.config.get("min_scale")
        max_scale = self.config.get("max_scale")

        step = self.config.get("dog_step")
        angle = self.config.get("dog_angle")
        ellipticity = self.config.get("dog_ellipticity")

        if max_scale is None:
            max_scale = min(self.img.data.shape) / 4

        if angle is 'beam' and self.img.has_beam():
            angle = img.get_beam().angle

        widths = np.arange(min_scale, max_scale + 2 * step, step)

        scales = wtutils.dogdec(self.img.data, widths=widths, angle=angle, ellipticity=ellipticity, boundary="symm")
        scales = [nputils.resize_like(s, self.img.data) for s in scales]

        scales_noises = wtutils.dog_noise_factor(self.bg, widths=widths, angle=angle,
                                                 ellipticity=ellipticity, beam=self.img.get_beam())

        return zip(scales, scales_noises, widths)


class MinScaleMultiscaleDecomposition(WaveletMultiscaleDecomposition):

    reversable = False

    def decompose(self):
        min_scale = int(2 ** self.config.get("min_scale"))
        max_scale = int(2 ** self.config.get("max_scale"))

        step = self.config.get("dog_step")
        angle = self.config.get("dog_angle")
        ellipticity = self.config.get("dog_ellipticity")

        if max_scale is None:
            max_scale = min(self.img.data.shape) / 4

        if angle is 'beam' and self.img.has_beam():
            angle = self.img.get_beam().angle

        widths = np.arange(min_scale, max_scale, step)

        scales = wtutils.pyramiddec(self.img.data, widths=widths, angle=angle, ellipticity=ellipticity, boundary="symm")
        scales = [nputils.resize_like(s, self.img.data) for s in scales]

        scales_noises = wtutils.dec_noise_factor(wtutils.pyramiddec, self.bg, widths=widths, angle=angle,
                                                 ellipticity=ellipticity, beam=self.img.get_beam())

        return zip(scales, scales_noises, widths)


class FeaturesFinder(object):

    def __init__(self, img, background, config=None,
                 segment=True, filter=None):
        if config is None:
            config = FinderConfiguration()
        self.img = img
        self.background = background
        self.wt_dec = config.get("dec")
        self.config = config
        self.filter = filter
        self.segment = segment

    def execute(self):
        alpha_detection = self.config.get("alpha_detection")
        alpha_threashold = self.config.get("alpha_threashold")
        ms_dec_klass = self.config.get("ms_dec_klass")

        if self.config.get("use_iwd"):
            ms_dec_klass = InterscalesWaveletMultiscaleDecomposition

        dec = ms_dec_klass(self.img, self.background, self.config)
        decomposed = dec.decompose()

        result = MultiScaleImage(self.img, self.background, approx=dec.approx)
        for scale, scale_noise, width in decomposed:
            detection = alpha_detection * scale_noise
            threshold = alpha_threashold * scale_noise
            scale_img = imgutils.Image.from_image(self.img, scale.real)

            if self.segment:
                exculde_dist = self.config.get("exclude_border_dist")
                features = FeaturesGroup.from_img_peaks(scale_img, min(width, 8),
                                                        detection, exclude_border_dist=exculde_dist)
                mask = (scale.real > threshold)

                res = SegmentedScale(scale_img, [], None, scale_noise, self.img, width)
                res.watershed_segmentation(features, mask, feature_filter=self.filter)
                # res.connected_structure(mask=mask, feature_filter=self.filter)

                # res.img.data = self.img.data.copy()

                if self.config.get("exclude_noise"):
                    res.img.data[res.labels == 0] = 0
                # res.img.data[res.labels <= scale_noise] = 0

                logger.info("Detected features at scale %s: %s" % (width, res.size()))
                logger.info("Threasholds: %s, %s" % (detection, threshold))
            else:
                res = FeaturesGroup.from_img_peaks(scale_img, width, detection, feature_filter=self.filter,
                                                   fit_gaussian=False)

            scale_snr_filter = self.config.get("scales_snr_filter")
            if scale_snr_filter is not None and width in scale_snr_filter:
                res.filter_out_snr(scale_snr_filter[width])

            result.append(res)

        return result

    def direct_detection(self, width=2, alpha_detection=None, alpha_threashold=None):
        if alpha_detection is None:
            alpha_detection = self.config.get("alpha_detection")
        if alpha_threashold is None:
            alpha_threashold = self.config.get("alpha_threashold")

        if isinstance(self.background, np.ndarray):
            rms_noise = self.background.std()
        else:
            rms_noise = self.background

        detection = alpha_detection * rms_noise
        threshold = alpha_threashold * rms_noise

        features = FeaturesGroup.from_img_peaks(self.img, width, detection, feature_filter=self.filter)
        if self.segment:
            mask = (self.img.data > threshold)
            res = SegmentedScale(self.img, [], None, rms_noise, self.img, width)
            res.watershed_segmentation(features, mask)
        else:
            res = features
        # print "Detected features at scale %s: %s" % (width, res.size())
        return res


def test_filter():
    s1 = Feature([2, 2])
    s2 = Feature([2, 7])
    s3 = Feature([7, 7])

    m1 = imgutils.Mask(imgutils.draw_rectangle(np.zeros([10, 10]), [0, 0], [5, 9]))
    m2 = imgutils.Mask(imgutils.draw_rectangle(np.zeros([10, 10]), [0, 5], [9, 9]))

    f1 = MaskFilter(m1)
    f2 = MaskFilter(m2)

    assert f1.filter(s1) == True
    assert f1.filter(s2) == True
    assert f1.filter(s3) == False

    assert f2.filter(s1) == False
    assert f2.filter(s2) == True
    assert f2.filter(s3) == True

    assert (f1 & f2).filter(s1) == False
    assert (f1 & f2).filter(s2) == True
    assert (f1 & f2).filter(s3) == False

    assert (f1 | f2).filter(s1) == True
    assert (f1 | f2).filter(s2) == True
    assert (f1 | f2).filter(s3) == True

    assert ((f1 | f2) & f2).filter(s1) == False
    assert ((f1 | f2) & f2).filter(s2) == True
    assert ((f1 | f2) & f2).filter(s3) == True


if __name__ == '__main__':
    test_filter()
