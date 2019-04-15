import os
import re
import glob
import itertools
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage

from wds import *
from features import *

from libwise import nputils, imgutils, plotutils
from libwise.nputils import validator_is, is_callable, validator_in_range, validator_in
from libwise.nputils import validator_list, validator_is_class, str2bool, str2jsonclass

import jsonpickle as jp

from astropy import units as u

p2i = imgutils.p2i
logger = logging.getLogger(__name__)


class FeaturesLink(object):

    def __init__(self, first, color, id, delta=None):
        # MEM ISSUE: using dict here does not scale well with increasing
        # number of features.
        self.features = dict()
        self.relations = dict()
        self.features[first.get_epoch()] = first
        self.delta_info = DeltaInformation([first])
        self.color = color
        self.id = str(id)

    def __str__(self):
        s = "Link(id:%s" % self.id
        for epoch, related in self.relations.items():
            s += "%s:%s" % (epoch, related.get_id())
        for epoch in self.get_epochs():
            feature = self.get(epoch)
            s += "\n  " + str(epoch) + ", " + str(feature) + ", " + str(self.delta_info.get_delta(feature))
        s += "\n)"
        return s

    def first(self):
        return self.get_features()[0]

    def get_first_epoch(self):
        return self.get_epochs()[0]

    def get_last_epoch(self):
        return self.get_epochs()[-1]

    def last(self):
        return self.get_features()[-1]

    def get(self, epoch):
        return self.features.get(epoch, None)

    def get_id(self):
        return self.id

    def get_related(self, epoch):
        return self.relations.get(epoch, None)

    def is_related(self, link):
        return link in self.relations.values()

    def get_relations(self):
        return self.relations.items()

    def add_relation(self, epoch, link):
        self.relations[epoch] = link

    def add(self, feature, delta):
        # print "Add:", delta, feature.get_epoch() - self.last().get_epoch()
        self.delta_info.add_delta(self.last(), delta, feature.get_epoch() - self.last().get_epoch())
        self.features[feature.get_epoch()] = feature

    def get_features(self, start_epoch=None, end_epoch=None):
        features = nputils.get_values_sorted_by_keys(self.features)
        epochs = self.get_epochs()
        i_start = i_end = None
        if start_epoch is not None and start_epoch in epochs:
            i_start = epochs.index(start_epoch)
        if end_epoch is not None and end_epoch in epochs:
            i_end = epochs.index(end_epoch) + 1
        return features[slice(i_start, i_end)]

    def get_epochs(self):
        return sorted(self.features.keys())

    def get_all(self):
        return nputils.get_items_sorted_by_keys(self.features)

    def get_delta_info(self, measured_delta=True, coord_mode='com'):
        if measured_delta:
            return self.delta_info
        else:
            delta_info = DeltaInformation([])
            for feature1, feature2 in nputils.pairwise(self.get_features()):
                time_delta = feature2.get_epoch() - feature1.get_epoch()
                delta = feature2.get_coord(mode=coord_mode) - feature1.get_coord(mode=coord_mode)
                delta_info.add_delta(feature1, delta, time_delta)
            return delta_info

    def get_delta(self, feature):
        return self.delta_info.get_delta(feature)

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def size(self):
        return len(self.features)

    def get_filtered(self, feature_filter):
        new = self.copy()
        new.feature_filter(feature_filter)
        return new

    def feature_filter(self, feature_filter):
        for epoch, feature in list(self.get_all()):
            if not feature_filter.filter(feature):
                self.features.pop(epoch)
                if epoch in self.relations:
                    self.relations.pop(epoch)
                self.delta_info.remove_delta(feature)

    def fit_xy(self, fct, prj):
        xs, ys = zip(*[prj.p2s(p2i(f.get_coord())) for f in self.get_features()])
        fitted_fct = fct.fit(np.array(xs), np.array(ys))

        return fitted_fct, xs, ys

    def fit_dfc(self, fct, prj, time_unit=u.year, coord_mode='com'):
        dfcs = [prj.dfc(p2i(f.get_coord(mode=coord_mode))) for f in self.get_features()]
        epochs = [(nputils.datetime_to_mjd(epoch) * u.day).to(time_unit).value for epoch in self.get_epochs()]
        fitted_fct = fct.fit(epochs, dfcs)

        return fitted_fct, epochs, dfcs

    def copy(self):
        new = FeaturesLink(self.first(), self.color, self.id)
        new.features = self.features.copy()
        new.relations = self.relations.copy()
        new.delta_info = self.delta_info.copy()
        return new


def get_all_epochs(links):
    epochs = set()
    for link in links:
        epochs.update(link.get_epochs())
    return sorted(epochs)


def reset_links_colors(links):
    self.color_selector = plotutils.ColorSelector()
    links = sorted(links, key=lambda link: -link.size())
    for link in links:
        link.set_color(self.color_selector.get())


class FeaturesLinkBuilder(object):

    TYPE = '.dfc.dat'
    FORMATS = {6: 0, 8: 1, 9: 2}

    def __init__(self, scale=0):
        self.links = dict()
        self.scale = scale
        self.id_num = 0
        self.color_selector = plotutils.ColorSelector()

    def __str__(self):
        return "\n".join([str(k) for k in self.links.values()])

    def reset_colors(self, shuffle=False, link_sort_key=None):
        self.color_selector = plotutils.ColorSelector()
        if shuffle:
            self.color_selector.colors = list(np.random.permutation(list(self.color_selector.colors)))
        links = self.links.values()
        if link_sort_key is not None:
            links.sort(key=link_sort_key)
        for link in links:
            link.set_color(self.color_selector.get())

    def merge(self, other):
        for link in other.get_links():
            link.id = self.get_new_id()
            link.color = self.color_selector.get()
            self.add(link)

    def set_min_link_size(self, min_link_size):
        for link_id, link in self.links.items():
            if link.size() < min_link_size:
                del self.links[link_id]

    def size(self):
        return len(self.links)

    def get_new_id(self):
        self.id_num += 1
        return "%s:%s" % (self.scale, self.id_num)

    def set_scale(self, scale):
        self.scale = scale

    def add(self, link):
        self.links[link.get_id()] = link

    def add_new(self, first):
        id = self.get_new_id()
        color = self.color_selector.get()
        link = FeaturesLink(first, color, id)
        self.add(link)
        return link

    def get_scale(self, projection=None):
        if projection is not None:
            return projection.mean_pixel_scale() * self.scale
        return self.scale

    def get_all_epochs(self):
        return get_all_epochs(self.links.values())

    def get_all_features(self):
        for link in self.links.values():
            for feature in link.get_features():
                yield feature

    def get_features_id_mapping(self):
        result = dict()
        for link in self.get_links():
            result.update(dict.fromkeys(link.get_features(), link.get_id()))

        return result

    def get_features_epoch(self, epoch):
        group = DatedFeaturesGroup(epoch=epoch)
        for link in self.links.values():
            if link.get(epoch) is not None:
                group.add_feature(link.get(epoch))
        return group

    def get_delta_info(self, measured_delta=True, coord_mode='com'):
        final_delta_info = DeltaInformation()
        for link in self.get_links():
            delta_info = link.get_delta_info(measured_delta=measured_delta, coord_mode=coord_mode)
            final_delta_info.merge(delta_info)

        return final_delta_info

    def get_match_results(self, min_link_size=2):
        epochs = self.get_all_epochs()
        matchs = defaultdict(FeaturesMatch)
        delta_infos = defaultdict(DeltaInformation)
        features1 = dict([(k, DatedFeaturesGroupScale(self.scale, epoch=k)) for k in epochs])
        features2 = dict([(k, DatedFeaturesGroupScale(self.scale, epoch=k)) for k in epochs])
        for link in self.get_links():
            if link.size() < min_link_size:
                continue

            for feature1, feature2 in nputils.pairwise(link.get_features()):
                epoch = feature1.get_epoch()
                features1[epoch].add_feature(feature1)
                features2[epoch].add_feature(feature2)
                matchs[epoch].add_feature_match(feature1, feature2)
                delta = link.get_delta_info().get_full_delta(feature1)
                delta_infos[epoch].add_full_delta(delta)

        match_results = dict()
        for epoch in epochs:
            match_result = ScaleMatchResult(features1[epoch], features2[epoch],
                                            matchs[epoch], delta_infos[epoch], None)
            match_results[epoch] = match_result

        return match_results

    def add_match(self, match, delta_info, epoch):
        last_links = [(k, k.get(epoch)) for k in self.links.values() if k.get(epoch) is not None]

        coord_feature_map = dict()
        for link, last in last_links:
            if last.has_inner_features():
                for inner_feature in last.get_inner_features():
                    coord_feature_map[tuple(inner_feature.get_coord('lm'))] = link
            else:
                coord_feature_map[tuple(last.get_coord('lm'))] = link

        for (feature1, feature2) in match.get_pairs():
            # print "Add:", feature1.get_segmentid(), feature2.get_segmentid()
            linked_links = []

            coord = tuple(feature1.get_coord('lm'))
            # print [coord_feature_map.get(coord)]
            if feature1.has_inner_features():
                # print [tuple(f.get_coord('lm')) for f in feature1.get_inner_features()]
                # print [coord_feature_map.get(tuple(f.get_coord('lm'))) for f in feature1.get_inner_features()]
                linked_links = [coord_feature_map.get(tuple(f.get_coord('lm'))) for f in feature1.get_inner_features()]
                # print feature1, [l.get_id() for l in linked_links]
            elif coord in coord_feature_map:
                linked_links = [coord_feature_map.get(coord)]
            linked_links = list(set([k for k in linked_links if k is not None]))

            # print "Possible links:", [k.get_id() for k in linked_links]
            if len(linked_links) == 0:
                new_link = self.add_new(feature1)
                # print "1:", new_link.last().get_epoch(), feature2.get_epoch()
                new_link.add(feature2, delta_info.get_delta(feature1))
            elif len(linked_links) == 1:
                link = linked_links[0]
                # check case: 1 (merged) -> 2
                last = link.get(feature1.get_epoch())
                # check if last is a merged segment and if any other inner segment is also matched
                inner_matched = [(k in match.get_ones()) for k in last.get_inner_features()]
                if sum(inner_matched) >= 2 and all(last.get_coord() != feature1.get_coord()):
                    # create new link because last is a merged segment with
                    # 2 inner segments that are matched and feature does not correspond to the
                    # brightest one
                    new_link = self.add_new(feature1)
                    # print "2:", new_link.last().get_epoch(), feature2.get_epoch()
                    new_link.add(feature2, delta_info.get_delta(feature1))
                    new_link.add_relation(feature1.get_epoch(), link)
                else:
                    # print "3:", link.last().get_epoch(), feature2.get_epoch()
                    link.add(feature2, delta_info.get_delta(feature1))
            elif len(linked_links) >= 2:
                # case: 2 -> 1 (merged)
                # Feature is merged in current match, check where is the com or coord of merged feature and
                # continue from there
                link_winner = link_loser = None
                for link in linked_links:
                    last = link.get(feature1.get_epoch())
                    last_segments = last.get_segmented_image()
                    matched = last_segments.get_segment_from_feature(feature1, mode='lm')
                    # print "Test:", link.get_id(), last.get_segmentid(), matched.get_segmentid()
                    if matched is None:
                        matched = last_segments.get_segment_from_feature(feature1, mode='lm')
                        # print "--:", matched.get_segmentid()
                    if matched == last:
                        # print "4:", link.last().get_epoch(), feature2.get_epoch()
                        link.add(feature2, delta_info.get_delta(feature1))
                        link_winner = link
                    else:
                        link_loser = link
                if link_loser is not None and link_winner is not None:
                    link_loser.add_relation(link_loser.get_last_epoch(), link_winner)

    def get_links_iter(self, feature_filter=None, min_link_size=0):
        for link in self.links.values():
            if feature_filter is not None:
                link = link.get_filtered(feature_filter)
            if link.size() > min_link_size:
                yield link

    def get_links(self, feature_filter=None, min_link_size=0):
        return list(self.get_links_iter(feature_filter=feature_filter, min_link_size=min_link_size))

    def get_filtered(self, link_ids):
        link_ids = map(str, link_ids)
        new = self.copy()
        for id, link in self.links.items():
            if not id in link_ids:
                del new.links[id]
        return new

    def get_link(self, id):
        return self.links[id]

    def has_link(self, id):
        return id in self.links

    def to_file(self, file, projection, coord_mode='com', suffix=None, measured_delta=True):
        '''Format is: epoch, x, y, intensity, snr, component id number, related component id, delta_x, delta_y
           delta_x, delta_y is delta between previous and current feature !!'''
        if suffix is None:
            suffix = FeaturesLinkBuilder.TYPE
        l = []
        for id, link in nputils.get_items_sorted_by_keys(self.links):
            delta_info = link.get_delta_info(measured_delta=measured_delta, coord_mode=coord_mode)
            delta_x = delta_y = 0
            for epoch, feature in link.get_all():
                x, y = projection.p2s(p2i(feature.get_coord(mode=coord_mode)))
                snr = feature.get_snr()
                epoch = float(nputils.datetime_to_epoch(epoch))
                intensity = feature.get_intensity()
                related = link.get_related(epoch)
                if related is None:
                    related_id = 'None'
                else:
                    related_id = related.get_id()
                l.append([epoch, x, y, intensity, snr, id, related_id, delta_x, delta_y])
                # TODO: need rotation independant delta!!
                delta_x, delta_y = projection.pixel_scales() * p2i(delta_info.get_delta(feature))

        unit = projection.unit
        header = 'WISE matched components list at scale %f %s\n' % (self.get_scale(projection), unit)
        header += 'Delta is between previous and current feature.\n'
        header += ('Epoch, X (%s), Y (%s), Intensity, SNR, Component ID, Related Component ID, '
                   'Delta_X (%s), Delta_Y (%s)\n' % (unit, unit, unit, unit))

        l = np.array(l, dtype=object)
        filename = file + suffix
        np.savetxt(filename, l, ["%f", "%.5f", "%.5f", "%.6f", "%.3f", "%s", "%s", "%.5f", "%.5f"],
                   delimiter=' ', header=header)
        print "Saved link builder @ %s" % filename

    @staticmethod
    def from_separation_file(file, projection, image_set, filter=None):
        '''Format is: epoch, distance, x, y, epoch number, component id number.'''
        new = FeaturesLinkBuilder()
        array = np.loadtxt(file, dtype=str)
        # sort by component id
        array = sorted(array, key=lambda line: line[5])
        current_component_id = 0
        coord_sys = projection.get_coordinate_system()
        img_metas = dict()
        if filter is not None:
            filter = map(str, filter)
        for line in array:
            date = nputils.epoch_to_datetime(line[0])
            x, y = np.array(map(float, line[2:4]))
            # TODO: check why we need -x
            x, y = projection.s2p([-x, y])
            component_id = str(line[5])
            if filter is not None and not component_id in filter:
                continue
            if date not in img_metas:
                img_metas[date] = imgutils.ImageMeta(date, coord_sys, image_set.get_beam(date))
            f = ImageFeature([y, x], img_metas[date], 1, 1)
            if current_component_id != component_id:
                component = FeaturesLink(f, new.color_selector.get(), component_id)
                new.links[component_id] = component
            else:
                delta = f.get_coord() - component.last().get_coord()
                component.add(f, delta)
            current_component_id = component_id
        new.reset_colors(link_sort_key=lambda link: -link.size())
        print "Loaded link builder from sep file %s" % file
        return new

    @staticmethod
    def get_file_version(file):
        array = np.loadtxt(file, dtype=str, delimiter=' ')
        if len(array) < 1:
            return None
        return FeaturesLinkBuilder.FORMATS[len(array[0])]

    @staticmethod
    def from_file(file, projection, image_set, suffix=None, filter=None, min_link_size=2):
        '''Format 0: epoch, x, y, intensity, component id number, related component id
           Format 1: epoch, x, y, intensity, component id number, related component id, delta_x, delta_y
           Format 2: epoch, x, y, intensity, snr, component id number, related component id, delta_x, delta_y

           delta_x, delta_y is delta between previous and current feature !!'''
        if suffix is None:
            suffix = FeaturesLinkBuilder.TYPE

        file = file + suffix
        new = FeaturesLinkBuilder()
        array = np.loadtxt(file, dtype=str, delimiter=' ')
        relations = []
        current_component_id = 0
        coord_sys = projection.get_coordinate_system()
        img_metas = dict()
        if filter is not None:
            filter = map(str, filter)
        for line in array:
            format = FeaturesLinkBuilder.FORMATS[len(line)]
            date = nputils.epoch_to_datetime(line[0])
            x, y = projection.s2p(map(float, line[1:3]))
            intensity = float(line[3])
            if format in [0, 1]:
                snr = 5
                inext = 4
            else:
                snr = float(line[4])
                inext = 5
            component_id = str(line[inext])
            related_id = str(line[inext + 1])
            if filter is not None and not component_id in filter:
                continue
            if date not in img_metas:
                img_metas[date] = imgutils.ImageMeta(date, coord_sys, image_set.get_beam(date))
            f = ImageFeature([y, x], img_metas[date], intensity, snr)
            if current_component_id != component_id:
                component = FeaturesLink(f, new.color_selector.get(), component_id)
                new.links[component_id] = component
            else:
                if format == 0:
                    delta = f.get_coord() - component.last().get_coord()
                else:
                    delta = p2i(np.array(line[inext + 2:inext + 4], dtype=float) / projection.pixel_scales())
                component.add(f, delta)
            if related_id != 'None':
                relations.append([date, component, related_id])
            current_component_id = component_id
        for epoch, component, related_id in relations:
            related_component = new.get_link(related_id)
            component.add_relation(epoch, related_component)
        new.set_min_link_size(min_link_size)
        new.reset_colors(link_sort_key=lambda link: -link.size())
        print "Loaded link builder from %s" % file
        return new

    def copy(self):
        new = FeaturesLinkBuilder(scale=self.scale)
        new.links = self.links.copy()

        return new


class MultiScaleFeaturesLinkBuilder(object):

    TYPE = '.ms' + FeaturesLinkBuilder.TYPE

    def __init__(self):
        self.link_builders = dict()

    def __str__(self):
        return "\n".join([str(k) for k in self.link_builders.values()])

    def merge(self, other):
        this_scales = self.link_builders.keys()
        other_scales = other.link_builders.keys()
        scales = set(this_scales + other_scales)

        for scale in scales:
            if scale not in this_scales:
                self.link_builders[scale] = other.get_scale(scale)
            elif scale in other_scales:
                self.get_scale(scale).merge(other.get_scale(scale))

    def __init_builders(self, scales):
        self.link_builders = dict([(k, FeaturesLinkBuilder(k)) for k in scales])

    def get_all(self):
        return nputils.get_values_sorted_by_keys(self.link_builders)

    def get_all_epochs(self):
        epochs = set()
        for link_builder in self.link_builders.values():
            epochs.update(link_builder.get_all_epochs())
        return sorted(epochs)

    def get_link(self, id):
        for link_builder in self.link_builders.values():
            if link_builder.has_link(id):
                return link_builder.get_link(id)

        return None

    def get_scale(self, scale):
        return self.link_builders.get(scale, None)

    def get_scales(self):
        return sorted(self.link_builders.keys())

    def add_match_result(self, ms_match_result):
        if len(self.link_builders) == 0:
            self.__init_builders(ms_match_result.get_scales())
        for scale_match_result in ms_match_result:
            link_builder = self.link_builders[scale_match_result.get_scale()]
            (segments1, segments2, match, delta_info) = scale_match_result.get_all()
            link_builder.add_match(match, delta_info, scale_match_result.get_epoch())
        for link_builder in self.link_builders.values():
            link_builder.reset_colors(link_sort_key=lambda link: -link.size())

    def get_ms_match_results(self, min_link_size=2):
        all_ms = defaultdict(MultiScaleMatchResult)
        for link_builder in self.get_all():
            for epoch, match_result in link_builder.get_match_results(min_link_size=min_link_size).items():
                all_ms[epoch].append(match_result)

        return MultiScaleMatchResultSet(nputils.get_values_sorted_by_keys(all_ms))

    def to_file(self, filename, projection, coord_mode='com', measured_delta=True):
        for link_builder in self.get_all():
            if link_builder.size() > 0:
                scale = link_builder.get_scale()
                scale_filename = '%s_%s' % (filename, str(scale))
                link_builder.to_file(scale_filename, projection, suffix=MultiScaleFeaturesLinkBuilder.TYPE,
                                     coord_mode=coord_mode, measured_delta=measured_delta)

    @staticmethod
    def from_file(filename, projection, image_set, min_link_size=2):
        new = MultiScaleFeaturesLinkBuilder()
        regex = '%s_[0-9]+%s' % (os.path.basename(filename), MultiScaleFeaturesLinkBuilder.TYPE)
        for file in glob.glob(filename + '_*' + MultiScaleFeaturesLinkBuilder.TYPE):
            if re.match(regex, os.path.basename(file)):
                scale = float(file.split('_')[-1].split(MultiScaleFeaturesLinkBuilder.TYPE)[0])
                link_builder = FeaturesLinkBuilder.from_file(file, projection, image_set,
                                                             suffix='', min_link_size=min_link_size)
                link_builder.set_scale(scale)
                new.link_builders[scale] = link_builder
        return new


class MergedFeatureLink(FeaturesLink):

    def __init__(self, links, color, id):
        print id, links
        first = links[0].link.get(links[0].start)
        FeaturesLink.__init__(self, first, color, id)

        for i in range(len(links)):
            link, start_epoch, end_epoch = links[i].get()
            # print link, start_epoch, end_epoch
            if start_epoch is None and i > 0:
                start_epoch = links[i - 1].end
            if end_epoch is None and i < len(links):
                end_epoch = links[i + 1].end
            for feature in link.get_features(start_epoch=start_epoch, end_epoch=end_epoch):
                self.add(feature, link.get_delta(feature))


class MergeLinkId(object):

    def __init__(self, link, start=None, end=None):
        self.link = link
        self.start = start
        self.end = end

    def get(self):
        return self.link, self.start, self.end

    @staticmethod
    def parse(ms_link_builder, str):
        start = end = None
        if '[' in str:
            link_id, dates = str.split('[')
            start_str, end_str = dates[:-1].split(":")
            if len(start_str) > 0:
                start = nputils.epoch_to_datetime(start_str)
            if len(end_str) > 0:
                end = nputils.epoch_to_datetime(end_str)
        else:
            link_id = str
        link = ms_link_builder.get_link(link_id)
        if start is None:
            start = link.get_first_epoch()
        if end is None:
            end = link.get_last_epoch()

        return MergeLinkId(link, start, end)


class MergedFeatureLinkBuilder(FeaturesLinkBuilder):

    def __init__(self, ms_link_bulder, merge_file):
        FeaturesLinkBuilder.__init__(self)
        self.ms_link_builder = ms_link_bulder
        self.merge_file = merge_file
        self.build()

    def build(self):
        all_links_id = []
        with open(self.merge_file) as file:
            for i, line in enumerate(file.readlines()):
                line = line.strip()
                if not line.startswith("#") and len(line) > 0:
                    all_links_id.append(line.split(','))

        for i, links_id in enumerate(all_links_id):
            start = end = None
            links = [MergeLinkId.parse(self.ms_link_builder, id) for id in links_id]
            link = MergedFeatureLink(links, self.color_selector.get(), i)
            self.add(link)


class BaseScaleMatcher(object):

    def __init__(self, segments1, segments2, upper_delta_info=None, match_config=None):
        self.segments1 = segments1
        self.segments2 = segments2
        if upper_delta_info is None:
            upper_delta_info = DeltaInformation(segments1)
        if match_config is None:
            match_config = MatcherConfiguration()
        self.config = match_config
        self.upper_delta_info = upper_delta_info
        if isinstance(self.segments1, SegmentedScale):
            self.scale = self.segments1.get_scale()
        else:
            self.scale = 0
        self.delta_time = self.segments2.get_epoch() - self.segments1.get_epoch()

        self.tol_k = self.config.get("tolerance_factor")

        self.delta_check_cache = nputils.LimitedSizeDict(size_limit=500)

    def get_features_to_match(self):
        f1, f2 = self.get_features_at_the_border()
        f1_no_input = self.get_no_input_no_match()

        to_match1 = FeaturesGroup(self.segments1.get_features() - f1.get_features() - f1_no_input.get_features())
        to_match2 = FeaturesGroup(self.segments2.get_features() - f2.get_features())

        return to_match1, to_match2

    def get_delta_time(self):
        return self.delta_time

    def get_tolerance(self, segments=[]):
        tol = np.round(self.tol_k * self.scale)
        for segment in segments:
            if self.config.get("increase_tol_for_no_input_delta"):
                if self.upper_delta_info.get_flag(segment) != DeltaInformation.DELTA_MATCH:
                    tol = np.round(1.2 * self.tol_k * self.scale)
                    break
        return max(tol, self.config.get("min_scale_tolerance").get(self.scale, 1))

    def get_delta_filter(self):
        return self.config.get("delta_range_filter")

    def get_no_input_no_match(self):
        if self.scale in self.config.get("no_input_no_match_scales"):
            return self.upper_delta_info.get_features(flag=DeltaInformation.NO_DELTA)
        return FeaturesGroup()

    def get_features_at_the_border(self):
        shape = self.segments1.get_img().data.shape
        features_at_the_border1 = FeaturesGroup()
        features_at_the_border2 = FeaturesGroup()

        if self.config.get("ignore_features_at_border") is True:
            for feature in self.segments1:
                distances1 = nputils.distance_from_border(feature.get_coord(), shape)
                if min(distances1) < self.config.get("features_at_border_k1") * self.scale:
                    features_at_the_border1.add_feature(feature)
                input_delta = self.upper_delta_info.get_delta(feature)
                distances2 = nputils.distance_from_border(feature.get_coord() + input_delta, shape)
                if min(distances2) < self.config.get("features_at_border_k2") * self.scale:
                    features_at_the_border1.add_feature(feature)
                # print feature, min(distances1), min(distances2), min(distances1) < self.config.get("features_at_border_k1") * self.scale, min(distances2) < self.config.get("features_at_border_k2") * self.scale
            for feature in self.segments2:
                distances1 = nputils.distance_from_border(feature.get_coord(), shape)
                # print feature, distances1, min(distances1) < self.config.get("features_at_border_k1") * self.scale
                if min(distances1) < self.config.get("features_at_border_k3") * self.scale:
                    features_at_the_border2.add_feature(feature)

        return features_at_the_border1, features_at_the_border2

    def check_delta(self, feature, delta):
        if self.get_delta_filter() is not None:
            key = (feature, tuple(delta))
            if not key in self.delta_check_cache:
                delta_obj = Delta(feature, delta, self.get_delta_time())
                self.delta_check_cache[key] = self.get_delta_filter().filter(delta_obj)
            return self.delta_check_cache[key]

        if np.linalg.norm(delta) > self.config.get("maximum_delta"):
            return False

        if not nputils.in_range(delta[0], self.config.get("range_delta_x")):
            return False

        if not nputils.in_range(delta[1], self.config.get("range_delta_y")):
            return False

        return True


class ScaleMatcherMSCI(BaseScaleMatcher):
    ''' Multi scale cross identification with simple merge '''

    def do_get_match_scale(self, features1, features2, tol, upper_delta_info):
        to_match1, to_match2 = self.get_features_to_match()

        if upper_delta_info is not None:
            to_match1.move_features_from_delta_info(upper_delta_info)
            match = to_match1.get_match2(to_match2, tol=tol)
            to_match1.move_back_to_initial()
        else:
            match = to_match1.get_match2(to_match2, tol=tol)

        logging.info("Matching features: %s", match.size())

        if self.config.get("simple_merge"):
            com_delta_fct = lambda x, y: y.get_center_of_shape() - x.get_center_of_shape()

            # simple merge (do not consider features at the border)
            delta_info1 = DeltaInformation(features1, features1.get_scale() * 10)
            delta_info1.add_match(match, delta_fct=com_delta_fct)
            delta_info1.complete_with_average_delta()

            for f1 in to_match1.get_features() - set(match.get_ones()):
                delta = delta_info1.get_delta(f1)
                f2 = features2.get_overlapping_segment(f1, delta)
                if f2 is not None:
                    f1peer = match.get_peer_of_two(f2)
                    if f1peer is not None and f1peer != f1:
                        # print "Merge1", f1peer.get_segmentid(), f1.get_segmentid(), f2.get_segmentid()
                        features1.merge_segments(f1peer, f1)

            delta_info2 = DeltaInformation(features2, features2.get_scale() * 10)
            delta_info2.add_match(match.reverse(), delta_fct=com_delta_fct)
            delta_info2.complete_with_average_delta()

            for f2 in to_match2.get_features() - set(match.get_twos()):
                delta = delta_info2.get_delta(f2)
                f1 = features1.get_overlapping_segment(f2, delta)
                if f1 is not None:
                    f2peer = match.get_peer_of_one(f1)
                    if f2peer is not None and f2peer != f2:
                        # print "Merge2", f2peer.get_segmentid(), f2.get_segmentid(), f1.get_segmentid()
                        features2.merge_segments(f2peer, f2)

        coord_mode = self.config.get('msci_coord_mode')
        delta_fct = lambda x, y: y.get_coord(coord_mode) - x.get_coord(coord_mode)

        delta_info = DeltaInformation(features1, features1.get_scale() * 10)
        delta_info.add_match(match, delta_fct=delta_fct)

        return match, delta_info

    def get_match(self, cb=None, verbose=True):
        tol = self.get_tolerance()

        logging.info("Start Matching at scale %s. Tolerence: %s" % (self.segments1.get_scale(), tol))

        match, delta_info = self.do_get_match_scale(self.segments1, self.segments2, tol, self.upper_delta_info)

        return ScaleMatchResult(self.segments1, self.segments2, match, delta_info, self.upper_delta_info)


class ScaleMatcherMSCSC(BaseScaleMatcher):
    ''' Multi scale cross segments correlation '''

    def __init__(self, segments1, segments2, upper_delta_info=None, match_config=None):
        BaseScaleMatcher.__init__(self, segments1, segments2, upper_delta_info, match_config)
        self._correlation_cache = dict()

        if self.config.get("find_distance_mode") == 'min':
            self.mode = ['lm', 'com', 'cos']
        else:
            self.mode = [self.config.get("find_distance_mode")]

        self.correlation_threshold = self.config.get("correlation_threshold")
        self.max_merge = self.config.get("mscsc_max_merge")

    def get_delta_proba(self, segment, delta):
        if self.upper_delta_info.get_flag(segment) != DeltaInformation.DELTA_MATCH:
            return 1
        return 1.5 - 1 * np.linalg.norm(delta) / self.get_tolerance([segment])

    def get_segment_image(self, features):
        mask = reduce(lambda x, y: x + y, [k.get_mask() for k in features])
        img = tuple(features)[0].get_segmented_image()
        seg_img = img.get_img().data * mask

        reducde_fct = lambda a, b: (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        x0, y0, x1, y1 = reduce(reducde_fct, [k.get_cropped_index() for k in features])

        croped_seg_img = seg_img[x0:x1, y0:y1]

        return seg_img, croped_seg_img, (x0, y0, x1, y1)

    def segments_correlation(self, segments1, segments2, window_search=None,
                             window_search_offset=None, cb=None, delta_tol=None):
        seg_img1, croped_seg_img1, index1 = self.get_segment_image(segments1)
        seg_img2, croped_seg_img2, index2 = self.get_segment_image(segments2)

        center1 = np.array([index1[0], index1[1]]) + np.array(croped_seg_img1.shape) / 2.
        center2 = np.array([index2[0], index2[1]]) + np.array(croped_seg_img2.shape) / 2.
        corr_center = center2 - center1

        corr = nputils.norm_xcorr2(croped_seg_img1, croped_seg_img2, mode='same')

        if cb is not None:
            cb(croped_seg_img1, croped_seg_img2, corr)

        if window_search_offset is None:
            window_search_offset = np.array([0, 0])

        if window_search is not None:
            window_search_coord = np.array(corr.shape) / 2 + corr_center - np.array(window_search_offset)
            corr = nputils.zoom(corr, window_search_coord, window_search)

        corr_center = np.array(window_search_offset)

        if corr.size > 1:
            coef = corr.max()
            peak = nputils.coord_max(corr)
            delta = (np.array(corr.shape) / 2 - peak)

            distance = np.round(corr_center + delta)

            if delta_tol is not None:
                if np.linalg.norm(np.array(corr.shape) / 2 - peak) > delta_tol:
                    # print 'Rejected:', delta_tol, window_search_offset
                    return 0, delta, distance

            coef = nputils.norm_xcorr_coef(seg_img1, seg_img2, distance)
            coef_shape = nputils.norm_xcorr_coef((seg_img1 > 0), (seg_img2 > 0), distance)
            coef = (coef + coef_shape) / 2.

            return coef, delta, distance

        return 0, [0, 0], [0, 0]

    def get_correlation(self, features1, features2, cb=None):
        if (features1, features2) in self._correlation_cache:
            return self._correlation_cache[(features1, features2)]

        if len(features1) == 0 or len(features2) == 0:
            return self.correlation_threshold, [0, 0]

        tol = self.get_tolerance(features1)
        window_search = [tol * 2 + 1] * 2
        window_search_offset = self.upper_delta_info.get_deltas(features1).mean(axis=0)
        # print "Upper deltas:", self.upper_delta_info.get_deltas(features1)

        result = self.segments_correlation(features1, features2,
                                           window_search=window_search,
                                           window_search_offset=window_search_offset,
                                           cb=cb, delta_tol=tol)

        self._correlation_cache[(features1, features2)] = result

        return result

    def get_combinations(self, item):

        def filter(combi):
            if len(combi) > self.max_merge:
                return False
            combi_group = FeaturesGroup(combi)
            for feature in combi:
                # if feature is connected to any feature from the group, always keep it
                if len(feature.get_connected_segments().intersect(combi_group)):
                    continue
                # One feature of the group shall not be of a distance > 2 * scale of an other feature
                if len(combi_group.find(feature, tol=2 * self.scale)) < 2:
                    return False
            return True

        for list1, list2 in nputils.lists_combinations(item.set1(), item.set2(), filter=filter):
            group = MatchingGroup()
            for f1s, f2s in zip(list1, list2):
                group.add(MatchingItem(f1s, f2s))
            yield group

    def optimize(self, indep_item, cb=None):
        results = []
        pmax = 6
        total_features = len(indep_item.set1()) + len(indep_item.set2())
        img2 = self.get_segment_image(indep_item.set2())[0]
        if total_features > pmax:
            print "Warning: high total features to optimize:", str([u.get_segmentid() for u in indep_item.set1()]) + " -> " + str([u.get_segmentid() for u in indep_item.set2()])
        # print "Optimize group:", str([u.get_segmentid() for u in indep_item.set1()]) + " -> " + str([u.get_segmentid() for u in indep_item.set2()])
        for group in self.get_combinations(indep_item):
            total_coef = 0
            total_matched_features = 0
            # print "Test combination:",
            for item in group.items():
                coef, scale_delta, delta = self.get_correlation(item.set1(), item.set2(), cb=cb)

                if not self.check_delta(list(item.set1())[0], delta):
                    coef = 0

                matched_features = len(item.set1()) + len(item.set2())
                total_matched_features += matched_features
                total_coef += matched_features * coef

                item.set_delta(scale_delta, delta)
                item.set_correlation(coef)
                # if total_features <= pmax:
                #     print "  ", [u.get_segmentid() for u in item.set1()], "->", [u.get_segmentid() for u in item.set2()], coef, delta, "",
            total_coef += self.correlation_threshold * (total_features - total_matched_features)
            total_coef = total_coef / float(total_features)
            diff = -1
            if total_coef >= self.correlation_threshold:
                img1 = np.zeros_like(img2)
                proba = []
                for item in group.items():
                    img1 += nputils.shift2d(self.get_segment_image(item.set1())[0], item.get_delta())
                    proba.extend([self.get_delta_proba(k, item.get_scale_delta()) for k in item.set1()])

                diff = ((img2 - img1) ** 2).sum()
                coef = nputils.norm_xcorr_coef(img1, img2)
                proba = np.array(proba).mean()
                # print "Proba:", proba, "Coef:", coef,
                coef = proba * coef

                # if cb is not None:
                #     cb("", img1, img2)

                results.append([group, total_coef, diff, coef])
            # if total_features <= pmax:
            #     print "=> Matched: ", total_matched_features, "/", total_features, "Coef:", coef, "Diff:", diff
        if len(results) == 0:
            return MatchingGroup()
        best = max(results, key=lambda res: res[1])
        # best = min(results, key=lambda res: res[2])
        # print "=> Result: ", ','.join([str(k) for k in best[0].items()]), "Correlation:", best[1]
        return best[0]

    def get_match(self, cb=None, verbose=True):
        print "\nStart Matching at scale %s. Tolerence: %s" % (self.scale, self.get_tolerance())

        if self.config.get("ignore_features_at_border"):
            features_at_the_border1 = self.get_features_at_the_border()[0]
            to_match1 = FeaturesGroup(self.segments1.get_features() - features_at_the_border1.get_features())
        else:
            to_match1 = self.segments1

        self.segments1.move_features_from_delta_info(self.upper_delta_info)

        to_match2_query = FeaturesQuery(self.segments2, coord_modes=self.mode)

        match = FeaturesMatch()
        delta_info = DeltaInformation(self.segments1, average_tol=self.scale * 10)

        matching_group = ScaleMatchingGroup()
        cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())

        for segment1 in to_match1.sorted_list(cmp=cmp_intensity):
            match_item = self.find_matches(segment1, to_match2_query)
            if match_item is not None:
                matching_group.add(match_item)

        total_correlation = 0

        for indep_item in matching_group.group_independant().items():
            for item in self.optimize(indep_item, cb=cb).items():
                features1 = list(item.set1())
                features2 = list(item.set2())

                if len(features1) == 0 or len(features2) == 0 or item.get_correlation() < self.correlation_threshold:
                    continue

                total_correlation += item.get_correlation()

                sort_key = lambda a: a.get_segmentid()

                features1.sort(key=sort_key)
                features2.sort(key=sort_key)

                master_feature1 = features1[0]
                for feature_to_merge in features1[1:]:
                    self.segments1.merge_segments(master_feature1, feature_to_merge)

                master_feature2 = features2[0]
                for feature_to_merge in features2[1:]:
                    self.segments2.merge_segments(master_feature2, feature_to_merge)

                match.add_feature_match(master_feature1, master_feature2)
                delta_info.add_delta(master_feature1, item.get_delta(), self.delta_time)

        self.segments1.move_back_to_initial()

        if match.size() > 0:
            print "Matching features:", match.size(), "Correlation:", total_correlation / match.size()

        return ScaleMatchResult(self.segments1, self.segments2, match, delta_info, self.upper_delta_info)

    def find_matches(self, segment1, segments2):
        tol = self.get_tolerance([segment1])

        seg2 = set(segments2.find(segment1, tol=tol))
        seg2 |= set(
            self.segments2.get_segments_inside2(segment1, delta=self.upper_delta_info.get_delta(segment1), ratio=0.4))
        if len(seg2) > 0:
            item = MatchingItem((segment1,), seg2)
            # print "Match:", item
            return item
        return None


class ScaleMatcherMSCSC2(BaseScaleMatcher):
    ''' Multi scale cross segments correlation '''

    def __init__(self, segments1, segments2, upper_delta_info=None, match_config=None):
        BaseScaleMatcher.__init__(self, segments1, segments2, upper_delta_info, match_config)
        self._matching_items_cache = nputils.LimitedSizeDict(size_limit=500)
        self._correlation_cache = nputils.LimitedSizeDict(size_limit=500)

        if self.config.get("find_distance_mode") == 'min':
            self.mode = ['lm', 'com', 'cos']
        else:
            self.mode = [self.config.get("find_distance_mode")]

        self.correlation_threshold = self.config.get("correlation_threshold")
        self.max_merge = self.config.get("mscsc_max_merge")

        self.debug = False
        # self.delta_year = (segments2.get_epoch() - segments1.get_epoch()).days / 365.

    def get_segment_image(self, features):
        mask = reduce(lambda x, y: x + y, [k.get_mask() for k in features])
        img = tuple(features)[0].get_segmented_image()
        seg_img = img.get_img().data * mask

        reducde_fct = lambda a, b: (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        x0, y0, x1, y1 = reduce(reducde_fct, [k.get_cropped_index() for k in features])

        croped_seg_img = seg_img[x0:x1, y0:y1]

        return seg_img, croped_seg_img, (x0, y0, x1, y1)

    def segments_correlation(self, segments1, segments2, window_search=None,
                             window_search_offset=None, cb=None, delta_tol=None):
        r1 = build_image(segments1)
        r2 = build_image(segments2)

        if window_search_offset is not None:
            if not r1.check_shift(np.round(window_search_offset)):
                return 0, np.array([0, 0]), window_search_offset

            r1.set_shift(np.round(window_search_offset))

        delta1 = r2.get_center_of_mass() - r1.get_center_of_mass()
        # delta2 = r2.get_coord_max() - r1.get_coord_max()

        def correlate(delta):
            if not r1.check_shift(np.round(window_search_offset + delta)):
                return delta, 0
            # print "Window search:", window_search_offset
            # print "Com:", r2.get_center_of_mass(), r1.get_center_of_mass()
            if delta_tol is not None and np.linalg.norm(delta) > delta_tol:
                # print "Rejected:", delta_tol, np.linalg.norm(delta)
                # return 0, delta, window_search_offset + delta
                return delta, 0

            r1.set_shift(np.round(window_search_offset + delta))

            if self.config.get("mscsc2_smooth"):
                # To improve performance, we need to run smooth only on the segments
                if min(r1.get_shape_region()) > 3:
                    r1.img = nputils.smooth(r1.img, 3, boundary="zero", mode="same")
                if min(r2.get_shape_region()) > 3:
                    r2.img = nputils.smooth(r2.img, 3, boundary="zero", mode="same")

            i1 = r1.get_data()
            i2 = r2.get_data()

            coef_data = nputils.norm_xcorr_coef(i1, i2)
            coef_shape = nputils.norm_xcorr_coef((i1 > 0), (i2 > 0))
            coef = (coef_data + coef_shape) / 2.

            r1.set_shift(None)

            return delta, coef

        results = [correlate(delta1)]
        delta, coef = max(results, key=lambda res: res[1])

        if cb is not None:
            title = [u.get_segmentid() for u in segments1], "->", [u.get_segmentid() for u in segments2]
            cb(title, region1.get_data(), region2.get_data())

        return coef, delta, window_search_offset + delta

    def get_correlation(self, features1, features2, cb=None):
        if (features1, features2) in self._correlation_cache:
            return self._correlation_cache[(features1, features2)]

        if len(features1) == 0 or len(features2) == 0:
            return self.correlation_threshold, [0, 0]

        tol = self.get_tolerance(features1)
        window_search = [tol * 2 + 1] * 2
        window_search_offset = self.upper_delta_info.get_deltas(features1).mean(axis=0)
        # print "Upper deltas:", self.upper_delta_info.get_deltas(features1)

        result = self.segments_correlation(features1, features2,
                                           window_search=window_search,
                                           window_search_offset=window_search_offset,
                                           cb=cb, delta_tol=tol)

        self._correlation_cache[(features1, features2)] = result

        return result

    def get_matchin_item(self, features1, features2):
        key = (features1, features2)
        if not key in self._matching_items_cache:
            matching_item = MatchingItem(*key)
            self._matching_items_cache[key] = matching_item

        return self._matching_items_cache[key]

    def get_combinations(self, item):

        def filter(combi):
            if len(combi) > self.max_merge:
                return False
            combi_group = FeaturesGroup(combi)
            for feature in combi:
                # if feature is connected to any feature from the group, always keep it
                if len(feature.get_connected_segments().intersect(combi_group)):
                    continue
                # One feature of the group shall not be of a distance > 3 * scale of an other feature
                if len(combi_group.find(feature, tol=3 * self.scale)) < 2:
                    return False
            return True

        for list1, list2 in nputils.lists_combinations(item.set1(), item.set2(), filter=filter):
            group = MatchingGroup()
            for f1s, f2s in zip(list1, list2):
                group.add(self.get_matchin_item(f1s, f2s))
            yield group

    def log(self, *msg):
        if self.debug:
            print ", :".join([str(k) for k in msg])

    def get_delta_proba(self, segment, delta):
        r = self.config.get("mscsc2_upper_delta_bonus_range")
        if self.upper_delta_info.get_flag(segment) != DeltaInformation.DELTA_MATCH:
            return 1
        return (1 + r / 2.) - r * np.linalg.norm(delta) / (self.get_tolerance([segment]) / self.tol_k)

    def get_nitems_bonus(self, group, total_features):
        r = self.config.get("mscsc2_nitems_bonus_range")
        # return 1
        return (1 - r / 2.) + r * len(group) / total_features

    def optimize(self, indep_item, cb=None):
        results = []
        pmax = 6
        total_features = len(indep_item.set1()) + len(indep_item.set2())
        img2 = self.get_segment_image(indep_item.set2())[0]

        if total_features > pmax:
            print "Warning: high total features to optimize:", str([u.get_segmentid() for u in indep_item.set1()]) + " -> " + str([u.get_segmentid() for u in indep_item.set2()])

        self.log("Optimize group:", str([u.get_segmentid() for u in indep_item.set1()]
                                        ) + " -> " + str([u.get_segmentid() for u in indep_item.set2()]))

        for group in self.get_combinations(indep_item):
            total_coef = 0
            total_matched_features = 0

            for item in group.items():
                coef, scale_delta, delta = self.get_correlation(item.set1(), item.set2(), cb=cb)

                if not self.check_delta(list(item.set1())[0], delta):
                    coef = 0

                matched_features = len(item.set1()) + len(item.set2())
                total_matched_features += matched_features
                total_coef += matched_features * coef

                item.set_delta(scale_delta, delta)
                item.set_correlation(coef)
                if self.debug and total_features <= pmax:
                    print "  ", [u.get_segmentid() for u in item.set1()], "->", [u.get_segmentid() for u in item.set2()], coef, delta, "",
            total_coef += 0.5 * (total_features - total_matched_features)
            total_coef = total_coef / float(total_features)
            # diff = ssd = -1
            if total_coef >= self.correlation_threshold:
                img1 = np.zeros_like(img2)
                proba = []
                for item in group.items():
                    img1 += nputils.shift2d(self.get_segment_image(item.set1())[0], item.get_delta())
                    proba.extend([self.get_delta_proba(k, item.get_scale_delta()) for k in item.set1()])

                ssd = ((img2 - img1) ** 2).sum()
                # diff = ssd / (((img2) ** 2).sum() + ((img1) ** 2).sum())
                # coef = nputils.norm_xcorr_coef(img1, img2)
                proba = np.array(proba).mean()
                total_coef = proba * total_coef * self.get_nitems_bonus(group, total_features)
                if total_features <= pmax:
                    self.log("Proba:", proba, "Coef:", coef, "Total coef:", total_coef)
                # coef = proba * coef

                if cb is not None:
                    cb(group, img1, img2)

                if total_coef >= 0.2:
                    # results.append([group, total_coef, diff, coef, proba])
                    results.append([group, total_coef])
            if self.debug and total_features <= pmax:
                print "=> Matched: ", total_matched_features, "/", total_features, "Coef:", coef, "Diff:", diff, "Total coef:", total_coef, ssd
        if len(results) == 0:
            return MatchingGroup()
        best = max(results, key=lambda res: res[1])
        # best = min(results, key=lambda res: res[2])
        if self.debug:
            print "=> Result: ", ','.join([str(k) for k in best[0].items()]), "Correlation:", best[1]
        return best[0]

    def get_match(self, cb=None, verbose=True):
        if verbose:
            print "\nStart Matching at scale %s. Tolerence: %s" % (self.scale, self.get_tolerance())

        to_match1, to_match2 = self.get_features_to_match()
        to_match2_query = FeaturesQuery(to_match2, coord_modes=self.mode)

        self.segments1.move_features_from_delta_info(self.upper_delta_info)

        match = FeaturesMatch()
        delta_info = DeltaInformation(self.segments1, average_tol=self.scale * 10)

        matching_group = ScaleMatchingGroup()
        cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())

        for segment1 in to_match1.sorted_list(cmp=cmp_intensity):
            match_item = self.find_matches(segment1, to_match2_query)
            if match_item is not None:
                matching_group.add(match_item)

        self.segments1.move_back_to_initial()

        total_correlation = 0

        for indep_item in matching_group.group_independant().items():
            for item in self.optimize(indep_item, cb=cb).items():
                features1 = list(item.set1())
                features2 = list(item.set2())

                if len(features1) == 0 or len(features2) == 0 or item.get_correlation() < self.correlation_threshold:
                    continue

                total_correlation += item.get_correlation()

                get_area = lambda features: float(np.array([k.get_area() for k in features]).sum())

                # print "Match:", item, "Corr:", total_correlation, "Area ratio", get_area(features1) / get_area(features2)

                sort_key = lambda a: a.get_segmentid()

                features1.sort(key=sort_key)
                features2.sort(key=sort_key)

                master_feature1 = features1[0]
                for feature_to_merge in features1[1:]:
                    self.segments1.merge_segments(master_feature1, feature_to_merge)

                master_feature2 = features2[0]
                for feature_to_merge in features2[1:]:
                    self.segments2.merge_segments(master_feature2, feature_to_merge)

                match.add_feature_match(master_feature1, master_feature2)
                delta_info.add_delta(master_feature1, item.get_delta(), self.delta_time)

        if match.size() > 0:
            r1 = build_image(to_match1, delta_info)
            r2 = build_image(to_match2)

            ssd = ((r1.get_data() - r2.get_data()) ** 2).sum()

            if verbose:
                print "Matching features: %s / %s (%s %%)" % (match.size(), to_match1.size(),
                                                              (match.size() / float(to_match1.size()) * 100))
            if verbose:
                print "Correlation:", total_correlation / match.size(), "SSD:", ssd

        return ScaleMatchResult(self.segments1, self.segments2, match, delta_info, self.upper_delta_info)

    def find_matches(self, segment1, segments2):
        tol = self.get_tolerance([segment1])

        seg2 = set(segments2.find(segment1, tol=tol))
        seg1_delta = self.upper_delta_info.get_delta(segment1)
        seg2_inside1 = self.segments2.get_segments_inside2(segment1, delta=seg1_delta, ratio=0.4)
        seg2 |= (set(seg2_inside1) & segments2.get_features())

        if len(seg2) > 0:
            item = MatchingItem((segment1,), seg2)
            # print "Match:", segment1, item
            return item
        return None


class DeltaInfoComparator(object):

    def __init__(self, segments1, segments2):
        self.delta_infos = []
        self.infos = []
        self.segments1 = segments1
        self.segments2 = segments2

    def add_delta_info(self, delta_info, info):
        self.delta_infos.append(delta_info)
        self.infos.append(info)

    def get_correlation_coeffs(self, delta_info):
        img2 = self.segments2.get_img()

        img_region1 = build_image(self.segments1, delta_info)
        img_region2 = imgutils.ImageRegion(img2.data, img_region1.get_index())

        i1 = img_region1.get_region()
        i2 = img_region2.get_region()

        mask = (i1 > 0)

        ssd = (((mask * (i2) - mask * (i1))) ** 2).sum()
        xcor = nputils.norm_xcorr_coef(i1, i2)
        return ssd, xcor

    def normalise_coefs(self, coefs):
        coefs = np.array(coefs)
        coefs -= coefs.min()
        coefs /= (coefs.max())
        return coefs

    def get_best(self):
        if len(self.delta_infos) < 2:
            return self.delta_infos[0], 1
        ssds = []
        xcors = []
        for info, delta_info in zip(self.infos, self.delta_infos):
            ssd, xcor = self.get_correlation_coeffs(delta_info)
            print info, ssd, xcor
            ssds.append(ssd)
            xcors.append(xcor)
        ssds = self.normalise_coefs(ssds)
        xcors = self.normalise_coefs(xcors)
        hybrids = 0.5 * (1 - ssds) + 0.5 * xcors
        if np.isfinite(hybrids).sum() == 0:
            return DeltaInformation([]), 0
        return self.delta_infos[np.nanargmax(hybrids)], hybrids.max()


class ScaleMatcherMSCC2(BaseScaleMatcher):
    ''' Multi scale cross correlation'''

    def __init__(self, segments1, segments2, upper_delta_info=None, match_config=None):
        BaseScaleMatcher.__init__(self, segments1, segments2, upper_delta_info, match_config)

        self._correlation_cache = dict()
        self.max_iteration = 1

    def ids(self, segments):
        return [k.get_segmentid() for k in segments]

    def get_independant_groups(self, segments, distance_threshold):
        if segments.size() < 2:
            return [segments.get_features()]
        dist_fct = lambda a, b: a.distance(b)
        x = list(segments.get_features())
        y = nputils.pdist(x, dist_fct)
        z = linkage(y)
        return nputils.cluster(x, z, distance_threshold, criterion='distance')

    def cross_correlate(self, region1, img2, tol, zoi_center, use_upper_delta=False, cb=None):
        result = []
        tol += 1
        img1 = region1.get_region()
        weight = (region1.get_region() > 0)

        zoi_shape = np.array([(tol) * 3] * 2) + np.array(img1.shape)
        # print "Center1:", zoi_center, img1.shape

        if min(img1.shape) < 2:
            return result

        # shape2 = imgutils.Image((self.segments2.get_labels() > 0).astype(np.float)).zoom(zoi_center, zoi_shape).get_region()

        region2 = img2.zoom(zoi_center, zoi_shape)
        shift = zoi_center - region2.get_center()
        # print "Center2:", region2.get_center(), shift, zoi_shape, region2.get_region().shape
        img2 = region2.get_region()

        # xcorr = nputils.weighted_norm_xcorr2(img2, img1, weight, mode='same')
        xcorr = nputils.norm_xcorr2(img2 > 0, img1, mode='same')
        xcorr = nputils.resize(xcorr, [(tol) * 2] * 2)
        # print "Subpixel:", nputils.fit_gaussian_on_max(xcorr, n=2)

        # xcorr = nputils.norm_xcorr2(shape2, (img1 > 0).astype(np.float), mode='same')
        # xcorr = nputils.resize(xcorr, [(tol) * 2] * 2)

        # ssd = nputils.weighted_ssd_fast(img2, img1, weight, mode='same')
        # ssd = nputils.resize(ssd, [(tol) * 2] * 2)

        # ssd -= ssd.min()
        # ssd /= (ssd.max())

        # corr = 0.33 * xcorr + 0.33 * (1 - ssd) + 0.33 * shape_xcorr
        corr = xcorr
        # corr[xcorr < 0.6] = 0
        # corr[ssd > 0.3] = 0

        if cb is not None:
            cb(img1, img2, corr.copy(), shape_xcorr.copy())

        corr_center = np.array(corr.shape) / 2
        center = corr_center + shift
        threshold = 0.5

        # if use_upper_delta and nputils.check_index(corr, *center) and corr[center[0], center[1]] >= threshold:
        #     # print "Add upper delta:", corr[center[0], center[1]], [0, 0]
        #     result.append(np.array([0, 0]))

        for minimum in nputils.find_peaks(corr, 2, threshold):
            delta = (minimum - center).astype(np.int)
            print "Min:", delta, corr[minimum[0], minimum[1]], np.linalg.norm(delta), tol - 1
            if np.linalg.norm(delta) <= tol - 1 and region1.check_shift(delta):
                result.append(delta)

        return result

    def get_potentiel_deltas(self, segment, others, delta_info, cb=None):
        # print "->", segment.get_segmentid(), "vs", self.ids(others)

        img2 = self.segments2.get_img()
        region1 = build_image([segment], delta_info)

        upper_delta = self.upper_delta_info.get_delta(segment)
        use_upper_delta = self.upper_delta_info.get_flag(segment) == DeltaInformation.DELTA_MATCH
        zoi_center = region1.get_center() - delta_info.get_delta(segment) + upper_delta

        # segments_with_delta = delta_info.get_features(flag=DeltaInformation.DELTA_MATCH)
        others_with_delta = others  # & segments_with_delta.get_features()
        if len(others_with_delta) > 0:
            builder = imgutils.ImageBuilder()
            for segment in others_with_delta:
                region = imgutils.ImageRegion(segment.get_mask(), segment.get_cropped_index())
                region.set_shift(delta_info.get_delta(segment))
                builder.add(region)
            img2 = img2 * (builder.get().get_data() == 0)

        return self.cross_correlate(region1, img2, self.get_tolerance([segment]), zoi_center,
                                    use_upper_delta=use_upper_delta, cb=cb)

    def minimize(self, group, delta_info, cb=None):
        print "Minimize:", group
        set1 = set(group)

        group_deltas = []
        for segment in group:
            others = set1 - set([segment])
            deltas = self.get_potentiel_deltas(segment, others, delta_info, cb=cb)
            # print "Potentiel deltas:", deltas
            group_deltas.append(deltas)

        # expend deltas and check validity
        segments_deltas = []
        segments = []
        for segment, deltas in zip(group, group_deltas):
            seg_deltas = []
            for delta in deltas:
                delta = self.upper_delta_info.get_delta(segment) + delta

                if not self.check_delta(segment, delta):
                    continue

                seg_deltas.append(delta)

            if len(seg_deltas) == 0:
                continue
            segments_deltas.append(seg_deltas)
            segments.append(segment)

        if len(segments_deltas) == 0:
            return delta_info, None

        # print "Potentiels:", segments_deltas

        comparator = DeltaInfoComparator(self.segments1, self.segments2)

        result = None
        for deltas_combi in itertools.product(*segments_deltas):
            temp_delta_info = delta_info.copy()
            info = []
            for segment, delta in zip(segments, deltas_combi):
                if delta is not None:
                    temp_delta_info.add_delta(segment, delta, self.delta_time)
                    info.append((segment.get_segmentid(), list(temp_delta_info.get_delta(segment))))

            comparator.add_delta_info(temp_delta_info, info)
            # ssd = self.get_correlation_coef(set1, temp_delta_info)

            # if result is None or ssd < result[1]:
            #     result = [temp_delta_info, ssd, info]

        # print "Result:", result[2], result[1]

        # return result[0], ssd
        return comparator.get_best()

    def get_correlation_coef(self, segments1, delta_info):
        img2 = self.segments2.get_img()

        img_region1 = build_image(segments1, delta_info)
        img_region2 = imgutils.ImageRegion(img2.data, img_region1.get_index())

        i1 = img_region1.get_region()
        i2 = img_region2.get_region()

        mask = (i1 > 0)
        # zero_mean = lambda a: a - a.mean()
        # norm = lambda a: zero_mean(nputils.safe_norm(a, (a > 0)))

        ssd = (((mask * (i2) - mask * (i1))) ** 2).sum()
        # ssd = nputils.norm_xcorr_coef(mask * i1, mask * i2)
        print "Result:", ssd, nputils.norm_xcorr_coef(i1, i2),\
              nputils.norm_xcorr_coef(mask * i1, mask * i2), (((i2 - i1)) ** 2).sum(),\
              (((i2 * mask - i1 * mask)) ** 2).sum()
        return ssd

    def merge_small_features(self):
        for feature in self.segments1.get_features():
            if feature.get_area() < 0.7 * self.scale * np.pi ** 2:
                interface = self.segments1.get_interface(feature.get_segmentid())
                l = [(id, max(self.segments1.get_values(coords))) for (id, coords) in interface.items()]
                nearest_id, value = max(l, key=lambda v: v[1])
                nearest_segment = self.segments1.get_segment_from_id(nearest_id)
                if nearest_segment is not None:
                    self.segments1.merge_segments(nearest_segment, feature)
                elif feature.get_area() < 0.4 * self.scale * np.pi ** 2:
                    self.segments1.remove_feature(feature)

    def get_match(self, cb=None, verbose=True):
        print "\nStart Matching at scale %s." % (self.scale)

        to_match1, to_match2 = self.get_features_to_match()

        if to_match1.size() == 0:
            return ScaleMatchResult(self.segments1, self.segments2, FeaturesMatch(), DeltaInformation(self.segments1), self.upper_delta_info)

        img2 = self.segments2.get_img()

        match = FeaturesMatch()
        delta_info = self.upper_delta_info.copy()
        delta_info.average_tol = self.scale * 10

        for feature in delta_info.get_features():
            if feature not in to_match1:
                delta_info.remove_feature(feature)
            delta_info.set_flag(feature, DeltaInformation.DELTA_COMPUTED)

        current_ssd = 1e99

        for i in range(self.max_iteration):
            to_match1.move_features_from_delta_info(delta_info)
            groups = self.get_independant_groups(to_match1, 2 * self.scale)
            groups_sort_key = lambda group: min(self.ids(group))

            groups.sort(key=groups_sort_key)
            new_delta_info = delta_info.copy()

            for group in groups:
                # if len(group) == 1:
                new_delta_info, ssd = self.minimize(group, new_delta_info, cb=cb)
                # print "Result:"
                # for feature in new_delta_info.get_features(DeltaInformation.DELTA_MATCH):
                #     print feature.get_segmentid(), new_delta_info.get_delta(feature)
                # else:
                #     current_group_ssd = 0
                #     for i in range(self.max_iteration):
                #         temp_delta_info, ssd = self.minimize(group, new_delta_info, cb=cb)
                #         print "Result iteration", i, ":", ssd, "(vs", current_group_ssd, ")\n"

                #         if ssd is None or ssd <= current_group_ssd:
                #             break

                #         current_group_ssd = ssd
                #         new_delta_info = temp_delta_info

            # if to_match1.size() > 1:
            #     segments_with_delta = new_delta_info.get_features(DeltaInformation.DELTA_MATCH)
            #     img1 = build_image(to_match1.get_features(), new_delta_info)
            #     for segment in segments_with_delta.sorted_list(key=lambda s: s.get_intensity()):
            #         contribution = self.get_contribuation(segment, img1, new_delta_info)
            #         if contribution < 0:
            #             old_delta = self.upper_delta_info.get_delta(segment)
            #             print "-> Contribution below threshold", segment, old_delta
            #             new_delta_info.add_delta(segment, old_delta, DeltaInformation.DELTA_COMPUTED)

            img1 = build_image(self.segments1.get_features(), new_delta_info)
            ssd = ((img2.get_data() - img1.get_data()) ** 2).sum()
            print "Result iteration", i, ":", ssd, "(vs", current_ssd, ")\n"

            to_match1.move_back_to_initial()

            if ssd >= current_ssd:
                break

            current_ssd = ssd
            delta_info = new_delta_info

        # find matching segments
        for segment in delta_info.get_features(DeltaInformation.DELTA_MATCH):
            delta = delta_info.get_delta(segment)
            segment2 = self.segments2.get_overlapping_segment(segment, delta=delta, min_ratio=0.4)
            print "Match:", segment, segment2, delta
            if segment2 is not None:
                match.add_feature_match(segment, segment2)
            # else:
            #     old_delta = self.upper_delta_info.get_delta(segment)
            #     delta_info.add_delta(segment, old_delta, self.delta_time, DeltaInformation.DELTA_COMPUTED)

        return ScaleMatchResult(self.segments1, self.segments2, match, delta_info, self.upper_delta_info)


class ScaleMatcherMSCC(BaseScaleMatcher):
    ''' Multi scale cross correlation'''

    def __init__(self, segments1, segments2, upper_delta_info=None, match_config=None):
        BaseScaleMatcher.__init__(self, segments1, segments2, upper_delta_info, match_config)

        self._correlation_cache = dict()
        self.max_iteration = 5

    def ids(self, segments):
        return [k.get_segmentid() for k in segments]

    def get_independant_groups(self, segments, distance_threshold):
        if segments.size() < 2:
            return [segments.get_features()]
        dist_fct = lambda a, b: a.distance(b)
        x = list(segments.get_features())
        y = nputils.pdist(x, dist_fct)
        z = linkage(y)
        return nputils.cluster(x, z, distance_threshold, criterion='distance')

    def cross_correlate(self, region1, img2, tol, zoi_center, use_upper_delta=False, cb=None):
        result = []
        tol += 1
        img1 = region1.get_region()
        weight = (region1.get_region() > 0)

        zoi_shape = np.array([(tol) * 3] * 2) + np.array(img1.shape)
        # print "Center1:", zoi_center, img1.shape

        if min(img1.shape) < 2:
            return result

        # shape2 = imgutils.Image((self.segments2.get_labels() > 0).astype(np.float)).zoom(zoi_center, zoi_shape).get_region()

        region2 = img2.get_region(zoi_center, zoi_shape)
        shift = zoi_center - region2.get_center()
        # print "Center2:", region2.get_center(), shift, zoi_shape, region2.get_region().shape
        img2 = region2.get_region()

        # xcorr = nputils.weighted_norm_xcorr2(img2, img1, weight, mode='same')
        xcorr = nputils.norm_xcorr2(img2, img1, mode='same')
        xcorr = nputils.resize(xcorr, [(tol) * 2] * 2)
        # print "Subpixel:", nputils.fit_gaussian_on_max(xcorr, n=2)

        # xcorr = nputils.norm_xcorr2(shape2, (img1 > 0).astype(np.float), mode='same')
        # xcorr = nputils.resize(xcorr, [(tol) * 2] * 2)

        # ssd = nputils.weighted_ssd_fast(img2, img1, weight, mode='same')
        # ssd = nputils.resize(ssd, [(tol) * 2] * 2)

        # ssd -= ssd.min()
        # ssd /= (ssd.max())

        ssd = nputils.ssd_fast(img2, img1, mode='same')
        ssd = nputils.resize(ssd, [(tol) * 2] * 2)
        ssd[np.isinf(ssd)] = np.nan
        ssd[np.isnan(ssd)] = np.nanmax(ssd)
        ssd[ssd > np.median(ssd)] = np.median(ssd)
        ssd -= 0
        ssd /= ssd.max()
        ssd = 1 - ssd
        ssd[ssd < 0.3] = 0

        # corr = 0.33 * xcorr + 0.33 * (1 - ssd) + 0.33 * shape_xcorr
        corr = ssd
        # corr[xcorr < 0.6] = 0
        # corr[ssd > 0.3] = 0
        corr[np.isnan(corr)] = 0
        corr[np.isinf(corr)] = 0

        if cb is not None:
            cb(str(region1.get_center()), img1, img2, corr.copy())

        corr_center = np.array(corr.shape) / 2
        center = corr_center + shift
        threshold = 0.3

        # if use_upper_delta and nputils.check_index(corr, *center) and corr[center[0], center[1]] >= threshold:
        #     # print "Add upper delta:", corr[center[0], center[1]], [0, 0]
        #     result.append(np.array([0, 0]))

        for minimum in nputils.find_peaks(corr, 2, threshold):
            delta = (minimum - center).astype(np.int)
            coef = corr[minimum[0], minimum[1]]
            if np.linalg.norm(delta) <= tol - 1 and region1.check_shift(delta) and np.isfinite(coef):
                # print "Min:", delta, coef, np.linalg.norm(delta), tol - 1
                result.append(delta)

        return result

    def get_potentiel_deltas(self, segment, others, delta_info, cb=None):
        # print "->", segment.get_segmentid(), "vs", self.ids(others)

        img2 = self.segments2.get_img()
        region1 = build_image([segment], delta_info)

        upper_delta = self.upper_delta_info.get_delta(segment)
        use_upper_delta = self.upper_delta_info.get_flag(segment) == DeltaInformation.DELTA_MATCH
        zoi_center = region1.get_center() - delta_info.get_delta(segment) + upper_delta

        # segments_with_delta = delta_info.get_features(flag=DeltaInformation.DELTA_MATCH)
        others_with_delta = others  # & segments_with_delta.get_features()
        if len(others_with_delta) > 0:
            builder = imgutils.ImageBuilder()
            for segment in others_with_delta:
                region = imgutils.ImageRegion(segment.get_mask(), segment.get_cropped_index())
                region.set_shift(delta_info.get_delta(segment))
                builder.add(region)
            img2 = img2 * (builder.get().get_data() == 0)

        return self.cross_correlate(region1, img2, self.get_tolerance([segment]), zoi_center,
                                    use_upper_delta=use_upper_delta, cb=cb)

    def minimize(self, group, delta_info, cb=None):
        # print "Minimize:", group
        set1 = set(group)

        group_deltas = []
        for segment in group:
            others = set1 - set([segment])
            deltas = self.get_potentiel_deltas(segment, others, delta_info, cb=cb)
            # print "Potentiel deltas:", deltas
            group_deltas.append(deltas)

        # expend deltas and check validity
        segments_deltas = []
        segments = []
        for segment, deltas in zip(group, group_deltas):
            seg_deltas = []
            for delta in deltas:
                delta = self.upper_delta_info.get_delta(segment) + delta

                if not self.check_delta(segment, delta):
                    continue

                seg_deltas.append(delta)

            if len(seg_deltas) == 0:
                continue
            segments_deltas.append(seg_deltas)
            segments.append(segment)

        if len(segments_deltas) == 0:
            return delta_info, None

        # print "Potentiels:", segments_deltas

        comparator = DeltaInfoComparator(self.segments1, self.segments2)

        result = None
        for deltas_combi in itertools.product(*segments_deltas):
            temp_delta_info = delta_info.copy()
            info = []
            for segment, delta in zip(segments, deltas_combi):
                if delta is not None:
                    temp_delta_info.add_delta(segment, delta, self.delta_time)
                    info.append((segment.get_segmentid(), list(temp_delta_info.get_delta(segment))))

            comparator.add_delta_info(temp_delta_info, info)
            # ssd = self.get_correlation_coef(set1, temp_delta_info)

            # if result is None or ssd < result[1]:
            #     result = [temp_delta_info, ssd, info]

        # print "Result:", result[2], result[1]

        # return result[0], ssd
        return comparator.get_best()

    def get_correlation_coef(self, segments1, delta_info):
        img2 = self.segments2.get_img()

        img_region1 = build_image(segments1, delta_info)
        img_region2 = imgutils.ImageRegion(img2.data, img_region1.get_index())

        i1 = img_region1.get_region()
        i2 = img_region2.get_region()

        mask = (i1 > 0)
        # zero_mean = lambda a: a - a.mean()
        # norm = lambda a: zero_mean(nputils.safe_norm(a, (a > 0)))

        ssd = (((mask * (i2) - mask * (i1))) ** 2).sum()
        # ssd = nputils.norm_xcorr_coef(mask * i1, mask * i2)
        # print "Result:", ssd, nputils.norm_xcorr_coef(i1, i2),\
        #       nputils.norm_xcorr_coef(mask * i1, mask * i2), (((i2 - i1)) ** 2).sum(),\
        #       (((i2 * mask - i1 * mask)) ** 2).sum()
        return ssd

    def merge_small_features(self):
        for feature in self.segments1.get_features():
            if feature.get_area() < 0.7 * self.scale * np.pi ** 2:
                interface = self.segments1.get_interface(feature.get_segmentid())
                l = [(id, max(self.segments1.get_values(coords))) for (id, coords) in interface.items()]
                nearest_id, value = max(l, key=lambda v: v[1])
                nearest_segment = self.segments1.get_segment_from_id(nearest_id)
                if nearest_segment is not None:
                    self.segments1.merge_segments(nearest_segment, feature)
                elif feature.get_area() < 0.4 * self.scale * np.pi ** 2:
                    self.segments1.remove_feature(feature)

    def get_contribuation(self, segment, img1, delta_info):
        seg_img1 = build_image([segment], delta_info)
        img2 = self.segments2.get_img()
        mask = (seg_img1.get_region() > 0)

        # zero_mean = lambda a: a - a.mean()
        # norm = lambda a: zero_mean(nputils.safe_norm(a, (a > 0)))

        a2 = imgutils.ImageRegion(img1.get_img(), seg_img1.get_index()).get_region()
        a1 = a2 - seg_img1.get_region()
        b = imgutils.ImageRegion(img2.get_img(), seg_img1.get_index()).get_region()
        corr_without = nputils.norm_xcorr_coef(mask * b, mask * a1)
        corr_with = nputils.norm_xcorr_coef(mask * b, mask * a2)

        # error_without = ((a1 - b) ** 2).sum()
        # error_with = ((a2 - b) ** 2).sum()
        # print "Contribution (not normalised) from ", segment, ":", error_without, error_with
        # error_without = ((norm(a1) - norm(b)) ** 2).sum()
        # error_with = ((norm(a2) - norm(b)) ** 2).sum()
        # print "Contribution (normalised) from ", segment, ":", error_without,\
        #       error_with, nputils.norm_xcorr_coef(b, a1), nputils.norm_xcorr_coef(b, a2)
        # error_without = (((norm(a1 * mask) - norm(b * mask))) ** 2).sum()
        # error_with = (((norm(a2 * mask) - norm(b * mask))) ** 2).sum()
        # print "Contribution (normalised, masked) from ", segment, ":",\
        #       delta_info.get_delta(segment), corr_without, corr_with
        # # return (error_without - error_with) / error_without
        # print "Criteria:", np.nanmin([corr_with - corr_without, (error_without - error_with) / error_without])
        return corr_with - corr_without

    def get_match(self, cb=None, verbose=True):
        print "\nStart Matching at scale %s." % (self.scale)

        # self.merge_small_features()

        if self.config.get("ignore_features_at_border"):
            features_at_the_border1 = self.get_features_at_the_border()[0]
            to_match1 = FeaturesGroup(self.segments1.get_features() - features_at_the_border1.get_features())
        else:
            to_match1 = self.segments1

        if to_match1.size() == 0:
            return ScaleMatchResult(self.segments1, self.segments2, FeaturesMatch(), DeltaInformation(self.segments1), self.upper_delta_info)

        img2 = self.segments2.get_img()

        match = FeaturesMatch()
        delta_info = self.upper_delta_info.copy()
        delta_info.average_tol = self.scale * 10

        for feature in delta_info.get_features():
            if feature not in to_match1:
                delta_info.remove_feature(feature)
            delta_info.set_flag(feature, DeltaInformation.DELTA_COMPUTED)

        current_ssd = 1e99

        for i in range(self.max_iteration):
            to_match1.move_features_from_delta_info(delta_info)
            groups = self.get_independant_groups(to_match1, 2 * self.scale)
            groups_sort_key = lambda group: min(self.ids(group))

            groups.sort(key=groups_sort_key)
            new_delta_info = delta_info.copy()

            for group in groups:
                # if len(group) == 1:
                new_delta_info, ssd = self.minimize(group, new_delta_info, cb=cb)
                # print "Result:"
                # for feature in new_delta_info.get_features(DeltaInformation.DELTA_MATCH):
                #     print feature.get_segmentid(), new_delta_info.get_delta(feature)
                # else:
                #     current_group_ssd = 0
                #     for i in range(self.max_iteration):
                #         temp_delta_info, ssd = self.minimize(group, new_delta_info, cb=cb)
                #         print "Result iteration", i, ":", ssd, "(vs", current_group_ssd, ")\n"

                #         if ssd is None or ssd <= current_group_ssd:
                #             break

                #         current_group_ssd = ssd
                #         new_delta_info = temp_delta_info

            # if to_match1.size() > 1:
            #     segments_with_delta = new_delta_info.get_features(DeltaInformation.DELTA_MATCH)
            #     img1 = build_image(to_match1.get_features(), new_delta_info)
            #     for segment in segments_with_delta.sorted_list(key=lambda s: s.get_intensity()):
            #         contribution = self.get_contribuation(segment, img1, new_delta_info)
            #         if contribution < 0:
            #             old_delta = self.upper_delta_info.get_delta(segment)
            #             print "-> Contribution below threshold", segment, old_delta
            #             new_delta_info.add_delta(segment, old_delta, DeltaInformation.DELTA_COMPUTED)

            img1 = build_image(self.segments1.get_features(), new_delta_info)
            ssd = ((img2.get_data() - img1.get_data()) ** 2).sum()
            # print "Result iteration", i, ":", ssd, "(vs", current_ssd, ")\n"

            to_match1.move_back_to_initial()

            if ssd >= current_ssd:
                break

            current_ssd = ssd
            delta_info = new_delta_info

        # find matching segments
        for segment in delta_info.get_features(DeltaInformation.DELTA_MATCH):
            delta = delta_info.get_delta(segment)
            segment2 = self.segments2.get_overlapping_segment(segment, delta=delta, min_ratio=0.2)
            print "Match:", segment, segment2, delta
            if segment2 is not None:
                match.add_feature_match(segment, segment2)
            else:
                old_delta = self.upper_delta_info.get_delta(segment)
                delta_info.add_delta(segment, old_delta, self.delta_time, DeltaInformation.DELTA_COMPUTED)

        return ScaleMatchResult(self.segments1, self.segments2, match, delta_info, self.upper_delta_info)


class NullFeature(Feature):

    def __init__(self):
        super(NullFeature, self).__init__([0, 0], 0)


class MatchingItem(object):

    def __init__(self, set1, set2, delta=[0, 0], scale_delta=[0, 0], correlation=0):
        self._set1 = frozenset(set1)
        self._set2 = frozenset(set2)
        self._delta = delta
        self._scale_delta = scale_delta
        self._correlation = correlation

    def __str__(self):
        return str([u.get_segmentid() for u in self.set1()]) + "->" + str([u.get_segmentid() for u in self.set2()])
        # return "%s -> %s" % (self._set1, self._set2)

    def set1(self):
        return self._set1

    def set2(self):
        return self._set2

    def merge(self, item):
        self._set1 |= item.set1()
        self._set2 |= item.set2()

    def set_delta(self, scale_delta, delta):
        self._scale_delta = scale_delta
        self._delta = delta

    def get_delta(self):
        return self._delta

    def build_correlation(self, mode, inital_delta, delta_tol):
        if self._delta is None:
            r1 = build_image(self.set1())
            r1.set_shift(np.round(inital_delta))
            r2 = build_image(self.set2())

            self._delta_scale = r2.get_center_of_mass() - r1.get_center_of_mass()
            self._delta = inital_delta + self._delta_scale

            if np.linalg.norm(delta) > delta_tol:
                pass

            r1.set_shift(np.round(inital_delta + delta))

            img1 = r1.get_data()
            img2 = r2.get_data()
            self._correlation = nputils.norm_xcorr_coef(img1, img2)
            self.intensity_ratio = img1.sum() / img2.sum()
            self.area_ratio = (img1 > 0).sum() / (img2 > 0).sum()

    def get_correlation(self):
        return self._correlation

    def get_scale_delta(self):
        return self._scale_delta

    def set_correlation(self, value):
        self._correlation = value

    def get_correlation(self):
        return self._correlation

    def size(self):
        return len(self._set1) + len(self._set2)


class MatchingGroup(object):

    def __init__(self):
        self.list = []

    def __len__(self):
        return len(self.list)

    def items(self):
        return self.list

    def add(self, item):
        self.list.append(item)

    def remove(self, item):
        self.list.remove(item)

    def sets1(self):
        return [k.set1() for k in self.list]

    def sets2(self):
        return [k.set2() for k in self.list]

    def all1(self):
        return reduce(lambda x, y: set(x) | set(y), self.sets1())

    def all2(self):
        return reduce(lambda x, y: set(x) | set(y), self.sets2())

    def get_correlation(self):
        pass


class ScaleMatchingGroup(MatchingGroup):

    def __init__(self):
        super(ScaleMatchingGroup, self).__init__()

    def group_independant(self):
        newgroup = MatchingGroup()
        for item in self.items():
            matched_item = None
            for itemnewgroup in newgroup.items()[:]:
                if not itemnewgroup.set2().isdisjoint(item.set2()):
                    if matched_item is None:
                        itemnewgroup.merge(item)
                        matched_item = itemnewgroup
                    else:
                        matched_item.merge(itemnewgroup)
                        newgroup.remove(itemnewgroup)
            if matched_item is None:
                newgroup.add(item)
        return newgroup


def build_delta_information_scale2(segments, scale_match_result, average_tol_factor=10):
    delta_info = DeltaInformation(segments, segments.get_scale() * average_tol_factor)
    if scale_match_result is None:
        return delta_info
    upper_delta_info = scale_match_result.get_delta_info()
    upper_segments_query = FeaturesQuery(scale_match_result.get_features1())
    upper_segments_with_delta = upper_delta_info.get_features(flag=DeltaInformation.DELTA_MATCH)
    for segment in segments.get_features():
        upper_segments_inside = FeaturesGroup(upper_segments_query.find(segment, tol=segments.get_scale()))
        # upper_segments_inside = upper_segments.get_segments_inside2(segment, ratio=0.7)
        upper_segments_inside_with_delta = FeaturesGroup(
            upper_segments_inside.get_features() & upper_segments_with_delta.get_features())
        # print "Build delta info for:", segment, upper_segments_inside_with_delta
        if upper_segments_inside_with_delta.size() > 0:
            upper_delta = upper_delta_info.get_deltas(upper_segments_inside_with_delta).mean(axis=0)
            upper_delta = upper_delta_info.get_average_delta_information(segment, upper_segments_inside_with_delta)
            if upper_delta is not None:
                # print "Upper delta:", upper_delta
                delta_info.add_delta(segment, np.round(upper_delta), scale_match_result.get_delta_time(),
                                     DeltaInformation.DELTA_MATCH)
    delta_info.complete_with_average_delta()
    # delta_info.dump()
    return delta_info


class MatcherConfiguration(nputils.BaseConfiguration):

    def __init__(self):
        data = [
            ["use_upper_info", True, "Use Pyramidal scheme for matching", validator_is(bool), str2bool, str, 0],
            ["upper_info_average_tol_factor", 10, "Tolerance factor that define the number of features for average upper delta calculation",
             validator_is(int), int, str, 1],
            ["mscsc2_upper_delta_bonus_range", 0.4, "Bonus for delta close to upper delta",
             validator_in_range(0, 1), float, str, 1],
            ["mscsc2_nitems_bonus_range", 0.4, "Bonus for fewer merge", validator_in_range(0, 1), float, str, 1],
            ["simple_merge", True, "MSCI: use segment merging", validator_is(bool), str2bool, str, 1],
            ["msci_coord_mode", 'com', "Coord mode used to determine the delta",
                validator_in(['lm', 'com']), str, str, 1],
            ["correlation_threshold", 0.65, "Correlation threshold", validator_in_range(0, 1), float, str, 0],
            ["ignore_features_at_border", False, "Ignore feature art border for matching",
             validator_is(bool), str2bool, str, 0],
            ["features_at_border_k1", 0.5, "At border param k1", validator_in_range(0, 2), float, str, 1],
            ["features_at_border_k2", 0.25, "At border param k2", validator_in_range(0, 2), float, str, 1],
            ["features_at_border_k3", 0.25, "At border param k3", validator_in_range(0, 2), float, str, 1],
            ["maximum_delta", 40, "Deprecated: use delta_range_filter", None, None, None, 2],
            ["range_delta_x", [-40, 40], "Deprecated: use delta_range_filter", None, None, None, 2],
            ["range_delta_y", [-40, 40], "Deprecated: use delta_range_filter", None, None, None, 2],
            ["increase_tol_for_no_input_delta", True, "Increase tolerance when no initial guess",
             validator_is(bool), str2bool, str, 1],
            ["delta_range_filter", None, "Delta range filter", validator_is(nputils.AbstractFilter),
             jp.decode, jp.encode, 0],
            ["mscsc_max_merge", 3, "MSCSC: Maximum number of segment merged", validator_in_range(1, 5, instance=int),
             int, str, 1],
            ["tolerance_factor", 1, "Tolerance factor", validator_in_range(0, 4), float, str, 0],
            ["method_klass", ScaleMatcherMSCSC2, "Matching method", validator_is_class(BaseScaleMatcher),
             lambda s: jp.decode(str2jsonclass(s)), jp.encode, 1],
            ["no_input_no_match_scales", [], "List of scales at which no match is performed if no initial guess",
             validator_is(list), jp.decode, jp.encode, 1],
            ["min_scale_tolerance", {2: 4, 3: 4, 4: 6}, "Per scale tolerance in pixel", validator_is(dict),
             jp.decode, jp.encode, 1],
            ["find_distance_mode", "min", "Method used for distance measure", validator_is(str), str, str, 1],
            ["mscsc2_smooth", True, "Apply smooth on merged features before correlation",
                validator_is(bool), str2bool, str, 1],
        ]

        super(MatcherConfiguration, self).__init__(data, title="Matcher configuration")


class ScaleMatchResult:

    def __init__(self, segments1, segments2, match, delta_info, upper_delta_info, correlation=None):
        self.segments1 = segments1
        self.segments2 = segments2
        self.match = match
        self.delta_info = delta_info
        self.upper_delta_info = upper_delta_info
        self.correlation = correlation

    def merge(self, other):
        self.segments1.merge(other.segments1)
        self.segments2.merge(other.segments2)
        self.match.merge(other.match)
        self.delta_info.merge(other.delta_info)
        if self.upper_delta_info is not None:
            self.upper_delta_info.merge(other.upper_delta_info)

    def get_all(self, feature_filter=None):
        if feature_filter is None:
            return (self.segments1, self.segments2, self.match, self.delta_info)

        new_match = self.match.get_filtered(feature_filter)
        new_delta_info = self.delta_info.get_filtered(feature_filter)

        segments1 = self.segments1.copy()
        segments1.set_features(new_match.get_ones())
        segments2 = self.segments2.copy()
        segments2.set_features(new_match.get_twos())

        return (segments1, segments2, new_match, new_delta_info)

    def get_match(self):
        return self.match

    def get_scale(self):
        return self.segments1.get_scale()

    def get_epoch(self):
        return self.segments1.get_epoch()

    def get_delta_info(self):
        return self.delta_info

    def get_features1(self):
        return self.segments1

    def get_features2(self):
        return self.segments2

    def get_delta_time(self):
        return self.segments2.get_epoch() - self.segments1.get_epoch()

    def get_upper_delta_info(self):
        return self.upper_delta_info


class MultiScaleMatchResult(AbstractKeyList):

    def get_epoch(self):
        if len(self) > 0:
            return self[0].get_features1().get_epoch()
        return None

    def get_item_key(self, item):
        return item.get_scale()

    def get_scales(self):
        return self.get_keys()

    def get_scale(self, scale):
        return self.get_key(scale)


class MultiScaleMatchResultSet(AbstractKeyList):

    def get_item_key(self, item):
        return item.get_epoch()

    def get_epochs(self):
        return self.get_keys()

    def get_epoch(self, epoch):
        return self.get_key(epoch)


class ImageMatcher(object):

    def __init__(self, finder_config, match_config, filter=None):
        self.finder_config = finder_config
        self.match_config = match_config
        self.filter = filter

    def get_match_scale(self, features1, features2, upper_delta_info,
                        do_merge=True, cb=None, verbose=True):
        klass = self.match_config.get("method_klass")
        matcher = klass(features1, features2, upper_delta_info, self.match_config)
        result = matcher.get_match(cb=cb, verbose=verbose)

        features1.discard_cache()
        features2.discard_cache()

        return result

    def get_match(self, finder_res1, finder_res2, cb=None, verbose=True):
        upper_delta_info = None
        result = MultiScaleMatchResult()
        average_tol_factor = self.match_config.get("upper_info_average_tol_factor")

        for i in range(len(finder_res1) - 1, -1, -1):
            if self.match_config.get("use_upper_info"):
                upper_delta_info = build_delta_information_scale2(finder_res1[i],
                                                                  result[-1] if len(result) > 0 else None,
                                                                  average_tol_factor=average_tol_factor)
            scale_match_result = self.get_match_scale(finder_res1[i], finder_res2[i],
                                                      upper_delta_info, i > 0, cb=cb, verbose=verbose)
            result.append(scale_match_result)

        result.sort(key=lambda k: k.get_scale())

        return result

    def find_features(self, img, bg):
        finder = FeaturesFinder(img, bg, self.finder_config, filter=self.filter)
        return finder.execute()

    def execute(self, img1, bg1, img2, bg2):
        print "Analysing image 1 ..."
        res1 = self.find_features(img1, bg1)

        print "Analysing image 2 ..."
        res2 = self.find_features(img2, bg2)

        return self.get_match(res1, res2)
