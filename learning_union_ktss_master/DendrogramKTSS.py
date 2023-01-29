﻿import warnings

from scipy.cluster.hierarchy import linkage, to_tree, fcluster, _append_singleton_leaf_node, _link_line_colors, \
    _plot_dendrogram, _append_nonsingleton_leaf_node, _append_contraction_marks
from six import string_types
import numpy as np
from sys import maxsize as MAX_DIST


def _dendrogram_calculate_info_ktss(Z, p, truncate_mode,
                               color_threshold=np.inf, get_leaves=True,
                               orientation='top', labels=None,
                               count_sort=False, distance_sort=False,
                               show_leaf_counts=False, i=-1, iv=0.0,
                               ivl=[], n=0, icoord_list=[], dcoord_list=[],
                               lvs=None, mhr=False,
                               current_color=[], color_list=[],
                               currently_below_threshold=[],
                               leaf_label_func=None, level=0,
                               contraction_marks=None,
                               link_color_func=None,
                               above_threshold_color='b'):
    """
    Calculate the endpoints of the links as well as the labels for the
    the dendrogram rooted at the node with index i. iv is the independent
    variable value to plot the left-most leaf node below the root node i
    (if orientation='top', this would be the left-most x value where the
    plotting of this root node i and its descendents should begin).

    ivl is a list to store the labels of the leaf nodes. The leaf_label_func
    is called whenever ivl != None, labels == None, and
    leaf_label_func != None. When ivl != None and labels != None, the
    labels list is used only for labeling the leaf nodes. When
    ivl == None, no labels are generated for leaf nodes.

    When get_leaves==True, a list of leaves is built as they are visited
    in the dendrogram.

    Returns a tuple with l being the independent variable coordinate that
    corresponds to the midpoint of cluster to the left of cluster i if
    i is non-singleton, otherwise the independent coordinate of the leaf
    node if i is a leaf node.

    Returns
    -------
    A tuple (left, w, h, md), where:

      * left is the independent variable coordinate of the center of the
        the U of the subtree

      * w is the amount of space used for the subtree (in independent
        variable units)

      * h is the height of the subtree in dependent variable units

      * md is the ``max(Z[*,2]``) for all nodes ``*`` below and including
        the target node.

    """
    if n == 0:
        raise ValueError("Invalid singleton cluster count n.")

    if i == -1:
        raise ValueError("Invalid root cluster index i.")

    if truncate_mode == 'lastp':
        # If the node is a leaf node but corresponds to a non-singleton
        # cluster, its label is either the empty string or the number of
        # original observations belonging to cluster i.
        if 2*n - p > i >= n:
            d = Z[i - n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts)
            if contraction_marks is not None:
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif truncate_mode == 'level':
        if i > n and level > p:
            d = Z[i - n, 2]
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts)
            if contraction_marks is not None:
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks)
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels)
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif truncate_mode in ('mlab',):
        msg = "Mode 'mlab' is deprecated in scipy 0.19.0 (it never worked)."
        warnings.warn(msg, DeprecationWarning)

    # Otherwise, only truncate if we have a leaf node.
    #
    # Only place leaves if they correspond to original observations.
    if i < n:
        _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                    leaf_label_func, i, labels)
        return (iv + 5.0, 10.0, 0.0, 0.0)

    # !!! Otherwise, we don't have a leaf node, so work on plotting a
    # non-leaf node.
    # Actual indices of a and b
    aa = int(Z[i - n, 0])
    ab = int(Z[i - n, 1])
    if aa > n:
        # The number of singletons below cluster a
        na = Z[aa - n, 3]
        # The distance between a's two direct children.
        da = Z[aa - n, 2]
    else:
        na = 1
        da = 0.0
    if ab > n:
        nb = Z[ab - n, 3]
        db = Z[ab - n, 2]
    else:
        nb = 1
        db = 0.0

    if count_sort == 'ascending' or count_sort:
        # If a has a count greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if na > nb:
            # The cluster index to draw to the left (ua) will be ab
            # and the one to draw to the right (ub) will be aa
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif count_sort == 'descending':
        # If a has a count less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if na > nb:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    elif distance_sort == 'ascending' or distance_sort:
        # If a has a distance greater than b, it and its descendents should
        # be drawn to the right. Otherwise, to the left.
        if da > db:
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif distance_sort == 'descending':
        # If a has a distance less than or equal to b, it and its
        # descendents should be drawn to the left. Otherwise, to
        # the right.
        if da > db:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    else:
        ua = aa
        ub = ab

    # Updated iv variable and the amount of space used.
    (uiva, uwa, uah, uamd) = \
        _dendrogram_calculate_info_ktss(
            Z=Z, p=p,
            truncate_mode=truncate_mode,
            color_threshold=color_threshold,
            get_leaves=get_leaves,
            orientation=orientation,
            labels=labels,
            count_sort=count_sort,
            distance_sort=distance_sort,
            show_leaf_counts=show_leaf_counts,
            i=ua, iv=iv, ivl=ivl, n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list, lvs=lvs,
            current_color=current_color,
            color_list=color_list,
            currently_below_threshold=currently_below_threshold,
            leaf_label_func=leaf_label_func,
            level=level + 1, contraction_marks=contraction_marks,
            link_color_func=link_color_func,
            above_threshold_color=above_threshold_color)

    h = Z[i - n, 2]
    if h >= color_threshold or color_threshold <= 0:
        c = above_threshold_color

        if currently_below_threshold[0]:
            current_color[0] = (current_color[0] + 1) % len(_link_line_colors)
        currently_below_threshold[0] = False
    else:
        currently_below_threshold[0] = True
        c = _link_line_colors[current_color[0]]

    (uivb, uwb, ubh, ubmd) = \
        _dendrogram_calculate_info_ktss(
            Z=Z, p=p,
            truncate_mode=truncate_mode,
            color_threshold=color_threshold,
            get_leaves=get_leaves,
            orientation=orientation,
            labels=labels,
            count_sort=count_sort,
            distance_sort=distance_sort,
            show_leaf_counts=show_leaf_counts,
            i=ub, iv=iv + uwa, ivl=ivl, n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list, lvs=lvs,
            current_color=current_color,
            color_list=color_list,
            currently_below_threshold=currently_below_threshold,
            leaf_label_func=leaf_label_func,
            level=level + 1, contraction_marks=contraction_marks,
            link_color_func=link_color_func,
            above_threshold_color=above_threshold_color)

    max_dist = max(uamd, ubmd, h)
    if h < MAX_DIST/2:
        icoord_list.append([uiva, uiva, uivb, uivb])
        dcoord_list.append([uah, h, h, ubh])
    if link_color_func is not None:
        v = link_color_func(int(i))
        if not isinstance(v, string_types):
            raise TypeError("link_color_func must return a matplotlib "
                            "color string!")
        color_list.append(v)
    else:
        color_list.append(c)

    return (((uiva + uivb) / 2), uwa + uwb, h, max_dist)






def dendrogram_ktss(Z, p=30, truncate_mode=None, color_threshold=None,
               get_leaves=True, orientation='top', labels=None,
               count_sort=False, distance_sort=False, show_leaf_counts=True,
               no_plot=False, no_labels=False, leaf_font_size=None,
               leaf_rotation=None, leaf_label_func=None,
               show_contracted=False, link_color_func=None, ax=None,
               above_threshold_color='b'):
    Z = np.asarray(Z, order='c')

    if orientation not in ["top", "left", "bottom", "right"]:
        raise ValueError("orientation must be one of 'top', 'left', "
                         "'bottom', or 'right'")

    # is_valid_linkage(Z, throw=True, name='Z')
    Zs = Z.shape
    n = Zs[0] + 1
    if type(p) in (int, float):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')

    if truncate_mode not in ('lastp', 'mlab', 'mtica', 'level', 'none', None):
        # 'mlab' and 'mtica' are kept working for backwards compat.
        raise ValueError('Invalid truncation mode.')

    if truncate_mode == 'lastp' or truncate_mode == 'mlab':
        if p > n or p == 0:
            p = n

    if truncate_mode == 'mtica':
        # 'mtica' is an alias
        truncate_mode = 'level'

    if truncate_mode == 'level':
        if p <= 0:
            p = np.inf

    if get_leaves:
        lvs = []
    else:
        lvs = None

    icoord_list = []
    dcoord_list = []
    color_list = []
    current_color = [0]
    currently_below_threshold = [False]
    ivl = []  # list of leaves

    if color_threshold is None or (isinstance(color_threshold, string_types) and
                                   color_threshold == 'default'):
        color_threshold = max(Z[:, 2]) * 0.7

    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'ivl': ivl,
         'leaves': lvs, 'color_list': color_list}

    # Empty list will be filled in _dendrogram_calculate_info
    contraction_marks = [] if show_contracted else None

    _dendrogram_calculate_info_ktss(
        Z=Z, p=p,
        truncate_mode=truncate_mode,
        color_threshold=color_threshold,
        get_leaves=get_leaves,
        orientation=orientation,
        labels=labels,
        count_sort=count_sort,
        distance_sort=distance_sort,
        show_leaf_counts=show_leaf_counts,
        i=2*n - 2,
        iv=0.0,
        ivl=ivl,
        n=n,
        icoord_list=icoord_list,
        dcoord_list=dcoord_list,
        lvs=lvs,
        current_color=current_color,
        color_list=color_list,
        currently_below_threshold=currently_below_threshold,
        leaf_label_func=leaf_label_func,
        contraction_marks=contraction_marks,
        link_color_func=link_color_func,
        above_threshold_color=above_threshold_color)

    if not no_plot:
        mh = max(Z[:, 2])
        _plot_dendrogram(icoord_list, dcoord_list, ivl, p, n, mh, orientation,
                         no_labels, color_list,
                         leaf_font_size=leaf_font_size,
                         leaf_rotation=leaf_rotation,
                         contraction_marks=contraction_marks,
                         ax=ax,
                         above_threshold_color=above_threshold_color)

    return R