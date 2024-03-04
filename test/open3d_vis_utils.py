"""
Open3d visualization tool box
Written by Jihan YANG
Modified by Uzuki Ishikawajima (2024) 
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw

box_colormap = [
    [1.0, 1.0, 1.0],
    [0, 1.0, 0],
    [0, 1.0, 1.0],
    [1.0, 1.0, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(
        color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def get_transform_matrix():
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 1],])


def draw_scenes(vis, ref_boxes=None, ref_labels=None, draw_origin=True, tracks=None):
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.detach().cpu().numpy()
    if isinstance(ref_labels, torch.Tensor):
        ref_labels = ref_labels.detach().cpu().numpy()
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.detach().cpu().numpy()

    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.zeros((3))

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1.0, 0),
                       ref_labels)

    if tracks is not None:
        vis = draw_tracks(vis, tracks, (0, 1.0, 0),
                          ref_labels)


def draw_tracks(vis, tracks, color=(0, 1.0, 0), ref_labels=None):
    for i in range(tracks.__len__()):
        if ref_labels is None:
            uni_color = color
        else:
            uni_color = box_colormap[ref_labels[i]]
        
        nodes = [node.cpu().numpy() for node in tracks[i]]
        if nodes.__len__() > 1:
            lines = [(i, i + 1) for i in range(nodes.__len__() - 1)]
            colors = [uni_color for _ in range(nodes.__len__() - 1)]
            
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(nodes)
            line_set.lines = open3d.utility.Vector2iVector(lines)
            line_set.colors = open3d.utility.Vector3dVector(colors)

            vis.add_geometry(line_set)

    return vis


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """

    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    box3d.color = np.ones((3))

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1.0, 0), ref_labels=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

    return vis
