import os
import numpy as np
from glob import glob
import open3d as o3d
import copy
import math as m
from scipy.spatial.transform import Rotation as R

# subsection 1
path_to_bunny = os.path.join(os.getcwd(), 'bunny', 'Q3')
bun_000_reading = o3d.io.read_point_cloud(
    os.path.join(path_to_bunny, 'bun000.ply'))
o3d.visualization.draw_geometries([bun_000_reading])

# subsection 2
file1 = open(os.path.join(path_to_bunny, 'bun.txt'), 'r')
Lines = file1.readlines()
pcds = o3d.geometry.PointCloud()

for i, line in enumerate(Lines):
    bunny = line.split(' ')[1]
    tx, ty, tz = line.split(' ')[2:5]
    bunny_ply = o3d.io.read_point_cloud(os.path.join(path_to_bunny, bunny))
    if i == 0:
        pcds += bunny_ply
    rot_deg = int(bunny.split('.')[0][-3:])
    R = bunny_ply.get_rotation_matrix_from_xyz((0, np.deg2rad(rot_deg), 0))
    bunny_ply.rotate(R, center=(0, 0, 0))
    if rot_deg == 45 or rot_deg == 315:
        bunny_ply.translate((tx, ty, tz))
    pcds += bunny_ply
pcds.estimate_normals()
o3d.visualization.draw_geometries([pcds])

aabb = pcds.get_axis_aligned_bounding_box()
a = aabb.get_max_bound()
b = aabb.get_min_bound()
object_size = np.sum((a-b)**2)


# section 3
sec_3_pcd = copy.deepcopy(pcds)
xyz = np.asarray(sec_3_pcd.points)
# RGB = np.asarray(sec_3_pcd.colors)
x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]

# XYZ ranges
print("X:"+str(np.min(x))+" - "+str(np.max(x)))
print("Y:"+str(np.min(y))+" - "+str(np.max(y)))
print("Z:"+str(np.min(z))+" - "+str(np.max(z)))

ears_indices_x = np.asarray(np.where(xyz[:, 0] > -0.08))
ears_indices_y = np.asarray(np.where(xyz[:, 1] > 0.142))
ears_indices_z = np.asarray(np.where(xyz[:, 2] < 0.01))
ears_indices_intersect = np.intersect1d(ears_indices_x, ears_indices_y)
ears_indices_intersect = np.intersect1d(ears_indices_intersect, ears_indices_z)
body_indices = np.asarray(
    np.delete(np.arange(xyz.shape[0]), ears_indices_intersect))
RGB = np.zeros_like(xyz)
RGB[ears_indices_intersect, 0] = 255
RGB[body_indices, 2] = 255
sec_3_pcd.colors = o3d.utility.Vector3dVector(RGB)
o3d.visualization.draw_geometries([sec_3_pcd])
