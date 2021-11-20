import open3d as o3d
import numpy as np
import os
path_to_bunny = os.path.join(os.getcwd(), 'bunny', 'Q3')
pcd = o3d.io.read_point_cloud(os.path.join(path_to_bunny, 'bun000.ply'))
# pcd = o3d.io.read_point_cloud("cloud.ply")
o3d.visualization.draw_geometries([pcd])

#Translate to Vectors
xyz = np.asarray(pcd.points)
RGB = np.asarray(pcd.colors)
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]

#XYZ ranges
print("X:"+np.str(np.min(x))+" - "+np.str(np.max(x)))
print("Y:"+np.str(np.min(y))+" - "+np.str(np.max(y)))
print("Z:"+np.str(np.min(z))+" - "+np.str(np.max(z)))



ind2 = np.asarray(np.where(xyz[:, 1] > 0.5)) #Remove ground in front
xyz2 = np.delete(xyz, ind2, 0)
RGB2 = np.delete(RGB, ind2, 0)

filter_pcd = pcd
filter_pcd.points = o3d.utility.Vector3dVector(xyz2)
filter_pcd.colors = o3d.utility.Vector3dVector(RGB2)
o3d.visualization.draw_geometries([filter_pcd])




ind3 = np.asarray(np.where(RGB[:, 0] < 0.6)) #Remove everything not highly red
ind3 = np.asarray(np.where(RGB[:, 2] < 0.6)) #Remove everything not highly blue

xyz3 = np.delete(xyz, ind3, 0)
RGB3 = np.delete(RGB, ind3, 0)

filter_pcd = pcd
filter_pcd.points = o3d.utility.Vector3dVector(xyz3)
filter_pcd.colors = o3d.utility.Vector3dVector(RGB3)
o3d.visualization.draw_geometries([filter_pcd])
