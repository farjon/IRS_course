import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("bun000.ply")

#Translate to Vectors
xyz = np.asarray(pcd.points)
y = xyz[:,1]
z = xyz[:,2]

miny = np.min(y)
maxy = np.max(y)
minz = np.min(z)
maxz = np.max(z)

pcd = o3d.io.read_point_cloud("bun090.ply")

#Translate to Vectors
xyz = np.asarray(pcd.points)
x = xyz[:,0]
z = xyz[:,2]

minx = np.min(x)
maxx = np.max(x)
minz = min(np.min(z),minz)
maxz = max(np.max(z),maxz)

pcd = o3d.io.read_point_cloud("bun180.ply")

#Translate to Vectors
xyz = np.asarray(pcd.points)
y = xyz[:,1]
z = xyz[:,2]

miny = min(np.min(y),miny)
maxy = max(np.max(y),maxy)
minz = min(np.min(z),minz)
maxz = max(np.max(z),maxz)

pcd = o3d.io.read_point_cloud("bun270.ply")

#Translate to Vectors
xyz = np.asarray(pcd.points)
x = xyz[:,0]
z = xyz[:,2]

minx = min(np.min(x),minx)
maxx = max(np.max(x),maxx)
minz = min(np.min(z),minz)
maxz = max(np.max(z),maxz)

area = (maxx-minx)*(maxy-miny)*(maxz-minz)
print(area)

