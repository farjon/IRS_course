import numpy as np
import copy
import open3d as o3d


if __name__ == "__main__":
    pcd000 = o3d.io.read_point_cloud("bun000.ply")
    pcd045 = o3d.io.read_point_cloud("bun045.ply")
    pcd090 = o3d.io.read_point_cloud("bun090.ply")
    pcd180 = o3d.io.read_point_cloud("bun180.ply")
    pcd270 = o3d.io.read_point_cloud("bun270.ply")
    pcd315 = o3d.io.read_point_cloud("bun315.ply")

    # Translate to Vectors
    xyz000 = np.asarray(pcd000.points)

    xyz090 = np.asarray(pcd090.points)

    xyz180 = np.asarray(pcd180.points)

    xyz270 = np.asarray(pcd270.points)

    lenxyz000 = np.max(xyz000[:, 0]) - np.min(xyz000[:, 0])
    lenxyz180 = np.max(xyz180[:, 0]) - np.min(xyz180[:, 0])
    xmax = max(lenxyz000, lenxyz180)

    lenxyz090 = np.max(xyz090[:, 1]) - np.min(xyz090[:, 1])
    lenxyz270 = np.max(xyz270[:, 1]) - np.min(xyz270[:, 1])
    ymax = max(lenxyz090, lenxyz270)

    zmax = np.max([np.max(xyz000[:, 2]) - np.min(xyz000[:, 2]), np.max(xyz090[:, 2]) - np.min(xyz090[:, 2]),
                   np.max(xyz180[:, 2]) - np.min(xyz180[:, 2]), np.max(xyz270[:, 2]) - np.min(xyz270[:, 2])])
    print(zmax * xmax * zmax)

    pcd = copy.deepcopy(pcd000)
    RGB = np.zeros_like(xyz000)
    red = [255, 0, 0]
    blue = [0, 0, 255]
    for i in range(len(xyz000)):
        if xyz000[i][2] > 0.015:
            RGB[i] = blue
        else:
            RGB[i] = red
    pcd.colors = o3d.utility.Vector3dVector(RGB)
    o3d.visualization.draw_geometries([pcd])
