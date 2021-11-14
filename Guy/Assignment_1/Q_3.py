import numpy as np
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from matplotlib.patches import Rectangle

def parse_args():
    parser = argparse.ArgumentParser('Provide laser scan of the environment')
    parser.add_argument('-f', '--file_path', default='laserscan.csv', help='path to the laser scan csv file')
    args = parser.parse_args()
    return args

def pnt2line(pnt, line_p1, line_p2):
    '''
    calculate the distance between a point (pnt) to a line
    :param pnt: the given point (x,y)
    :param line_p1: first line point (x,y)
    :param line_p2: second line point (x,y)
    :return: the distance (dist) and the nearest point on the line (nearest)
    '''
    line_vec = (line_p2[0]-line_p1[0], line_p2[1]-line_p1[1])
    pnt_vec = (line_p1[0]-pnt[0], line_p1[1]-pnt[1])
    line_len = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
    line_unitvec = (line_vec[0]/line_len, line_vec[1]/line_len)
    pnt_vec_scaled = (pnt_vec[0]*(1.0/line_len), pnt_vec[1]*(1.0/line_len))
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = (line_vec[0]*t, line_vec[1]*t)
    nearest_pnt_vec = (nearest[0]-pnt_vec[0], nearest[1]-pnt_vec[1])
    dist = np.sqrt(nearest_pnt_vec[0]**2 + nearest_pnt_vec[1]**2)
    nearest = nearest + line_p1
    return (dist, nearest)

def polar_to_cartesian(r, theta, epsilon = 0.01):
    '''
    change polar representation to cartesian
    :param r: the radius
    :param theta: the value of the angle in degrees
    :param epsilon: default is 0.01, to eliminate noise
    :return: the cartesian value of the given vector
    '''
    x = r*math.cos(math.radians(theta))
    y = r*math.sin(math.radians(theta))
    if np.abs(x) > epsilon or np.abs(y) > epsilon:
        return x, y
    else:
        return 0, 0

def convert_to_wcs(x_rcs, y_rcs, psi):
    '''
    convert the robot coordinates (rcs) to the real world coordinates (wcs)
    :param x_rcs: the point X values in rcs
    :param y_rcs:
    :param psi:
    :return:
    '''
    x_wcs = x_rcs*math.cos(psi) - y_rcs*math.sin(psi)
    y_wcs = x_rcs*math.sin(psi) + y_rcs*math.cos(psi)
    return x_wcs, y_wcs

def calc_angle_between_vectors(vec_a, vec_b):
    '''
    calculate the angle between two given vector
    :param vec_a: an array of size 2 (x,y)
    :param vec_b: an array of size 2 (x,y)
    :return: return the angle in degrees
    '''
    inner_prod = np.dot(vec_a, vec_b)
    norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    rad = np.arccos(np.clip(inner_prod/norms, -1., 1.))
    deg = np.rad2deg(rad)
    if vec_a[1] > vec_b[1]:
        deg = 360-deg
    return deg

def main(args):
    laser_scan = pd.read_csv(args.file_path, header=None)
    # ------------ plot the graph ------------
    # average the measurements to eliminate noise
    laser_mean_scan = np.average(laser_scan, 0)
    # keep only positive values since other are just noise
    laser_mean_scan = np.abs(laser_mean_scan)
    # gather cartesian representation of the scanner
    laser_cartesian = []
    for i in range(laser_mean_scan.size):
        x, y = polar_to_cartesian(laser_mean_scan[i], i)
        if x != 0 or y != 0:
            laser_cartesian.append([x,y])
    laser_cartesian = np.array(laser_cartesian)
    # plot the cartesian representation and the regular scan vector
    f, (ax1, ax2) = plt.subplots(1, 2)
    # plot regular representation to visualize the relevant angles
    ax1.scatter(np.arange(0,359), laser_mean_scan[laser_mean_scan>0])
    ax1.set_title('Relevant section of degrees')
    ax1.set_xlabel('Degrees')
    ax1.set_ylabel('Measured distance')
    ax1.set_xlim([0,360])
    ax1.set_ylim([0, np.max(np.abs(laser_mean_scan)) + 10])

    # plot the real world as the robot see's it
    x_limit = np.round(np.max(np.abs(laser_cartesian[:,0])))
    y_limit = np.round(np.max(np.abs(laser_cartesian[:,1])))
    ax2.plot(laser_cartesian[:, 0], laser_cartesian[:, 1], linewidth=5, color='k')
    ax2.set_title('Real world - Robot POW')
    ax2.set_xlabel('X_rcs')
    ax2.set_ylabel('Y_rcs')
    ax2.add_patch(Rectangle((-10, -2), 20, 4, color='b', fill=False, linewidth=2))
    ax2.plot([0,0], [0, 15], '-.', color='g')
    ax2.plot([0,30], [0, 0], '-.', color='g')
    ax2.text(25,2, "X_rcs")
    ax2.text(-12,8, "Y_rcs", rotation='vertical')
    ax2.set_xticks(np.arange(-x_limit-11,x_limit + 11 ,10))
    ax2.set_yticks(np.arange(-y_limit-11,y_limit + 11 ,5))
    plt.grid()
    plt.show()
    # ------------ find D, W, theta ------------
    indices_of_obstacle = np.where(laser_mean_scan > 1)[0]
    obstacle_1_d = laser_mean_scan[indices_of_obstacle[0]]
    obstacle_2_d = laser_mean_scan[indices_of_obstacle[-1]]
    angle_between_obstacle_points = indices_of_obstacle[-1] - indices_of_obstacle[0] + 1
    W = (obstacle_1_d**2 + obstacle_2_d**2 - 2*obstacle_1_d*obstacle_2_d*np.cos(angle_between_obstacle_points))**0.5
    # TODO - add the cosine sentence here
    # W = np.linalg.norm(obstacle_p_1 - obstacle_p_2)
    print(f'Obstacle length is {W}')
    # to find theta and D, we are going to find P first
    # P is the smallest values among the readings, to verify, we'll check the Euclidean distance
    dist_to_points = [np.linalg.norm([0,0] - laser_cartesian[i,:]) for i in range(laser_cartesian.shape[0])]
    P = laser_cartesian[np.argmin(dist_to_points), :]
    D = np.min(dist_to_points)
    D_reading = np.min(laser_mean_scan[laser_mean_scan > 1])
    assert (D - D_reading < 0.05), 'we have a problem with the distance calculation'
    # D, P = pnt2line((0, 0), obstacle_p_1, obstacle_p_2)
    print(f'The Distance to the obstacle "D" is {D}')
    # the theta angle is the column number of the point P, to verify, we'll calculate the angle
    theta_deg = calc_angle_between_vectors([1,0], P)
    theta_reading = np.where(laser_mean_scan == D_reading)
    assert (theta_reading - theta_deg < 0.05), 'we have a problem with the theta calculation'
    print(f'Theta is {theta_deg} degrees')
    # ------------ find P coordinates ------------
    print(f'The closest point of the obstacle to the robot is {P}')
    # ------------ find P in the real world ------------
    # shift P to the (X_wcs,Y_wcs) coordinates
    x_0 = 30
    y_0 = 40
    psi_deg = 30
    P_wcs = convert_to_wcs(P[0], P[1], psi_deg)
    print(f'Real world coordinates for P is ({P_wcs[0]+x_0},{P_wcs[1]+y_0})')


if __name__ == '__main__':
    args = parse_args()
    main(args)

