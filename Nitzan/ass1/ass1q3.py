import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scans = pd.read_csv("laserscan.csv", header=None)
    _, samples = scans.shape
    mean_scans = [np.average(scans[i]) for i in range(samples)]
    angles = np.arange(samples)
    plt.scatter(angles, mean_scans)
    plt.xlabel("Angel(deg)")
    plt.ylabel("Distance(cm)")
    plt.title("Distance per angel graph")
    plt.show()
    
    meaningful_points = [x for x in mean_scans if x > 50]
    W = np.sqrt(meaningful_points[0] ** 2 + meaningful_points[-1] ** 2 - 2 *
                meaningful_points[0]*meaningful_points[-1] *
                np.cos(np.radians(len(meaningful_points))))
    D = np.min(meaningful_points)
    Theta = mean_scans.index(D)
    P = (D*np.cos(np.radians(Theta)), D*np.sin(np.radians(Theta)))
    x0 = 30
    y0 = 40
    psi = 30
    xw = x0+P[0]*np.cos(np.radians(psi)) - P[1]*np.sin(np.radians(psi))
    yw = y0 + P[0]*np.sin(np.radians(psi)) + P[1] * np.cos(np.radians(psi))
    print("W: ", W, "D: ", D, "Theta: ", Theta,
          "P: ", P, "xw: ", xw, "yw: ", yw)