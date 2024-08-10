import dv_processing as dv
import cv2 as cv
import numpy as np
import glob
from utils import set_kalman_filter, set_visualizer, get_coordinates_adaptive

if __name__ == "__main__":
    files = glob.glob('../dataset/*.aedat4')
    for f in files:
        print(f)
        reader, visualizer = set_visualizer(f)
        kalman = set_kalman_filter()
        corrected_coordinatess = get_coordinates_adaptive(reader, kalman)
        # Disable visaulaizer to have lower latency