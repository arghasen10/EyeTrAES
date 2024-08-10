import dv_processing as dv
import cv2 as cv
import numpy as np
import glob
from utils import set_kalman_filter, set_visualizer, get_coordinates

if __name__ == "__main__":
    files = glob.glob('../dataset/*.aedat4')
    for f in files:
        print(f)
        reader, visualizer = set_visualizer(f)
        kalman = set_kalman_filter()
        min_events = 4000 
        max_events = 17000
        event_volume = (min_events, max_events)
        corrected_coordinatess = get_coordinates(reader, event_volume, visualizer, kalman)
        # Disable visaulaizer to have lower latency