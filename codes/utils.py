import dv_processing as dv
import cv2 as cv
import numpy as np
import glob

def set_visualizer(filename):
    reader = dv.io.MonoCameraRecording(filename)
    visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
    visualizer.setBackgroundColor(dv.visualization.colors.white())
    visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
    visualizer.setNegativeColor(dv.visualization.colors.darkGrey())
    return reader, visualizer


def set_kalman_filter():
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) * 0.03
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kalman.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
    kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
    return kalman


def euclidean_distance(point1, point2):
    point1 = np.asarray(point1, dtype=np.float64)
    point2 = np.asarray(point2, dtype=np.float64)
    return np.sqrt(np.sum(np.square(point2 - point1)))


def adaptive_event_slicing(events, thresh, width, height, sigma_running=0):
    # Initialize variables
    mu_running = 0.0
    slice_data = []
    frame = np.zeros((height, width), dtype=np.float64)
    downsample_factor = 2
    
    # Process each event in the stream
    for event in events:
        x, y, polarity = event.x(), event.y(), event.polarity()  # Unpack the tuple directly
        slice_data.append((x, y, polarity))
        
        if x % downsample_factor == 0 or y % downsample_factor == 0:
            p_abs = abs(polarity)
            frame[x, y] = p_abs
            mu_current = np.mean(frame)
            sigma_current = np.sqrt((1 / (height * width)) * (mu_current - p_abs)**2)
            
            num_events = len(slice_data)
            sigma_running = ((1 - 1/num_events) * sigma_running + (1/num_events) * sigma_current)
            if sigma_running > thresh:
                return slice_data, sigma_running
    
    return slice_data, sigma_running


def get_pupil_each_frame(frame, corrected_coordinatess, set_kalman_flag, avg_coordinate, kalman, cnt_count, cnt_area, if_adapt):
    og_frame = frame.copy()
    og_frame2 = frame.copy()
    kernel = np.ones((3, 3), np.uint8) 
    frame = cv.dilate(frame, kernel, iterations=1)
    edged = cv.Canny(frame, 10, 20)
    edged = cv.dilate(edged, kernel, iterations=1)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    for cnt in contours:
        if len(cnt) > cnt_count and cv.contourArea(cnt) > cnt_area:
            cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            circle_frame = frame.copy() 
            circles = cv.HoughCircles(circle_frame, cv.HOUGH_GRADIENT,3,100, param1=20,param2=30,minRadius=0,maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    if (i[2] > 80) or (i[2] < 35):
                        continue 
                    if i[0] > i[1]:
                        if i[0]/i[1] > 6:
                            continue
                    elif i[1] > i[0]:
                        if i[1]/i[0] > 6:
                            continue
                    if set_kalman_flag == 0:
                        avg_coordinate = i[:2]
                        kalman.statePre = np.array([i[0], i[1], 0, 0], np.float32)
                        kalman.statePost = np.array([i[0], i[1], 0, 0], np.float32)
                        set_kalman_flag=1
                    else:
                        distance = euclidean_distance(avg_coordinate, i[:2])
                        if distance < 20:
                            prediction = kalman.predict()
                            kalman.correct(np.array([[np.float32(i[0])], [np.float32(i[1])]]))

                            corrected_coordinates = np.array([kalman.statePost[0], kalman.statePost[1]])
                            avg_coordinate = (avg_coordinate + i[:2]) / 2
                            corrected_coordinatess.append(corrected_coordinates)
                            if if_adapt:
                                og_frame = cv.cvtColor(og_frame, cv.COLOR_GRAY2BGR)
                                cv.circle(og_frame,(int(corrected_coordinates[0]), int(corrected_coordinates[1])),i[2],(255,0,0),2)
                                cv.circle(og_frame,(i[0],i[1]),2,(255,0,0),3)
                            else:
                                cv.circle(og_frame,(int(corrected_coordinates[0]), int(corrected_coordinates[1])),i[2],(255,0,0),2)
                                cv.circle(og_frame,(i[0],i[1]),2,(255,0,0),3)
                            cv.imshow("Preview", og_frame)
                            key = cv.waitKey(1)
                    og_frame = og_frame2.copy() 
    if len(corrected_coordinatess) > 0:
        return corrected_coordinatess, set_kalman_flag, avg_coordinate
    

def global_event_stream(events):
    event_iter = []
    for event in events:
        try:
            event_iter.append((event.x(), event.y(), event.polarity()))
        except AttributeError as e:
            print("Error", e)
    return event_iter



def get_coordinates(reader, event_volume, visualizer, kalman, cnt_count, cnt_area):
    set_kalman_flag = 0
    corrected_coordinatess = []
    avg_coordinate = np.array([0,0])
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is not None:
            event_count = len(events)
            if event_volume[1] > event_count > event_volume[0]:
                frame = visualizer.generateImage(events)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                try:
                    corrected_coordinatess, set_kalman_flag, avg_coordinate = get_pupil_each_frame(frame, corrected_coordinatess, set_kalman_flag, avg_coordinate, kalman, 150, 500, False)
                except TypeError as e:
                    continue
    cv.destroyAllWindows()
    return corrected_coordinatess

def get_coordinates_adaptive(reader, event_volume, visualizer, kalman):
    set_kalman_flag = 0
    corrected_coordinatess = []
    avg_coordinate = np.array([0,0])
    old_events = None
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is not None:
            if old_events is None:
                old_events = global_event_stream(events)
            sigma_running = 0
            prev_sigma_running = 0
            thresh = 0.018
            width=reader.getEventResolution()[1]
            height=reader.getEventResolution()[0]
            events, prev_sigma_running = adaptive_event_slicing(events, thresh=thresh, width=width, height=height, sigma_running=0)
            
            while sigma_running < thresh:
                new_events = reader.getNextEventBatch()
                new_events, prev_sigma_running = adaptive_event_slicing(new_events, thresh=0.5, width=reader.getEventResolution()[1], height=reader.getEventResolution()[0], sigma_running=sigma_running)
                old_events.extend([*new_events])
                sigma_running = (sigma_running+prev_sigma_running)
            frame = events_to_frame(old_events, height, width)
            old_events = None
            try:
                corrected_coordinatess, set_kalman_flag, avg_coordinate = get_pupil_each_frame(frame, corrected_coordinatess, set_kalman_flag, avg_coordinate, kalman, 80, 150, True)
            except TypeError as e:
                print("error")    
        
    cv.destroyAllWindows()
    return corrected_coordinatess



def events_to_frame(events, width, height):
    # Initialize an empty frame
    frame = np.ones((height, width), dtype=np.uint8)*255
    
    # Populate the frame based on the event data
    for event in events:
        x, y, polarity = event
        frame[y, x] = 0
        
    return frame


if __name__ == "__main__":
    files = glob.glob('dataset/*.aedat4')
    for f in files:
        print(f)
        reader, visualizer = set_visualizer(f)
        kalman = set_kalman_filter()
        min_events = 4000 
        max_events = 17000
        event_volume = (min_events, max_events)
        corrected_coordinatess = get_coordinates(reader, event_volume)
