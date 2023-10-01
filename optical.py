import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import imutils
from config import INPUT_VIDEO_PATH, NEW_DESIRED_FRAME


def opticalFlow():
    # Initialize video capture
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # Create a VideoWriter object to save the output
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
    size = (w, h)
    out = cv2.VideoWriter('output.avi', fourcc, fps, size, True)

    # Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors for drawing optical flow
    color = np.random.randint(0, 255, (100, 3))

    # Read the first frame
    ret, old_frame = cap.read()
    old_frame = imutils.resize(old_frame, width=w, height=h)  # Resize the frame
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Initialize variables for tracking speed and direction
    average_speed_per_second = []
    average_speed_per_minute = []
    frame_count = 0

    # Initialize feature points for optical flow (p0)
    # You can use a feature detection method (e.g., Shi-Tomasi or Harris) to initialize p0
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Check if any feature points were found
    if p0 is None or len(p0) == 0:
        print("No feature points found in the first frame.")
        exit()

    # Ensure data type and shape of p0 are as expected
    p0 = p0.astype(np.float32)
    p0 = p0.reshape(-1, 1, 2)

    # Initialize mask for drawing optical flow vectors and circles
    mask = np.zeros_like(old_frame)

    # Histogram for flow angles
    angle_hist = []

    # CSV file for data logging
    csv_filename = 'crowd_flow_data.csv'

    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Time (s)', 'Average Speed (px/s)', 'Dominant Direction'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            frame = imutils.resize(frame, width=w, height=h)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow using Lucas-Kanade
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Calculate the flow vectors and speeds
            flow_vectors = good_new - good_old
            speeds = np.linalg.norm(flow_vectors, axis=1)
            average_speed = np.mean(speeds)

            # Calculate the flow angles
            angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
            angle_hist.extend(angles)

            # Update frame count
            frame_count += 1
            DATA_RECORD_FRAME = fps / NEW_DESIRED_FRAME
            # Append the speed to per-second and per-minute lists
            average_speed_per_second.append(average_speed)
            if frame_count % int(DATA_RECORD_FRAME) == 0:
                average_speed_per_minute.append(np.mean(average_speed_per_second))
                # Calculate dominant direction based on the histogram
                hist, bin_edges = np.histogram(angle_hist, bins=20, range=(-np.pi, np.pi))
                dominant_angle = bin_edges[np.argmax(hist)] 
                # Modify based on the dominant angle
                if -np.pi / 4 < dominant_angle <= np.pi / 4:
                    dominant_direction = "East"
                elif np.pi / 4 < dominant_angle <= 3 * np.pi / 4:
                    dominant_direction = "North"
                elif -3 * np.pi / 4 < dominant_angle <= -np.pi / 4:
                    dominant_direction = "South"
                else:
                    dominant_direction = "West"
                angle_hist = []  # Reset histogram

                # Write data to CSV file
                elapsed_time = frame_count / (fps/DATA_RECORD_FRAME)
                csv_writer.writerow([elapsed_time, np.mean(average_speed_per_minute), dominant_direction])
                average_speed_per_second = []

            # Draw optical flow vectors on the frame
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)  # Convert to integers
                c, d = old.ravel().astype(int)  # Convert to integers
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

            # Combine the frame and the mask
            img = cv2.add(frame, mask)

            # Display the frame
            cv2.imshow('Crowd Flow', img)

            # Write the frame to the output video
            out.write(img)

            # Update the previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Plot the histogram
        plt.hist(angle_hist, bins=20, range=(-np.pi, np.pi))
        plt.title("Flow Angle Histogram")
        plt.xlabel("Angle (radians)")
        plt.ylabel("Frequency")
        plt.show()
