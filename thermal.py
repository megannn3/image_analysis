import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d


def extract_thermal(path):

    #clear previous frames
    folder = 'frames/'

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # Input video path
    video_path = path

    # Create output directory for frames
    output_dir = 'frames'
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Extracting frames
    frame_num = 0

    counter =0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        
        # Save current frame as image
        # frame_filename = os.path.join(output_dir, f'frame_{frame_num:04d}.jpg')
        # cv2.imwrite(frame_filename, frame)

        img = frame

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data_max = np.max(gray)
        data_min = np.min(gray)

        # Angle range
        angle_min = -15
        angle_max = 22

        # We want to find the value that corresponds to 0 degrees
        target_angle = 0

        # Linear interpolation
        if frame_num ==0:
            value_at_0 = data_min + ((target_angle - angle_min) / (angle_max - angle_min)) * (data_max - data_min)
            print(f"Data value corresponding to 0°: {value_at_0}")
            data_max = value_at_0
            temp_min = -15  # in °C
            temp_max = 0

            # Normalize image to temperature scale
        temps = temp_min + ((gray - data_min) / (data_max - data_min)) * (temp_max - temp_min)    


        

        plt.figure(figsize=(8, 6))
        im = plt.imshow(temps, cmap='inferno', vmin=temp_min, vmax=temp_max)
        cbar = plt.colorbar(im)
        cbar.set_label('Temperature')
        plt.title("Thermal Camera")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"frames/plot_{frame_num:04d}.png")
        plt.close()  # Close to free memory

        
        frame_num += 1

    cap.release()
    frames_folder = 'frames/'
    frame_files = sorted(f for f in os.listdir(frames_folder)
                        if f.endswith(('.png', '.jpg')))
    print(f"Done! Extracted {frame_num} frames to '{output_dir}/'")
   
    


def video():
    frame_folder = 'frames'
    output_video = 'thermal_video.mp4'
    fps = 30

    # Get image files
    images = sorted([img for img in os.listdir(frame_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, _ = frame.shape

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in images:
        frame = cv2.imread(os.path.join(frame_folder, img))
        out.write(frame)

    out.release()
    print("Video saved as", output_video)

extract_thermal('3mlminthermal.mp4')
video()


# Load image and convert to grayscale or a single channel (e.g. green)
