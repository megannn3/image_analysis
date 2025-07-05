import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d



# %% EXTRACTING FRAMES -------------------------------------------
def extract(path):

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Split color channels
        r, g, b = cv2.split(frame_rgb)
        green_channel = frame[:, :, 1]
        frame[:, :, 0] = 0  # Blue channel
        frame[:, :, 2] = 0  # Red channel

        #Counts frames before injection starts
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(green_channel)
        if max_val < 255:
            counter += 1

        
        # Save current frame as image
        frame_filename = os.path.join(output_dir, f'frame_{frame_num:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        frame_num += 1

    cap.release()
    frames_folder = 'frames/'
    frame_files = sorted(f for f in os.listdir(frames_folder)
                        if f.endswith(('.png', '.jpg')))
    print(f"Done! Extracted {frame_num} frames to '{output_dir}/'")
    print(f"Done! Extracted {len(frame_files)} frames to '{output_dir}/'")

    return frames_folder, frame_files, counter

# %% CENTERING ------------------------------------------------------

def find_center(folder,files, count) : 
    frames_folder = folder
    frame_files = files
    counter = count
    img = cv2.imread(os.path.join(frames_folder, frame_files[counter+1]))  # first frame with pure water


    green_channel = img[:, :, 1]

    # Find the location of the maximum green value in frame where injection begins
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(green_channel)

    print(f"Brightest green pixel intensity: {max_val}")
    print(f"Location (x, y): {max_loc}") 
    x,y =max_loc
    marked_img = img.copy()
    cv2.circle(marked_img, (x, y), radius=5, color=(0, 0, 255), thickness=2)
    # Save image with brightest pixel marked
    cv2.imwrite('marked_brightest_green.jpg', marked_img)
    return max_loc







# %% FIND INTENSITY -------------------------------------------------------

#function for finding average intensity of whole cell
def find_intensity(folder,files, max):
    frames_folder = folder
    frame_files = files
    avg_green = []
    green = []
    max_loc = max
    x,y = max_loc
    print("Number of valid frames:", len(frame_files))

    for filename in frame_files:
        img_path = os.path.join(frames_folder, filename)
        img = cv2.imread(img_path)  
        if img is None:
            print("Warning: Couldn't read", img_path)
            continue
        center_x, center_y = max_loc
        radius = 380

# Create a mask the same size as the image, with a white circle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # single channel mask
        marked = img.copy()
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # fill circle with white
        cv2.circle(marked, (center_x, center_y), radius, color=(255, 255, 255), thickness=2)  
        # draw white border and save
        
        

# only keep pixels with white mask
        green_channel = img[:, :, 1]
        masked_values = green_channel[mask == 255]
    #find and add the average intensities to a list
        avg_g = np.mean(masked_values)
        avg_green.append(avg_g) 
    #recording values of single pixel (currently injection point)
        pixel = img[y,x]  
        green.append(pixel[1])
    
    cv2.imwrite('image_with_circle.jpg', marked)

    avg_green = gaussian_filter1d(avg_green, sigma=2)
    green =gaussian_filter1d(green, sigma=2)
    return avg_green, green




# %% PLOTTING -------------------------------------------------------------

def plotting(avg,g, path):
    avg_green = avg
    green = g
    video_path = path
    #converting y axis from frames to seconds
    fps = 30  
    times_in_seconds = [i / fps for i in range(len(avg_green))]
    #normalizing x axis
    avg_intensity = [(x) / (255) for x in avg_green]
    intensity = [(x) / (255) for x in green]

    print(len(avg_green))

    #Plotting 


    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1)
    #plt.xlim(right=100)

    plt.plot(times_in_seconds, avg_intensity, color='green')


    plt.title('Average Green Intensity Over Time')
    plt.xlabel('Seconds')
    plt.ylabel('Average Intensity')
    plt.legend()
    plt.savefig(video_path+'green_channel_only.png')



    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1)
    #plt.xlim(right=100)

    plt.plot(times_in_seconds, intensity, color='green')


    plt.title('Green Intensity Over Time')
    plt.xlabel('Seconds')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig(video_path+'green_channel.png')

    plt.figure(figsize=(10, 5))
    plt.plot(times_in_seconds[::2], avg_intensity[::2], color='green', label='Sampled')
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.title("Green Intensity (Sampled Every Other Point)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(video_path+'green_channel_every_other.png')

    for t, val in zip(times_in_seconds, avg_intensity):
        if 100 <= t <= 110:
            print(f"Time: {t:.2f} s, Intensity: {val:.3f}")


    print("Frames processed:", len(avg_intensity), len(intensity))
    print("Time entries:", len(times_in_seconds))
    print("FPS used:", fps)
    print("Any NaN in green values:", np.isnan(avg_intensity).any())
    print("Any absurd values:", np.max(avg_intensity), np.min(avg_intensity))




def derivative(video_path,intensity):
    fps = 30  
    times_in_seconds = [i / fps for i in range(len(intensity))]
    

    if np.array_equal(intensity, avg):
        title = 'Average'
    elif np.array_equal(intensity,g):
        title = 'Point'
    d_intensity = np.gradient(intensity,times_in_seconds)

    print(len(times_in_seconds) )
    print(len(np.unique(times_in_seconds)))

    plt.figure(figsize=(10, 5))
    plt.plot(times_in_seconds, d_intensity , label='Rate of Change of Intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('d(Intensity)/dt')
    plt.title('Rate of Change of '+ title+  ' Intensity')

    plt.legend()
    plt.savefig(video_path+title+'derivative.png')







path = '5mlmin_trimmed.MP4'
files, folder, counter = extract(path)
max = find_center(files,folder, counter)
avg,g = find_intensity(files, folder, max)
plotting(avg,g,path)
derivative(path,avg)





 

