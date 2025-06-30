import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Input video path
video_path = '8pt6C1mlmin100 - Copy.MP4'

# Create output directory for frames
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Frame counter
frame_num = 0
avg_green = []
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

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(green_channel)
    if max_val < 255:
        counter += 1

    

    # print(f"Average Green intensity: {avg_g:.2f}")

    
    # Save current frame as image
    frame_filename = os.path.join(output_dir, f'frame_{frame_num:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    
    frame_num += 1

cap.release()
print(f"Done! Extracted {frame_num} frames to '{output_dir}/'")


frames_folder = 'frames/'
frame_files = sorted(os.listdir(frames_folder))  # Sort to keep order
img = cv2.imread(os.path.join(frames_folder, frame_files[counter+1]))  # OpenCV loads in BGR order

# Extract the green channel (channel index 1 in BGR)
green_channel = img[:, :, 1]

# Find the location of the maximum green value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(green_channel)

print(f"Brightest green pixel intensity: {max_val}")
print(f"Location (x, y): {max_loc}")  # OpenCV returns (x, y) format
x,y =max_loc
marked_img = img.copy()
cv2.circle(marked_img, (x, y), radius=5, color=(0, 0, 255), thickness=2)
# Save or show the result
cv2.imwrite('marked_brightest_green.jpg', marked_img)


def find_avg():
    for filename in frame_files:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            path = os.path.join(frames_folder, filename)
            img = cv2.imread(path)  
        center_x, center_y = max_loc
        radius = 400

# Create a mask the same size as the image, with a white circle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # single channel mask
        marked = img.copy()
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # fill circle with white
        cv2.circle(marked, (center_x, center_y), radius, color=(255, 255, 255), thickness=2)  # green border

        cv2.imwrite('image_with_circle.jpg', marked)

# Extract green channel
        

# Apply mask: only pixels inside circle are kept
        green_channel = img[:, :, 1]
        masked_values = green_channel[mask == 255]
    # Compute average intensity for each channel
        avg_g = np.mean(masked_values)
        avg_green.append(avg_g) 


green =[]
def check_regions(folder):

    for filename in frame_files:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            path = os.path.join(frames_folder, filename)
            img = cv2.imread(path)
            pixel = img[y,x]  
            
            green.append(pixel[1])

find_avg()
check_regions('frames/')

print(avg_green)
fps = 30  

times_in_seconds = [i / fps for i in range(len(avg_green))]
avg_intensity = [(x) / (255) for x in avg_green]
intensity = [(x) / (255) for x in green]


plt.figure(figsize=(10, 5))

plt.plot(times_in_seconds, avg_intensity, color='green', label='Average Green Intensity')


plt.title('Average Green Intensity Over Frames')
plt.xlabel('Seconds')
plt.ylabel('Average Intensity')
plt.legend()
plt.savefig('green_channel_only.png')



plt.figure(figsize=(10, 5))

plt.plot(times_in_seconds, intensity, color='green', label='Average Green Intensity')


plt.title('Green Intensity Over Frames')
plt.xlabel('Seconds')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('green_channel.png')


# Load image
# for filename in frames/:
#    path = os.path.join(frames_folder, filename)
#    img = cv2.imread(path)
#    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Split color channels
#     r, g, b = cv2.split(img_rgb)
#     img[:, :, 0] = 0  # Blue channel
#     img[:, :, 2] = 0  # Red channel

#         # Compute average intensity for each channel

#     avg_g = np.mean(g)



#     print(f"Average Green intensity: {avg_g:.2f}")




# Show histograms of color intensity
# plt.figure(figsize=(10, 4))



# plt.subplot(1, 3, 2)
# plt.hist(g.ravel(), bins=256, color='green', alpha=0.7)
# plt.title('Green Intensity')



# plt.tight_layout()
 

