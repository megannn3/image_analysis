import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import re



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

    title1 ='Average Green Intensity Over Time'
    plt.title(title1)
    plt.xlabel('Seconds')
    plt.ylabel('Average Intensity')
    plt.legend()
    plt.savefig(video_path+'green_channel_only.png')



    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1)
    #plt.xlim(right=100)

    plt.plot(times_in_seconds, intensity, color='green')

    title2 = 'Green Intensity Over Time'
    plt.title(title2)
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

    return times_in_seconds, title1, avg_intensity,title2, intensity




def derivative(video_path,x,y):
    fps = 30  
    #times_in_seconds = [i / fps for i in range(len(intensity))]
    

    # if np.array_equal(intensity, avg):
    #     title = 'Average Intensity'
    # elif np.array_equal(intensity,g):
    #     title = 'Point Intensity'
    title = 'Area'
    y =gaussian_filter1d(y, sigma=2)
    d_intensity = np.gradient(y,x)

    print(len(x) )
    print(len(np.unique(x)))
    titles='Rate of Change of '+ title
    plt.figure(figsize=(10, 5))
    plt.plot(x, d_intensity )
    plt.xlabel('Time (s)')
    plt.ylabel('d('+title+')/dt')
    plt.title('Rate of Change of '+ title)

    
    plt.savefig(video_path+title+'derivative.png')

    return x, titles, d_intensity






def contours(folder, files, path, skip):
    import cv2
    import numpy as np
    import os

    video_name = os.path.splitext(os.path.basename(path))[0]
    input_folder = folder
    image_files = files[:1470:skip]
    first_img = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width = first_img.shape[:2]
    choose_img_path = os.path.join(folder, image_files[int(len(image_files)/2)])
    choose_img = cv2.imread(choose_img_path)

    # Black background
    upscale = 2
    high_h, high_w = height * upscale, width * upscale
    high_background = np.zeros((high_h, high_w, 3), dtype=np.uint8)

    threshold_value = 200
    alpha_values = np.linspace(0.05, 0.9, len(image_files))

    roi_mask = np.zeros(choose_img.shape[:2], dtype=np.uint8)
    roi_center = []
    radius = 379

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_center, radius
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_center = [x, y]
        
    

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", mouse_callback)
    cv2.imshow("Draw ROI", choose_img)
    while not roi_center:
        cv2.waitKey(1)

    print('point selected')
    cv2.destroyAllWindows()

    if not roi_center :
        print("ROI selection cancelled.")
        return

    # Create circular mask
    mask1 = np.zeros(choose_img.shape[:2], dtype=np.uint8)
    cv2.circle(mask1, tuple(roi_center), radius, 1, -1)

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        mask1 = mask1.astype(np.uint8)
        green = img[:, :, 1]
        print("green shape:", green.shape)
        print("mask shape:", mask1.shape)
        masked_green = cv2.bitwise_and(green, green, mask=mask1.astype(np.uint8))

        # Threshold the masked region
        _, thresh = cv2.threshold(masked_green, threshold_value, 255, cv2.THRESH_BINARY)

        # Now find contours only within ROI
        contours_found, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale contours for smoother drawing
        scaled_contours = [cnt * upscale for cnt in contours_found]
        mask = np.zeros((high_h, high_w), dtype=np.uint8)
        cv2.drawContours(mask, scaled_contours, -1, 255, thickness=3)  # White contours

        alpha = alpha_values[i]
        for c in range(3):
            high_background[:, :, c] = np.where(
                mask == 255,
                high_background[:, :, c] * (1 - alpha) + 255 * alpha,
                high_background[:, :, c]
            )

    # Downscale to original resolution
    final_image = cv2.resize(high_background, (width, height), interpolation=cv2.INTER_AREA)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    cv2.imwrite(video_name + '_white_on_black_scaled.png', final_image)
    print("Saved", video_name + '_white_on_black_scaled.png')


def black_contours(folder, files, start):

    input_folder = folder
    for filename in os.listdir('tracked_frames'):
        file_path = os.path.join('tracked_frames', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    fps = 30
    image_files = files
    image_paths = [os.path.join(input_folder, f) for f in image_files]

    output_folder = 'tracked_frames'
    os.makedirs(output_folder, exist_ok=True)

    positions = []

    # ---------- POINT SELECTION ----------
    first_frame = cv2.imread(image_paths[start])
    selected_point = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point.append((x, y))
            print(f"Selected: ({x}, {y})")
            
            

    print("Click on the point you want to track...")
    cv2.imshow("Select Point", first_frame)
    cv2.setMouseCallback("Select Point", click_event)
    while not selected_point:
        cv2.waitKey(1)

    print('point selected')
    cv2.destroyAllWindows()

    if not selected_point:
        raise ValueError("No point was selected.")

    prev_point = selected_point[0]
    positions.append(prev_point)

    print(f"Total frames: {len(image_paths)}")
    print(f"Loop will process these frame indices: {list(range(start, len(image_paths), 20))}")
    #length = len(image_paths)
    length = 1470

    # ---------- TRACKING LOOP ----------
    for i in range(start, length, 10):  # start=210, step=20, go to end
        print('doing something')
        path = image_paths[i]
        img = cv2.imread(path)
        if img is None:
            positions.append(None)
            continue

        green = img[:, :, 1]
        threshold_value = 100
        _, thresh = cv2.threshold(green, threshold_value, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            positions.append(None)
            continue
        if i==210:
            cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

        # Find closest contour
        closest_contour = None
        min_distance = float('inf')

        for cnt in contours:
            for pt in cnt:
                pt = pt[0]
                dist = np.linalg.norm(np.array(pt) - np.array(prev_point))
                if dist < min_distance:
                    min_distance = dist
                    closest_contour = cnt

        # Get point on closest contour
        all_points = closest_contour.reshape(-1, 2)
        distances = np.linalg.norm(all_points - np.array(prev_point), axis=1)
        min_idx = np.argmin(distances)
        tracked_point = tuple(all_points[min_idx])
        positions.append(tracked_point)
        prev_point = tracked_point

        # ---------- DRAW ----------
        cv2.drawContours(img, [closest_contour], -1, (0, 0, 255), 2)
        cv2.circle(img, tracked_point, 5, (255, 255, 255), -1)

        output_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, img)

    print(f"Saved {len(image_paths)-1} tracked frames to '{output_folder}/'")



    velocities = []
    valid_frames = []
    for i in range(1, len(positions)):
        if positions[i] is None or positions[i - 1] is None:
            velocities.append(0)
            continue

        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2) * (mm_per_pixel)
        velocity = dist * fps
        velocities.append(velocity)
        valid_frames.append(i)

    velocities = [0] + velocities
    valid_frames = range(len(velocities))  # pad first

    velocities = np.array(velocities)
    valid_frames = np.array(valid_frames)

    # Compute IQR
    q1 = np.percentile(velocities, 25)
    q3 = np.percentile(velocities, 75)
    iqr = q3 - q1

    # Define outlier bounds
    lower_bound = q1 - 6.5 * iqr
    upper_bound = q3 + 6.5 * iqr

    # Create mask
    mask = (velocities >= lower_bound) & (velocities <= upper_bound)

    # Apply mask to both arrays
    filtered_velocities = velocities[mask]
    filtered_frames = valid_frames[mask]
    time_seconds = [(f / 30)+(start/30) for f in filtered_frames]
    #velocities = gaussian_filter1d(velocities, sigma=2)
    def exp_func(x, a, b):
        return a * np.exp(b * x)

    # Fit the curve
    params, _ = curve_fit(exp_func, time_seconds, filtered_velocities, p0=(1, 0.1))
    a, b = params

    # Generate fitted curve
    x_fit = np.linspace(min(time_seconds), max(time_seconds), 100)
    y_fit = exp_func(x_fit, a, b)

    # Plot
    title= "Velocity of Tracked Point"
    plt.scatter(time_seconds, filtered_velocities, color='blue', label='Velocity Data', s=20)
    plt.plot(x_fit, y_fit, color='red', linestyle='-', label=f'Exp Fit: {a:.2f}e^({b:.2f}x)')

    plt.title(title)
    plt.xlabel("Time in Seconds")
    plt.ylabel("Velocity (mm/sec)")
    plt.savefig(video_name+"tracked_velocity.png")

    return time_seconds, title, filtered_velocities



def velocity_by_points(path,folder, files, start,end):
    output_folder = 'point_tracked_frames'
    input_folder = folder
    start = start-counter
    #x if taking video to the end
    if end == 'x':
        image_files = files[counter:]
    else:
        image_files = files[counter:int(end)]


    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir('point_tracked_frames'):
        file_path = os.path.join('point_tracked_frames', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    fps = 30
    
    image_paths = [os.path.join(input_folder, f) for f in image_files]

    
    os.makedirs(output_folder, exist_ok=True)

    positions = []

    # ---------- POINT SELECTION ----------
    first_frame = cv2.imread(image_paths[start])
    selected_point = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point.append((x, y))
            print(f"Selected: ({x}, {y})")
            
            

    print("Click on the point you want to track...")
    cv2.imshow("Select Point", first_frame)
    cv2.setMouseCallback("Select Point", click_event)
    while not selected_point:
        cv2.waitKey(1)

    print('point selected')
    cv2.destroyAllWindows()

    if not selected_point:
        raise ValueError("No point was selected.")

    reference_point = selected_point[0]
    positions.append(reference_point)

    print(f"Total frames: {len(image_paths)}")
    print(f"Loop will process these frame indices: {list(range(start, len(image_paths), 20))}")


    # ---------- TRACKING LOOP ----------
    for i in range(start, len(image_paths), 10):
        print(f"Processing frame {i}")
        path = image_paths[i]
        img = cv2.imread(path)
        if img is None:
            positions.append(None)
            continue

        green = img[:, :, 1]
        threshold_value = 50
        green_mask = green > threshold_value
        coords = np.column_stack(np.where(green_mask))  # (y, x) format

        if coords.size == 0:
            positions.append(None)
            continue

        # Convert to (x, y)
        coords = coords[:, ::-1]

        # Find closest pixel above threshold to previous point
        dists = np.linalg.norm(coords - np.array(reference_point), axis=1)
        closest_idx = np.argmin(dists)
        px, py = coords[closest_idx]

        # Subpixel refinement using weighted centroid in 5x5 patch
        patch_size = 2  # 5x5 neighborhood
        if py - patch_size < 0 or py + patch_size >= green.shape[0] or \
        px - patch_size < 0 or px + patch_size >= green.shape[1]:
            subpixel_point = (px, py)
        else:
            patch = green[py - patch_size: py + patch_size + 1,
                        px - patch_size: px + patch_size + 1].astype(np.float32)
            total = patch.sum()
            if total > 0:
                x_coords, y_coords = np.meshgrid(np.arange(-patch_size, patch_size + 1),
                                                np.arange(-patch_size, patch_size + 1))
                dx = (patch * x_coords).sum() / total
                dy = (patch * y_coords).sum() / total
                subpixel_point = (px + dx, py + dy)
            else:
                subpixel_point = (px, py)

        positions.append(subpixel_point)
        reference_point = subpixel_point

        # ---------- DRAW ----------
        cv2.circle(img, (int(round(subpixel_point[0])), int(round(subpixel_point[1]))),
                5, (255, 255, 255), -1)

        output_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, img)

    print(f"Saved {len(image_paths)-1} tracked frames to '{output_folder}/'")



    velocities = []
    valid_frames = []
    for i in range(1, len(positions)):
        if positions[i] is None or positions[i - 1] is None:
            velocities.append(0)
            continue

        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2) * (mm_per_pixel)
        velocity = dist * fps
        velocities.append(velocity)
        valid_frames.append(i)

    velocities = [0] + velocities
    valid_frames = range(len(velocities))  # pad first

    velocities = np.array(velocities)
    valid_frames = np.array(valid_frames)

    # Compute IQR
    q1 = np.percentile(velocities, 25)
    q3 = np.percentile(velocities, 75)
    iqr = q3 - q1

    # Define outlier bounds
    lower_bound = q1 - 6.5 * iqr
    upper_bound = q3 + 6.5 * iqr

    # Create mask
    mask = (velocities >= lower_bound) & (velocities <= upper_bound)

    # Apply mask to both arrays
    filtered_velocities = velocities[mask]
    filtered_frames = valid_frames[mask]
    time_second = [(f / 30)+(start/30) for f in filtered_frames]
    time_seconds = [t - time_second[0] for t in time_second] 
    #velocities = gaussian_filter1d(velocities, sigma=2)
    x = time_seconds # time_seconds
    y = filtered_velocities  # filtered_velocities

    #Define model (example: exponential)
    smoothed_velocity = gaussian_filter1d(filtered_velocities, sigma=6)



    # Compute rolling standard deviation for error band
    window_size = 15
    std_dev = np.array([
        np.std(filtered_velocities[max(0, i - window_size):min(len(filtered_velocities), i + window_size)])
        for i in range(len(filtered_velocities))
    ])
    # Smooth standard deviation for shaded area (optional)
    smoothed_std = gaussian_filter1d(std_dev, sigma=6) 

    match = re.search(r'\d+', path)
    if match:
        rate = int(match.group())  # ➝ 3
        print("Rate (mL/min):", rate)
    else:
        raise ValueError("No number found in path")


    # Plot smoothed velocity with shaded error
    title = "Velocity of Tracked Point (Smoothed)"
    plt.figure(figsize=(12, 6))
    #plt.scatter(time_seconds, filtered_velocities, color='blue', s=10, alpha=0.5, label='Raw Velocity')
    plt.plot(time_seconds, smoothed_velocity, color='red', label='Smoothed Velocity')
    plt.fill_between(
        time_seconds,
        smoothed_velocity - smoothed_std,
        smoothed_velocity + smoothed_std,
        color='red',
        alpha=0.3,
        label='±1 SD'
    )

    plt.xlabel("Tracking Time (s)")
    plt.ylabel("Velocity (mm/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(video_name + "point_tracked_velocity.png")
    print(len(smoothed_std))


    x=time_seconds
    y=filtered_velocities
    def exp_func(x, a, b):
        return a * np.exp(b * x)

    # Fit exponential curve
    params, cov = curve_fit(exp_func, x, y, p0=(1, 0.1))
    a, b = params
    print("Fit parameters:", a, b)

    # Predict y values
    x_fit = np.linspace(min(x), max(x), 200)
   

    y_fit = exp_func(x_fit, a, b)
    y_fit_at_data_points = exp_func(np.array(time_seconds), a, b)

    # Standard error from the covariance matrix
    perr = np.sqrt(np.diag(cov))  # Standard error of a and b

    # Estimate error in y_fit using simple propagation of uncertainty:
    # dy = sqrt( (df/da * da)^2 + (df/db * db)^2 )
    da, db = perr
    dy = np.sqrt((np.exp(b * x_fit) * da)**2 + (a * x_fit * np.exp(b * x_fit) * db)**2)

    equations.append(f"Fitted equation: y = {a:.3f} * e^({b:.3f} * x)")

    # Plot
    title = "Velocity of Tracked Point (Exponential Fit)"
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data', alpha=0.5)
    ml_conversion = [(t*rate)/60 for t in time_seconds]
    plt.plot(ml_conversion, y_fit_at_data_points, color='red', label='Exponential Fit')
    #plt.fill_between(x_fit, y_fit - dy, y_fit + dy, color='red', alpha=0.2, label='±1σ Error Band')
    plt.xlabel("Pore Volume Delivered (ml)")
    plt.ylabel("Velocity (mm/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(video_name + "point_tracked_velocity_exponential.png")
    
    #return time_seconds, title, smoothed_velocity, smoothed_std
    #return x_fit, title, y_fit,dy
    return ml_conversion, title, smoothed_velocity, smoothed_std, y_fit_at_data_points




def frames_to_video(frame_folder, output_path, fps=30):
    # Get sorted list of image filenames
    images = sorted([f for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg'))])

    if not images:
        print("No frames found.")
        return

    # Read first frame to get dimensions
    first_frame_path = os.path.join(frame_folder, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    # Define video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Write each frame to the video
    for img_name in images:
        img_path = os.path.join(frame_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Video saved to: {output_path}")

def area(path,frame_folder, threshold):
    print('running')
    all_files = sorted(f for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg')))
    frame_files = all_files
    green_pixels = set()
    green_pixel_counts = []
    choose_img_path = os.path.join(frame_folder, frame_files[int(len(frame_files)/2)])
    choose_img = cv2.imread(choose_img_path)

    match = re.search(r'\d+', path)
    if match:
        rate = int(match.group())  # ➝ 3
        print("Rate (mL/min):", rate)
    else:
        raise ValueError("No number found in path")

    # Let user draw a circular ROI on the first frame
    roi_mask = np.zeros(choose_img.shape[:2], dtype=np.uint8)
    roi_center = []
    radius = 379

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_center, radius
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_center = [x, y]
        
    

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", mouse_callback)
    cv2.imshow("Draw ROI", choose_img)
    while not roi_center:
        cv2.waitKey(1)

    print('point selected')
    cv2.destroyAllWindows()

    if not roi_center :
        print("ROI selection cancelled.")
        return

    # Create circular mask
    mask = np.zeros(choose_img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, tuple(roi_center), radius, 1, -1)

    # marked_img = choose_img.copy()
    # cv2.circle(marked_img, tuple(roi_center), radius[0], (255, 255, 255), 2)
    # cv2.imwrite("marked_roi.png", marked_img)
    # print("Saved ROI image as 'marked_roi.png'")


    for filename in frame_files:
        print('one done')
        paths = os.path.join(frame_folder, filename)
        img = cv2.imread(paths)

        # Extract green channel
        green = img[:, :, 1]

        # Count pixels above threshold
        green_mask = (green > threshold)&(mask==1)
        count = np.count_nonzero(green_mask)
        green_coords = np.column_stack(np.where(green_mask))
        # Convert to list of (x, y) tuples if needed
        for coord in green_coords:
            green_pixels.add(tuple(coord[::-1]))
        green_pixel_counts.append(len(green_pixels) * (mm_per_pixel ** 2))

    
    # Plot
    fps=30
    title=  "Total Green Area Over Time"
    print(counter*fps)
    times_in_seconds = [i / fps for i in range(len(green_pixel_counts))]
    print(times_in_seconds)
    ml_conversion = [(t*rate)/60 for t in times_in_seconds]
    plt.figure(figsize=(10, 5))
    plt.plot(ml_conversion,green_pixel_counts, color='green')
    plt.title(title)
    plt.xlabel("Pore Volume Delivered (ml)")
    plt.ylabel("Area in mm$^2$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(video_name+"green_pixel_counts.png")
    

    return ml_conversion,title, green_pixel_counts




def overlaid_plots(x_axes,dict,title, dict2 = {}, dict3={}):
    #dicts are y axes
    plt.figure(figsize=(10, 6))
    names = list(dict.keys())
    #or this for constant set of videos and order 
    names = ['1 ml/min','3 ml/min', '5 ml/min']
    x_axis = list(x_axes.values())

    values = list(dict.values())
    values2=list(dict2.values())
    values3 =list(dict3.values())
    colors = ['orange','blue','green']

    max_len = max(len(x) for x in x_axes.values())
    #for velocity 
    #max_x_axis = next(x for x in x_axes.values() if len(x) == max_len) 

    # for name, x_vals in x_axes.items():
    #     print(f"{name}: len={len(x_vals)}, first={x_vals[:5]}, last={x_vals[-5:]}")
    # all_x = [x for sublist in x_axes.values() for x in sublist]
    # all_y = [y for sublist in dict.values() for y in sublist]

    # plt.xlim(min(all_x), max(all_x))
    # plt.ylim(min(all_y), max(all_y))
    

    for i in range(len(dict)):
        # Pad values to match longest lengths
        #padded = np.array(list(values[i]) + [np.nan] * (max_len - len(values[i])))
        #for area (to preserve x axis for each experiment)
        #padded_x  = np.array(list(x_axis[i]) + [np.nan] * (max_len - len(x_axis[i])))
        # #area version
        x_vals = np.array(x_axis[i])
        y_vals = np.array(values[i])

        # Find first index where y > 0
        valid_start_idx = np.argmax(y_vals > 0)

        x_trimmed = x_vals[valid_start_idx:] - x_vals[valid_start_idx]
        y_trimmed = y_vals[valid_start_idx:]
        plt.plot(x_trimmed, y_trimmed, label=names[i])


        #velocity version
        # padded_std = np.array(list(values2[i]) + [np.nan] * (max_len - len(values2[i])))
        # #for doing both exponential and smoothed, this is exponential y values
        # padded_exp = np.array(list(values3[i]) + [np.nan] * (max_len - len(values2[i])))
        # title = "Velocity of Tracked Point (Exponential and Smoothed Fit)"
        # if i==0:
        #     plt.fill_between(
        #         x_axis[i],
        #         values[i] - values2[i],
        #         values[i] + values2[i],
        #         alpha=0.3,
        #         color = colors[i],
        #         label='±1 SD')
        # else:
        #    plt.fill_between(
        #         x_axis[i],
        #         values[i] - values2[i],
        #         values[i] + values2[i],
        #         color = colors[i],
        #         alpha=0.3,
        #         ) 
        # plt.plot(x_axis[i], values[i],color = colors[i], label=names[i]+'(Smoothed)')

        # print(f"Plotted {names[i]}")
        # print('plotted one line')

        #exponential velocity version
        #padded_std is not actually std, but is error
        #plt.plot(max_x_axis,padded_exp,linestyle='--',color = colors[i], label=names[i]+"(Exponential)")
        #plt.fill_between(max_x_axis, padded -padded_std, padded + padded_std, alpha=0.2, label='±1σ Error Band')

    

    plt.title(title)
    plt.xlabel("Pore Volume Delivered (ml) ")
    plt.ylabel("d(Area)/dt (mm/s)")  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(title+"_comparison_plot.png")


def get_values(*args):
    x_axes ={}
    std = []
    for arg in args:
        path = arg
        video_name = os.path.splitext(os.path.basename(path))[0]
        folder, files, counter = extract(path)
        print("Returned from extract:")
        print("  folder:", folder)
        print("  files (count):", len(files) if files else "None")
        print("  counter:", counter)
       
        
        #choose which plot to make overlaid version of
        # x_axes is dictionary where key is path and value is list of times, y is list of area
        #area version
        #x_axes[arg], title, y_axis = area(arg,folder,50)
        # print('ran area')

        #velocity version
        #std is list of standard deviation for error shading
        #x_axes[arg], title, y_axis, std = velocity_by_points(folder,files,int(input("Starting frame for "+arg) ))
        #x_axes[arg], title, y_axis, std,y_axis_exp = velocity_by_points(arg,folder,files,int(input("Starting frame for "+arg) ), (input("Ending frame for "+arg)))
        # x_axes[arg], title, y_axis,dy = velocity_by_points(folder,files,int(input("Starting frame for "+arg) ))
        
        #for derivative
        x,title,y = area(arg,folder,50)
        x_axes[arg], title, y_axis = derivative(path,x,y)

        #x_axis, title, y_axis,na,na2 = plotting(avg,g, path)
        #x_axis, na, na2,title,x_axis = plotting(avg,g, path)
        
        #creates dictionary where key is path name and value is x values
        points[arg]=y_axis
        # if len(std)>0:
        #     error[arg] = std
        # elif len(dy)>0:
        #     error[arg]=dy
        # else:
        #     print("no std")
        # print('added line values')

        # if len(y_axis_exp)>0: 
        #     points2[arg]=y_axis_exp
        

    

    return x_axes, points, title, error, points2








path = '1mlmin_22C_Trimmed.MP4' 
video_name = os.path.splitext(os.path.basename(path))[0]
mm_per_pixel = 60/378.08
equations = []
folder, files, counter = extract(path)
#max = find_center(folder,files, counter)
#avg,g = find_intensity(folder, files, max)
#plotting(avg,g,path)
#derivative(path,avg)
#folder = 'frames/'
#files = sorted(f for f in os.listdir(folder) if f.endswith(('.png', '.jpg')))

#black_contours(folder,files, 570)

#last two arguments are staring and ending frames (x just goes to end)
velocity_by_points(path,folder,files,480,'x')
#print(len(files))
#contours(folder,files,path,50)
#frames_to_video('tracked_frames',video_name+'tracked_video.mp4')
#frames_to_video('point_tracked_frames',video_name+'point_tracked_video.mp4')
#x,title,y=area(folder,50)
# derivative(path,x,y)
points = {}
points2={}
error = {}
counter = 0

#overlaid_plots(*get_values('1mlmin_22C_Trimmed.MP4','3mlmin_22C.MP4','5mlmin_trimmed.MP4'))
print(equations)









 
#12 cm

#check --- quantify area plots + plot on the same plot for all 4 experiments w legend (units should be mm^2/s)
    #area plots were wrong, double counting
#check--quantify velocity plots to be in mm/sec + plot the best fit on the same plot for the 3 experiments, shaded error bars for the actual point values
#check?---4.4C experiment consistency map
    #not really consistent enough to skip so many, doesn't show opacity difference
#Use less frames for consistency map to show space between each frame
#Check--Subpixel interpolation to smooth velocity front results

#print out equations
