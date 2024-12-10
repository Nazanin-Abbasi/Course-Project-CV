import numpy as np
import cv2

# File selection
filename = "Data/new_batch/640middleclose_depth.npy"  # Update this to your desired file
# filename = "Data/new_batch/160middleclose_depth.npy"  # Update this to your desired file
# filename = "Data/lama/depth.npy"

# Load the depth frames
depth_frames = np.load(filename)

# Define the depth range (adjust for each object as needed)
lower_bound = 1
upper_bound = 10000

# Apply the range filter
depth_frames[(depth_frames < lower_bound) | (depth_frames > upper_bound)] = 0

paused = False  # Pause state
current_frame = 0  # Track the current frame number
pause_frames = []  # Frames to pause at
already_paused = set()  # Track frames that have been paused programmatically

while current_frame < depth_frames.shape[0]:
    # Check if we should pause programmatically
    if current_frame in pause_frames and current_frame not in already_paused:
        paused = True
        already_paused.add(current_frame)  # Mark this frame as paused
        print(f"Programmatically paused at frame: {current_frame}")

    if not paused:
        # frame = depth_frames[current_frame, 30:150, 100:200]
        frame = depth_frames[current_frame, 10 * 4 : 78 * 4, 50 * 4 : 120 * 4]
        # Normalize the frame to the range [0, 255] and convert to uint8
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_normalized = frame_normalized.astype("uint8")

        # Resize for better visibility (optional)
        resized_frame = frame_normalized
        # cv2.resize(
        #     frame_normalized, (400, 400), interpolation=cv2.INTER_NEAREST
        # )

        # Display the frame
        cv2.imshow("Animation", resized_frame)
        print(f"Showing frame: {current_frame}")
        current_frame += 1

    # Handle key inputs
    key = cv2.waitKey(0 if paused else 100) & 0xFF  # Wait indefinitely if paused

    if key == 27:  # ESC key to exit
        print("Exiting...")
        break
    elif key == ord("p"):  # Toggle pause/unpause
        paused = not paused
        if paused:
            print(f"Paused at frame: {current_frame - 1}")
        else:
            print("Resumed playback.")
    elif key == ord("s") and paused:  # Save current frame while paused
        cv2.imwrite(f"lama/frame_{current_frame - 1}.png", resized_frame)
        print(f"Saved frame: {current_frame - 1}")

# Clean up
cv2.destroyAllWindows()
