import numpy as np
import cv2

# File selection
filename = "Data/ukulele2/depth.npy"  # Update this to your desired file

# Load the depth frames
depth_frames = np.load(filename)

# Define the depth range (adjust for each object as needed)
lower_bound = 1
upper_bound = 650

# Apply the range filter
depth_frames[(depth_frames < lower_bound) | (depth_frames > upper_bound)] = 0

paused = False  # Pause state
current_frame = 0  # Track the current frame number
pause_frames = [
    72,
    95,
    110,
    122,
    132,
    142,
    151,
    160,
    170,
    178,
    185,
]  # Frames to pause at
already_paused = set()  # Track frames that have been paused programmatically

while current_frame < depth_frames.shape[0]:
    # Check if we should pause programmatically
    if current_frame in pause_frames and current_frame not in already_paused:
        paused = True
        already_paused.add(current_frame)  # Mark this frame as paused
        print(f"Programmatically paused at frame: {current_frame}")

    if not paused:
        frame = depth_frames[current_frame, 70:240, 0:260]

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
    key = cv2.waitKey(0 if paused else 50) & 0xFF  # Wait indefinitely if paused

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
        cv2.imwrite(f"ukulele2_frames/frame_{current_frame - 1}.png", resized_frame)
        print(f"Saved frame: {current_frame - 1}")

# Clean up
cv2.destroyAllWindows()
