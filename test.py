import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def load_video(filename):
    """Load video frames from a file."""
    cap = cv2.VideoCapture(filename)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def gaussian_pyramid(frame, levels):
    """Construct a Gaussian pyramid for a given frame."""
    pyramid = [frame]
    for _ in range(levels - 1):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def apply_temporal_filter(frames, filter_kernel):
    """Apply a temporal filter across video frames."""
    filtered_frames = []
    kernel_len = len(filter_kernel)

    for i in range(len(frames)):
        filtered_frame = np.zeros_like(frames[i], dtype=np.float32)
        for k in range(kernel_len):
            if i + k < len(frames):
                filtered_frame += frames[i + k] * filter_kernel[k]
        filtered_frames.append(filtered_frame)

    return filtered_frames

def amplify_and_add(frames, filtered_frames, amplification_factor):
    """Amplify the filtered frames and add them back to the original frames."""
    amplified_frames = []
    for original, filtered in zip(frames, filtered_frames):
        amplified = cv2.addWeighted(original.astype(np.float32), 1.0, filtered, amplification_factor, 0)
        amplified_frames.append(amplified.astype(np.uint8))
    return amplified_frames

def reconstruct_video(pyramid):
    """Reconstruct the video from its Gaussian pyramid."""
    reconstructed_frame = pyramid[0]
    for i in range(1, len(pyramid)):
        reconstructed_frame = cv2.pyrUp(reconstructed_frame)
        # Ensure the dimensions match
        if reconstructed_frame.shape != pyramid[i].shape:
            reconstructed_frame = cv2.resize(reconstructed_frame, (pyramid[i].shape[1], pyramid[i].shape[0]))
        reconstructed_frame = cv2.add(reconstructed_frame, pyramid[i])
    return reconstructed_frame

def calculate_pulse(frames):
    """Calculate the average color of each frame, simulating pulse detection."""
    pulse_data = []
    for frame in frames:
        avg_color = np.mean(frame, axis=(0, 1))
        pulse_data.append(avg_color)
        return pulse_data
    return np.array(pulse_data)

def save_video(frames, output_filename, fps):
    """Save processed frames as a video."""
    if len(frames) == 0:
        print("No frames to save.")
        return

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def main(video_file):
    # Load video
    frames = load_video(video_file)
    if not frames:
        print("Failed to load video.")
        return

    fps = cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FPS)

    # Gaussian pyramid decomposition
    pyramids = [gaussian_pyramid(frame, levels=3) for frame in frames]

    # Temporal filtering
    filter_kernel = [0.25, 0.5, 0.25]  # Example filter kernel
    filtered_frames = apply_temporal_filter(frames, filter_kernel)

    # Amplify and add back
    amplification_factor = 1.5
    amplified_frames = amplify_and_add(frames, filtered_frames, amplification_factor)

    # Reconstruct video from pyramid (not necessary for output in this context)
    # reconstructed_video = [reconstruct_video(pyr) for pyr in pyramids]

    # Calculate pulse
    pulse = calculate_pulse(amplified_frames)
    print("Pulse data:", pulse)

    # Save amplified video
    output_filename = "amplified_video_vikas.mp4"
    save_video(amplified_frames, output_filename, fps)
    print(f"Amplified video saved as {output_filename}")

if __name__ == "__main__":
    main("sample.mp4")