import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# ==========================================
# 1. Signal Processing (Math & BPM)
# ==========================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Standard Butterworth bandpass filter.
    Filters out noise, leaving only frequencies between 0.8Hz (48 BPM) and 3.0Hz (180 BPM).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def calculate_bpm(frames, fps):
    """
    Analyzes the video to calculate the heart rate (BPM).
    Returns the numeric BPM value.
    """
    print(" [1/3] Analyzing video signal for heart rate...")
    
    raw_signal = []
    for frame in frames:
        # Focus on the center 50% of the frame (face area)
        h, w, _ = frame.shape
        roi = frame[h//4:3*h//4, w//4:3*w//4]
        
        # Extract Green Channel average (Index 1)
        g_avg = np.mean(roi[:, :, 1])
        raw_signal.append(g_avg)

    # Detrend (remove average brightness) and Filter
    signal = np.array(raw_signal)
    signal = signal - np.mean(signal)
    clean_signal = butter_bandpass_filter(signal, 0.8, 3.0, fps, order=4)

    # FFT (Fast Fourier Transform) to find the dominant frequency
    window = np.hanning(len(clean_signal))
    signal_fft = np.abs(np.fft.rfft(clean_signal * window))
    freqs = np.fft.rfftfreq(len(clean_signal), d=1.0/fps)

    # Find the peak frequency within human range (48-180 BPM)
    valid_idx = np.where((freqs >= 0.8) & (freqs <= 3.0))
    valid_fft = signal_fft[valid_idx]
    
    if len(valid_fft) == 0:
        return 0.0
    
    peak_idx = np.argmax(valid_fft)
    dominant_freq = freqs[valid_idx][peak_idx]
    bpm = dominant_freq * 60.0
    
    return bpm

# ==========================================
# 2. Video Magnification (Visuals)
# ==========================================

def magnify_video(frames, fps, bpm_text):
    """
    Applies Eulerian Video Magnification (Color) and overlays the BPM text.
    """
    print(" [2/3] Generating magnified video (this may take a moment)...")
    
    # Convert BGR to YIQ (Y=Light, I=Orange/Blue, Q=Purple/Green)
    video_data = np.array(frames, dtype=np.float32)
    yiq_matrix = np.array([[0.114, 0.587, 0.299], 
                           [-0.321, -0.275, 0.596], 
                           [0.311, -0.523, 0.212]]).T
    yiq_data = np.dot(video_data, yiq_matrix)
    
    # Filter the I and Q channels (Color) - leave Y (Light) alone
    filtered = yiq_data.copy()
    filtered[:, :, :, 1] = butter_bandpass_filter(filtered[:, :, :, 1], 0.8, 2.0, fps)
    filtered[:, :, :, 2] = butter_bandpass_filter(filtered[:, :, :, 2], 0.8, 2.0, fps)
    
    # Amplify the color change
    alpha = 50 
    filtered[:, :, :, 1] *= alpha
    filtered[:, :, :, 2] *= alpha
    
    # Add back to original YIQ
    yiq_data[:, :, :, 1] += filtered[:, :, :, 1]
    yiq_data[:, :, :, 2] += filtered[:, :, :, 2]
    
    # Convert back to BGR
    rgb_matrix = np.array([[1.0, -1.106, 1.703], 
                           [1.0, -0.272, -0.647], 
                           [1.0, 0.956, 0.621]]).T
    bgr_frames = np.dot(yiq_data, rgb_matrix)
    bgr_frames = np.clip(bgr_frames, 0, 255).astype(np.uint8)

    # Overlay BPM Text on every frame
    print(" [3/3] Saving to file...")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter("pulse_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for frame in bgr_frames:
        # Add Green Text: "Pulse: 72 BPM"
        cv2.putText(frame, bpm_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        
    out.release()
    print(" [Done] Video saved as 'pulse_result.mp4'")

# ==========================================
# 3. Main Execution
# ==========================================

def main(video_file):
    print(f"Loading {video_file}...")
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    
    frames = []
    # Limit to 300 frames (approx 10s) to keep it fast
    while cap.isOpened() and len(frames) < 300:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Error: Could not load video.")
        return

    # --- STEP 1: CALCULATE BPM ---
    bpm = calculate_bpm(frames, fps)
    
    # --- STEP 2: DISPLAY IN CONSOLE (The requested output) ---
    print("\n" + "="*40)
    print(f"   DETECTED PULSE: {bpm:.1f} BPM")
    print("="*40 + "\n")

    # --- STEP 3: CREATE VIDEO ---
    bpm_text = f"Pulse: {bpm:.1f} BPM"
    magnify_video(frames, fps, bpm_text)

if __name__ == "__main__":
    main("sample.mp4")
