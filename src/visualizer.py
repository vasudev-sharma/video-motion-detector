# visualizer.py
"""
Visualization functions for displaying motion detection and viewport tracking results.
"""

import os
import cv2


def visualize_results(
    frames, motion_results, viewport_positions, viewport_size, output_dir
):
    """
    Create visualization of motion detection and viewport tracking results.

    Args:
        frames: List of video frames
        motion_results: List of motion detection results for each frame
        viewport_positions: List of viewport center positions for each frame
        viewport_size: Tuple (width, height) of the viewport
        output_dir: Directory to save visualization results
    """
    # Create output directory for frames
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    viewport_dir = os.path.join(output_dir, "viewport")
    os.makedirs(viewport_dir, exist_ok=True)

    # Get dimensions for the output video
    height, width = frames[0].shape[:2]

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "motion_detection.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    viewport_video_path = os.path.join(output_dir, "viewport_tracking.mp4")
    vp_width, vp_height = viewport_size
    viewport_writer = cv2.VideoWriter(
        viewport_video_path, fourcc, 5, (vp_width, vp_height)
    )

    # Example starter code:
    for i, frame in enumerate(frames):
        # Your implementation here

        frame_copy = frame.copy()

        for box in motion_results[i]:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0))

        # Draw viewport rectangle
        cx, cy = viewport_positions[i]
        x1 = int(max(cx - vp_width // 2, 0))
        y1 = int(max(cy - vp_height // 2, 0))
        x2 = int(min(cx + vp_width // 2, width))
        y2 = int(min(cy + vp_height // 2, height))

        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            frame_copy,
            f"frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Viewport rectangle", frame_copy)
        cv2.waitKey(0)

        # viewport crop
        crop = frame_copy[y1:y2, x1:x2]

        # Save frame as images
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.jpg"), frame_copy)
        cv2.imwrite(os.path.join(viewport_dir, f"viewport_{i:04d}.jpg"), crop)

        # Frames to  vide writers
        # Write frames to videos
        video_writer.write(frame_copy)
        # Resize viewport crop to fixed size in case of edge crop
        viewport_writer.write(cv2.resize(crop, (vp_width, vp_height)))

    video_writer.release()
    viewport_writer.release()

    print(f"Visualization saved to {video_path}")
    print(f"Viewport video saved to {viewport_video_path}")
    print(f"Individual frames saved to {frames_dir} and {viewport_dir}")
