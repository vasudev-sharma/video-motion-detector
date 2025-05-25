import cv2


def process_video(video_path, target_fps=5, resize_dim=(1280, 720)):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize_dim: Dimensions to resize frames to (width, height)

    Returns:
        List of extracted frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    print("The captured vide is", cap)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval for the target FPS
    frame_interval = max(1, int(original_fps / target_fps))

    # Example starter code:
    frames = []
    frame_index = 0

    print(f"The frame count is {frame_count} and {frame_interval}")
    while cap.isOpened():
        # reading frames
        success, image = cap.read()
        frame_index += 1

        # Target Frames
        if success and (frame_index % frame_interval == 0):
            # Resize image
            frames.append(cv2.resize(image, resize_dim))
        elif success:
            continue
        else:
            break

    # Release the vide
    cap.release()
    print(frames[0].shape)


    return frames
