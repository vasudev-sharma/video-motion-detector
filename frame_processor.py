import cv2


def process_video(video_path, target_fps=5, resize_dim=(1280, 720)):
    """
    Processes a video file by extracting frames at a specified target frames per second (FPS)
    and resizing each extracted frame to the given dimensions.
    Args:
        video_path (str): Path to the input video file.
        target_fps (int, optional): The desired number of frames per second to extract from the video. Defaults to 5.
        resize_dim (tuple, optional): The target size for resizing each frame as (width, height). Defaults to (1280, 720).
    Returns:
        list: A list of numpy.ndarray objects, each representing a resized frame extracted from the video.
    Raises:
        ValueError: If the video file cannot be opened.
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
