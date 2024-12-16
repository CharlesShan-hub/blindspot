import cv2
import click

@click.command()
@click.option('--src', type=str, required=True)
@click.option('--dest', type=str, required=True)
@click.option('--width', type=int, required=True, help='Width of the subvideo.')
@click.option('--height', type=int, required=True, help='Height of the subvideo.')
def extract_center_subvideo(src, dest, width, height):
    """
    Extract a subvideo of specified width and height from the center of the input video.
    """
    # Open the input video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        click.echo("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for MP4 format

    # Calculate the starting point for the center subvideo
    x_start = (frame_width - width) // 2
    y_start = (frame_height - height) // 2

    # Create VideoWriter object for the output video
    out = cv2.VideoWriter(dest, fourcc, fps, (width, height))

    # Read frames and extract subvideo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the center subvideo
        subvideo_frame = frame[y_start:y_start+height, x_start:x_start+width]

        # Write the frame to the output video
        out.write(subvideo_frame)

    # Release everything when done
    cap.release()
    out.release()
    click.echo(f"Subvideo saved as {dest}")

if __name__ == '__main__':
    extract_center_subvideo()
