import cv2
import os


def images_to_video(image_folder, output_video, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure the images are in the correct order

    if not images:
        print("No images found in the folder.")
        return

    # Get the dimensions of the images
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")


# Example usage
image_folder = "C:\\Users\\abhin\\OneDrive\\Desktop\\jum1_images"
output_video = 'output_video.mp4'
fps = 30  # Frames per second

images_to_video(image_folder, output_video, fps)
