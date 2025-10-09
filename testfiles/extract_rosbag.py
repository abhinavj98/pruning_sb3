import os
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def extract_images(bag_path, image_topic, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_type_dict = {t.name: t.type for t in topics_and_types}

    if image_topic not in topic_type_dict:
        print(f"Topic '{image_topic}' not found in the bag file.")
        return

    msg_type = get_message(topic_type_dict[image_topic])
    bridge = CvBridge()
    count = 0
    frame_paths = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == image_topic:
            msg = deserialize_message(data, msg_type)
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                filename = os.path.join(output_dir, f"frame_{count:05d}.png")
                cv2.imwrite(filename, cv_image)
                frame_paths.append(filename)
                count += 1
            except Exception as e:
                print(f"Error converting image at index {count}: {e}")

    print(f"Saved {count} images to {output_dir}")
    if count > 0:
        make_video_from_images(frame_paths, os.path.join(output_dir, "output_video.mp4"))

def make_video_from_images(image_paths, video_path, fps=30):
    print(f"Creating video at {video_path}")
    image_paths = sorted(image_paths)
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        out.write(img)
        #Delete the image after writing to video
        os.remove(img_path)

    out.release()
    print(f"Video saved to {video_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from a ROS 2 bag file and create a video")
    parser.add_argument('--bag', default='2025_pruning_trials/rosbags', help="Path to the folder containing the bag")
    parser.add_argument('--topic', default='/camera/camera/color/image_raw', help="Image topic to extract (e.g., /camera/image_raw)")
    parser.add_argument('--out', default="2025_pruning_trials/videos", help="Output directory to save images and video")
    args = parser.parse_args()
    rclpy.init()

    #Get the list of bag files in the directory as entire path

    for i in os.listdir(args.bag):
        input_bag = os.path.join(args.bag, i)
        out = os.path.join(args.out, i.split("/")[-1])
        print(f"Processing bag file: {input_bag} with output directory: {out}")
        extract_images(input_bag, args.topic, out)
    rclpy.shutdown()
