import sys
import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
import time


class GazeboListener:
    def __init__(self):
        self.node = Node()
        self.image_data = None
        self.image_received = False
        
    def image_callback(self, msg: Image):
        """
        Callback function when image message is received
        """

        print(msg.width, msg.height, msg.pixel_format_type)

        width = msg.width
        height = msg.height
        pixel_format = msg.pixel_format_type
        

        image_data = np.frombuffer(msg.data, dtype=np.uint8)

        print(image_data)
        
        if pixel_format == 3:
            image = image_data.reshape((height, width, 3))
            print(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        self.image_data = image
        self.image_received = True
        
        cv2.imshow("Gazebo Camera Feed", image)
        cv2.waitKey(1)
        
    
    def subscribe_to_camera(self, topic_name="/camera"):
        """
        Subscribe to a camera topic
        """
        if not self.node.subscribe(Image, topic_name, self.image_callback):
            print(f"Error subscribing to topic: {topic_name}")
            return False
        
        print(f"Successfully subscribed to: {topic_name}")
        return True
    
    def save_image(self, filename="gazebo_image.png"):
        """
        Save the current image to file
        """
        if self.image_data is not None:
            cv2.imwrite(filename, self.image_data)
            print(f"Image saved as: {filename}")
            return True
        else:
            print("No image data available to save")
            return False
    
    def run(self):
        """
        Main loop to keep the subscriber running
        """
        print("Listening for images... Press Ctrl+C to stop")
        print("Press 's' to save the current image")
        
        try:
            while True:
                time.sleep(0.1)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    timestamp = int(time.time())
                    self.save_image(f"gazebo_image_{timestamp}.png")
                elif key == ord('q'):
                    print("Quitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    listener = GazeboListener()

    print("STARTING.....")

    topic_name = ""
    
    if len(sys.argv) > 1:
        topic_name = sys.argv[1]
    
    print(f"Attempting to subscribe to: {topic_name}")
    
    if listener.subscribe_to_camera(topic_name):
        listener.run()
    else:
        print("Failed to subscribe to camera topic")