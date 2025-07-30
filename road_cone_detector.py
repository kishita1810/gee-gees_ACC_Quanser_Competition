#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Bool

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.depths = Image()
        # Create subscriber to the camera/color_image topic
        self.subscriber_depth = self.create_subscription(Image, 'camera/depth_image', self.calldepth, qos_profile)
        self.subscriber = self.create_subscription(Image, 'camera/color_image', self.callback, qos_profile)
        
        #self.publisher = self.create_publisher(Image, '/detected_cones', qos_profile)
        self.depth_pub = self.create_publisher(Float32, '/cone_depth', qos_profile)
        self.cone_flags = self.create_publisher(Bool, '/cone_flag', qos_profile)

    def calldepth(self, depth_message):
        self.depths = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding='16UC1')
        
    def callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image using CvBridge
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if frame is None or frame.size == 0:
                return
            
            #resize if necessary.
            #frame = cv2.resize(frame, (640, 480))

            # Convert the current frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the orange color range for detecting road cones.
            lower_orange = np.array([10, 150, 150])  # Lower bound for orange color in HSV
            upper_orange = np.array([18, 255, 255]) # Upper bound for orange color in HSV

            # Threshold the image to get the mask for orange color.
            mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # Apply some filtering to reduce noise.
            mask = cv2.bilateralFilter(mask, 9, 75, 75)

            # Perform morphological transformations to connect detect orange parts.
            # This could overlap and cover white strip.
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cone_alert = Bool()
            cone_alert.data = False
            # Loop through the contours and filter for cone-like shapes
            for contour in contours:
                
                if cv2.contourArea(contour) < 15:  # set to 3 for small sized ones.
                    continue

                # Get the bounding box for the contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(h) / w
                if aspect_ratio < 0.3:  # Skip wide or square-ish shapes
                    continue
                centroid_x = x + w // 2
                centroid_y = y + h // 2

                
                
                if self.depths is not None:  # Ensure depth image is available
                # Get the depth value at the centroid of the bounding box
                    depth_value = self.depths[centroid_y, centroid_x]  # Corrected indexing
                    depth_value = depth_value/1000
                    #self.get_logger().info(f"Centroid at ({centroid_x}, {centroid_y}) - Depth: {depth_value} meters")
                    
                    depth_msg = Float32()
                    depth_msg.data = depth_value
                    self.depth_pub.publish(depth_msg)
                    
                else:
                    self.get_logger().warn("Depth image is not available yet.")

                cone_alert.data = True
                # Draw the bounding box for detected cones
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (centroid_x, centroid_y), 2, (0, 255, 0), -1)
            self.cone_flags.publish(cone_alert)

            # Convert the processed OpenCV image back to ROS Image message
            cone_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

            # Publish the processed image with detected cones
            #self.publisher.publish(cone_image)

            # Show the original frame
            #cv2.imshow("Original Frame", frame)

            # Show the HSV image
            #cv2.imshow("HSV Image", hsv)

            # Show the binary mask (shows the detected cone regions)
            #cv2.imshow("Mask", mask)
            
            # Wait for keypress to close the windows (for debugging)
            #cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")
               

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    cone_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



