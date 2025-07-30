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

class StopDetector(Node):
    def __init__(self):
        super().__init__('stop_sign_detector')
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        self.subscription_stop = self.create_subscription(Image, '/camera/color_image', self.stop_callback, qos_profile)
        self.subscriber_depth = self.create_subscription(Image, '/camera/depth_image', self.calldepth, qos_profile)
        
        self.subscriber_yield_color = self.create_subscription(Image, '/camera/color_image', self.yield_callback, qos_profile)
        self.subscriber_yield_depth = self.create_subscription(Image, '/camera/depth_image', self.calldepth_yield, qos_profile)
        
        self.depth_stop = self.create_publisher(Float32, '/stop_depth', qos_profile)
        self.depth_yield = self.create_publisher(Float32, '/yield_depth', qos_profile)
        self.yield_flag = self.create_publisher(Bool, '/yield_flag', qos_profile)
        self.stop_flags = self.create_publisher(Bool, '/stop_flag', qos_profile)
        self.depths = None
        self.depths_yield = None
        
        self.stflag = Float32()
        self.yiflag = Float32()
        
    def calldepth_yield(self, yielddepth):
        self.depths_yield = self.bridge.imgmsg_to_cv2(yielddepth, desired_encoding='16UC1')
        
    def calldepth(self, depth_message):
        self.depths = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding='16UC1')
    

    def yield_callback(self, yieldmsg):
        frameforyield = self.bridge.imgmsg_to_cv2(yieldmsg, desired_encoding='bgr8')
        if frameforyield is None:
            return
    
        frameforyield_cropped = frameforyield[300:1280, :]
        # Look for yield sign.
        image_yield, yield_flag, yield_x, yield_y = self.detect_yield_sign(frameforyield_cropped.copy())
        #cv2.imshow("yield", image_yield)
        #cv2.waitKey(2)

        yield_alert = Float32()

        # Check if yield_x is not None before accessing depth image
        if yield_x is not None:
            if self.depths_yield is not None:
                depth_from_yield_sign = float(self.depths_yield[yield_y, yield_x] / 1000.0)

                depthyy_msg = Float32()  # Correct initialization
                depthyy_msg.data = float(depth_from_yield_sign)  # Correct assignment of float value
                self.depth_yield.publish(depthyy_msg)
                
        else:
            yield_alert.data = 0.0  # no yield sign detected
 
    
    def stop_callback(self, stopmsg):
        frameforstop = self.bridge.imgmsg_to_cv2(stopmsg, desired_encoding='bgr8')
        if frameforstop is None:
            return
        #Look for stop sign.
        frameforstop_cropped = frameforstop[300:1280, :]
        image_stop, stop_flag, stop_x, stop_y = self.detect_stop_sign(frameforstop_cropped.copy())
        
        if self.depths is not None:
            #cv2.imshow("stop", image_stop)
            #cv2.waitKey(2)
            depth_from_stop_sign = self.depths[stop_y, stop_x] / 1000.0
            #depth_msg_stop = Float32()
            #depth_msg_stop.data = float(depth_from_stop_sign)
            if stop_x is not None:
                depth_msg = Float32()
                depth_msg.data = depth_from_stop_sign
                self.depth_stop.publish(depth_msg)           

            #self.get_logger().info(f"Depth from Stop Sign: {depth_from_stop_sign:.2f} centimeters")
        
        
    def detect_stop_sign(self, image):
        gray_stop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_stop = cv2.GaussianBlur(gray_stop, (1, 1), 0)
        edges_stop = cv2.Canny(blur_stop, 50, 150)
        stop_sign_detected = False
        centroid_x_stop = None
        centroid_y_stop = None
        
        # Find contours in the edge-detected image for stop sign
        contours, _ = cv2.findContours(edges_stop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert the frame to HSV color space for color detection
        hsv_stop = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range of red color in HSV space (stop sign is typically red)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask for red areas in the image
        mask1_stop = cv2.inRange(hsv_stop, lower_red1, upper_red1)
        mask2_stop = cv2.inRange(hsv_stop, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1_stop, mask2_stop)
        
        # Process each contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40:  # Ignore small contours
                continue

            stop_sign_msg = Bool()
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            # Check if the shape is an octagon (8 sides)
            if len(approx) >= 7 and len(approx) <= 8:
                # Check if the area of the detected octagon is within a region that has red color
                x, y, w, h = cv2.boundingRect(approx)
                roistop = red_mask[y:y+h, x:x+w]  # Region of interest from red mask
                red_area = cv2.countNonZero(roistop)  # Count non-zero pixels in the red mask

                aspect_ratio = float(w) / h
                if aspect_ratio >= 0.79 and aspect_ratio <= 1.5: #0.94:
                    continue
                
                
                # If the red area is sufficiently large, it's a stop sign
                if red_area > (w * h * 0.60):  # Threshold for the amount of red in the region
                    stop_sign_detected = True
                    stop_sign_msg.data = True
                    self.stop_flags.publish(stop_sign_msg)
                    # Draw the bounding box around the detected stop sign
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    centroid_x_stop = x + w // 2
                    centroid_y_stop = y + h // 2
                    print(aspect_ratio)
                else:
                    stop_sign_msg.data = False
                    self.stop_flags.publish(stop_sign_msg)
        return image, stop_sign_detected, centroid_x_stop, centroid_y_stop
        
    def detect_yield_sign(self, imageforyield):
        #gray_yield = cv2.cvtColor(imageforyield, cv2.COLOR_BGR2GRAY)
        #blur_yield = cv2.GaussianBlur(gray_yield, (5, 5), 0)
        #edges_yield = cv2.Canny(blur_yield, 100, 250)
        yield_sign_detected = False
        centroid_x_yield = None
        centroid_y_yield = None
        
        # Convert the frame to HSV color space for color detection
        hsv_yield = cv2.cvtColor(imageforyield, cv2.COLOR_BGR2HSV)
        
        lower_red1_y = np.array([0, 120, 70])  # Lower bound of red (light red)
        upper_red1_y = np.array([10, 255, 255])  # Upper bound of red (light red)

        lower_red2_y = np.array([170, 120, 70])  # Lower bound of red (dark red)
        upper_red2_y = np.array([180, 255, 255])  # Upper bound of red (dark red)

        # Define the HSV range for white (inner triangle) of the yield sign
        lower_white_y = np.array([0, 0, 200])  # Lower bound of white (high brightness)
        upper_white_y = np.array([180, 25, 255])  # Upper bound of white (low saturation)
        
        # -----------
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask for red areas in the image
        #mask1_stop = cv2.inRange(hsv_stop, lower_red1, upper_red1)
        #mask2_stop = cv2.inRange(hsv_stop, lower_red2, upper_red2)
        #red_mask = cv2.bitwise_or(mask1_stop, mask2_stop)

        # Create masks for red and white areas
        mask1_y = cv2.inRange(hsv_yield, lower_red1, upper_red1)
        mask2_y = cv2.inRange(hsv_yield, lower_red2, upper_red2)
        red_mask_y = cv2.bitwise_or(mask1_y, mask2_y)

        # Create mask for white areas (inner triangle of yield sign)
        white_mask_y = cv2.inRange(hsv_yield, lower_white_y, upper_white_y)

        # Combine the red and white masks
        combined_mask_y = cv2.bitwise_or(red_mask_y, white_mask_y)
        # Apply morphological closing to bridge gaps between red and white regions
        kernelw = np.ones((2, 2), np.uint8)
        combined_mask_y = cv2.morphologyEx(combined_mask_y, cv2.MORPH_CLOSE, kernelw)

        # Find contours in the edge-detected image for stop sign
        contours_yield, _ = cv2.findContours(red_mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #cv2.("Combined Mask", combined_mask_y)
        
        triangle_template = np.array([[[0, 100]], [[50, 0]], [[100, 100]]], dtype=np.int32)
        triangle_template_cnt = triangle_template.reshape((-1, 1, 2))
        
        # Process each contour
        for cnt_y in contours_yield:
            area_y = cv2.contourArea(cnt_y)
            if area_y < 2:  # Ignore small contours
                continue

            # Approximate the contour to a polygon
            approx_y = cv2.approxPolyDP(cnt_y, 0.04 * cv2.arcLength(cnt_y, True), True)
            yield_sign_msg = Bool()
            # Check if the shape is an octagon (8 sides)
            if len(approx_y) >= 2 and len(approx_y) <= 5:
                # Check if the area of the detected octagon is within a region that has red color
                xy, yy, wy, hy = cv2.boundingRect(approx_y)
                roiy = red_mask_y[yy:yy+hy, xy:xy+wy]  # Region of interest from red mask
                roi_white = white_mask_y[yy:yy+hy, xy:xy+wy]
                
                red_area_y = cv2.countNonZero(roiy)  # Count non-zero pixels in the red mask
                white_area = cv2.countNonZero(roi_white)
                
                match = cv2.matchShapes(cnt_y, triangle_template_cnt, 1, 0.0)
                
                if red_area_y > (wy * hy * 0.1) and white_area > (wy * hy * 0.1):  # Threshold for the amount of red in the region\
                    yield_sign_detected = True
                   
                    yield_sign_msg.data = True
        
                    # Publish the message
                    self.yield_flag.publish(yield_sign_msg)
                        
                        # Draw the bounding box around the detected yield sign
                    cv2.rectangle(imageforyield, (xy, yy), (xy + wy, yy + hy), (0, 255, 0), 3)
                    centroid_y_yield = yy + hy // 2
                else: 
                    yield_sign_msg.data = False
        
                    # Publish the message
                    self.yield_flag.publish(yield_sign_msg)
        return imageforyield, yield_sign_detected, centroid_x_yield, centroid_y_yield
               

def main(args=None):
    rclpy.init(args=args)
    stop_sign_detector = StopDetector()
    rclpy.spin(stop_sign_detector)
    stop_sign_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
