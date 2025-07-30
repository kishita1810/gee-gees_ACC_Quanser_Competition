#!/usr/bin/env python3
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
import cv2
import time
from std_msgs.msg import Float32, Bool

class LightPublisher(Node):
    def __init__(self):
        print("Running!")
        super().__init__('traffic_light_publisher')
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,  # Volatile (sensor data) history
            depth=10,  # Buffer depth for messages
            reliability=QoSReliabilityPolicy.RELIABLE  # Best Effort Reliability
        )

        self.redlight = self.create_publisher(Bool, '/traffic_light_red', qos)
        self.greenlight = self.create_publisher(Bool, '/traffic_light_green', qos)
        self.redlightmessage = Bool()
        self.greenlightmessage = Bool()

        self.subscription = self.create_subscription(Image, '/camera/color_image', self.image_callback, qos)
        self.bridge = CvBridge()

        
    def image_callback(self, msg):
        frame_received = msg
        if frame_received is None:
            return
        frame_cropped = self.bridge.imgmsg_to_cv2(frame_received, desired_encoding='bgr8')
        if frame_cropped is None:
            return
        frame  = frame_cropped[0:400, 585:1030]
        
        self.detect(frame)
        

    def detect(self, frame):
        def keep_largest_and_least_error(r_circles, g_circles, y_circles, image_width):
            def get_largest_circle(circles):
                if circles is not None and circles.size > 0:
                    # Convert to a list of circles and find the largest by radius
                    circles = np.uint16(np.around(circles)).reshape(-1, 3)  # Ensure it's a 2D array
                    largest_circle = max(circles, key=lambda x: x[2])  # x[2] is the radius
                    return largest_circle
                return None

            # Step 1: Find the largest circle among all circles
            largest_red = get_largest_circle(r_circles)
            largest_green = get_largest_circle(g_circles)
            largest_yellow = get_largest_circle(y_circles)

            # Combine all largest circles into a list
            largest_circles = [largest_red, largest_green, largest_yellow]
            # Check for non-None circles before finding the largest
            valid_circles = [circle for circle in largest_circles if circle is not None]
            largest_circle = max(valid_circles, key=lambda x: x[2]) if valid_circles else None

            # print(largest_circle)

            if largest_circle is None:
                # print("NONE!")
                return None, None, None  # If no circles are found, return None for all

            largest_circle_radius = largest_circle[2]  # Get the radius of the largest circle
            center_x = image_width / 2

            # Step 2: Calculate the error for each circle
            errors = {}
            
            for color, circles in zip(['red', 'green', 'yellow'], [r_circles, g_circles, y_circles]):
                if circles is not None and circles.size > 0:
                    circles = np.uint16(np.around(circles)).reshape(-1, 3)  # Ensure it's a 2D array
                    for circle in circles:
                        x, y, radius = circle
                        error = np.sqrt(((x - center_x) / 50) ** 2 + (radius - largest_circle_radius) ** 2)
                        errors[color] = (circle, error) if color not in errors else min(errors[color], (circle, error), key=lambda e: e[1])

            # Step 3: Find the circle with the least error
            least_error_circle = min(errors.values(), key=lambda e: e[1]) if errors else None
            # print(least_error_circle)
            # Prepare the output arrays
            r_circles_out = None
            g_circles_out = None
            y_circles_out = None
            # print("Errors", errors)
            if least_error_circle is not None:
                circle = least_error_circle[0]  # Get the circle
                x, y, radius = circle  # Unpack the circle properties
                # print('circle', circle)
                # Check which color the circle belongs to by comparing individual properties
                if r_circles is not None and r_circles.size > 0:
                    for r_circle in r_circles.reshape(-1, 3):
                        if np.array_equal(r_circle, circle):
                            r_circles_out = np.array([[circle]], dtype=np.uint16)
                            break

                if g_circles is not None and g_circles.size > 0:
                    for g_circle in g_circles.reshape(-1, 3):
                        if np.array_equal(g_circle, circle):
                            g_circles_out = np.array([[circle]], dtype=np.uint16)
                            break

                if y_circles is not None and y_circles.size > 0:
                    for y_circle in y_circles.reshape(-1, 3):
                        if np.array_equal(y_circle, circle):
                            y_circles_out = np.array([[circle]], dtype=np.uint16)
                            break

            return r_circles_out, g_circles_out, y_circles_out


        font = cv2.FONT_HERSHEY_SIMPLEX
        img = frame
        cimg = img
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title('Original Image')
        # plt.imshow(cimg)
        # plt.axis('off')


        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # color range
        # lower_red1 = np.array([0,100,100])
        # upper_red1 = np.array([10,255,255])
        # lower_red2 = np.array([160,100,100])
        # upper_red2 = np.array([180,255,255])
        # lower_green = np.array([40,50,50])
        # upper_green = np.array([90,255,255])
        lower_red1 = np.array([1, 15, 230]) #correct s it was 180 not 15
        upper_red1 = np.array([40, 255, 255])
        lower_red2 = np.array([160, 15, 230]) #correct s
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([89, 2, 254]) #simulated 's' was 150
        upper_green = np.array([91, 4, 255])

        # lower_yellow = np.array([15,100,100])
        # upper_yellow = np.array([35,255,255])
        lower_yellow = np.array([15,150,150])
        upper_yellow = np.array([35,255,255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskg = cv2.inRange(hsv, lower_green, upper_green)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        maskr = cv2.add(mask1, mask2)

        size = img.shape
        # print size

        # hough circle detect
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                param1=50, param2=10, minRadius=0, maxRadius=30)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                    param1=50, param2=10, minRadius=0, maxRadius=30)

        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                    param1=50, param2=5, minRadius=0, maxRadius=30)

        # Filter circles based on size and position.
        # r_circles, g_circles, y_circles = keep_center_most_and_largest(r_circles, g_circles, y_circles, size[1])
        # print("Red Circles:", r_circles)
        # print("Green Circles:", g_circles)
        # print("Yellow Circles:", y_circles)
        # traffic light detect
        r = 5
        bound = 4.0 / 10
        r_circles_filtered = []
        g_circles_filtered = []
        y_circles_filtered = []
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))
        # r_circles_filtered, g_circles_filtered, y_circles_filtered = keep_center_most_and_largest(r_circles_filtered, g_circles_filtered, y_circles_filtered, size[1])
            for i in r_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                    continue
                
                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):
                        if (int(i[1])+m) >= size[0] or (int(i[0])+n) >= size[1]:
                            continue
                        h += maskr[int(i[1])+m, int(i[0])+n]
                        s += 1
                if h / s > 50:
                    r_circles_filtered.append(i)
                    # cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                    # cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                    # cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

        if g_circles is not None:
            g_circles = np.uint16(np.around(g_circles))

            for i in g_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (int(i[1])+m) >= size[0] or (int(i[0])+n) >= size[1]:
                            continue
                        h += maskg[int(i[1])+m, int(i[0])+n]
                        s += 1
                if h / s > 100:
                    g_circles_filtered.append(i)
                    # cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                    # cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                    # cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

        if y_circles is not None:
            y_circles = np.uint16(np.around(y_circles))

            for i in y_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (int(i[1])+m) >= size[0] or (int(i[0])+n) >= size[1]:
                            continue
                        h += masky[int(i[1])+m, int(i[0])+n]
                        s += 1
                if h / s > 50:
                    y_circles_filtered.append(i)
                    # cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                    # cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                    # cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

        # Convert the list to a NumPy array
        r_circles_filtered = np.array([r_circles_filtered], dtype=np.float32)  
        y_circles_filtered = np.array([y_circles_filtered], dtype=np.float32) 
        g_circles_filtered = np.array([g_circles_filtered], dtype=np.float32) 
        # print("filtered circles: ", r_circles_filtered, y_circles_filtered, g_circles_filtered)

        r_circles_filtered, g_circles_filtered, y_circles_filtered = keep_largest_and_least_error(r_circles_filtered, g_circles_filtered, y_circles_filtered, size[1])
        # print("filtered circles:", r_circles_filtered, g_circles_filtered, y_circles_filtered)

        def draw_circles(circles, color, text, image, mask):
            for i in circles:
                cv2.circle(image, (int(i[0]), int(i[1])), int(i[2]) + 10, color, 2)
                cv2.circle(mask, (int(i[0]), int(i[1])), int(i[2]) + 30, (255, 255, 255), 2)
                cv2.putText(image, text, (int(i[0]), int(i[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw circles for each color
        if r_circles_filtered is not None and r_circles_filtered.size > 0:
            self.redlightmessage.data = True
            draw_circles(r_circles_filtered[0], (0, 0, 255), 'RED', cimg, maskg)  # Red color in BGR
        else:
            self.redlightmessage.data = False
        self.redlight.publish(self.redlightmessage)
            

        # if y_circles_filtered is not None and y_circles_filtered.size > 0:
            # draw_circles(y_circles_filtered[0], (0, 255, 255), 'YELLOW', cimg, maskg)  # Yellow color in BGR

        if g_circles_filtered is not None and g_circles_filtered.size > 0:
            self.greenlightmessage.data = True
            draw_circles(g_circles_filtered[0], (0, 255, 0), 'GREEN', cimg, maskg)  # Green color in BGR
        else:
            self.greenlightmessage.data = False
        self.greenlight.publish(self.greenlightmessage)


        # print("final g", g_circles_filtered, "final r", r_circles_filtered, "final y", y_circles_filtered)


        cv2.imshow('detected results', cimg)
        # cv2.imwrite(path+'//result//'+file, cimg)
        # cv2.imshow('maskr', maskr)
        # cv2.imshow('maskg', maskg)
        # cv2.imshow('masky', masky)

        cv2.waitKey(2)
        # cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    light_publisher = LightPublisher()
    rclpy.spin(light_publisher)

if __name__ == '__main__':
    main()

