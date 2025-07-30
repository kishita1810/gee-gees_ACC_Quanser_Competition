#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from qcar2_interfaces.msg import MotorCommands
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Pose
from tf2_msgs.msg import TFMessage 
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Float32, Bool

#To live debug lane detector, lidar object detector, or traffic light detector:
#set self.starting variable to False.
#comment self.publish_motor_command(steering_angle, self.throttle) above self.red_flag at the bottom.
#comment self.publish_motor_command(steering_angle, self.throttle) above the statement, "check status of coordinate 2".
#uncomment the cv2.imshow commands corresponding to the lane detection window display.
#uncomment the cv2.imshow commands corresponding to the display of red and green mask filters. 
#lidar detector can be debugged if the 2 motor commands stated above have been commented to auto stop the wheels.

class LaneDetectionPIDController(Node):
    def __init__(self):
        super().__init__('lane_detection_pid')

        self.bridge = CvBridge()
        
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,  # Volatile (sensor data) history
            depth=10,  # Buffer depth for messages
            reliability=QoSReliabilityPolicy.RELIABLE  # Best Effort Reliability
        )
        
        #Initialize subscribers
        self.subscriber_depth = self.create_subscription(Image, 'camera/depth_image', self.calldepth, qos)
        self.subscription = self.create_subscription(Image, '/camera/color_image', self.image_callback, qos)
        self.subscription_light = self.create_subscription(Image, '/camera/color_image', self.light_callback, qos)

        self.subscription_stop = self.create_subscription(Float32, '/stop_depth', self.stop_callback, qos)
        self.subscription_lidar_alert = self.create_subscription(Float32, '/collision_alert', self.alert_callback, qos)
        self.subscription_cone_alert = self.create_subscription(Float32, '/cone_depth', self.alert_callback, qos)

        self.subscription_stopsign_alert = self.create_subscription(Bool, '/stop_flag', self.flagstop_callback, qos)
        self.subscription_yieldsign_alert = self.create_subscription(Bool, '/yield_flag', self.flagyield_callback, qos)
        self.subscription_cone_alert = self.create_subscription(Bool, '/cone_flag', self.flagcone_callback, qos)

        #Initialize publishers
        self.motor_pub = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', qos)

        self.get_led = self.create_publisher(Float32, '/desired_led', qos)
        
        
        #---------------- Declare Global Variables whose boolean values and magnitudes can be altered throughout the main loop execution. ----------------------
        self.initial_speed = True
        
        self.return_to_hub = False
        self.hub_eta = 0
        #self.car_pose = Pose()

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

        # Control parameters
        self.throttle = 0.23
        self.image_center_buffer = deque(maxlen=5)
        self.start_time = time.time()
        self.get_logger().info("Lane Detection with PID Initialized")
        
        #Variables for stop sign
        self.stop_start_time = None
        self.stop_duration = 3.5
        self.stop_cooldown = 5
        self.last_stop_time = 0
        self.is_stopping = False

        self.time_lost = 0
        self.lidar_stop_start_time = 0.0
        
        #Variables for yield sign
        self.yield_start_time = None
        self.yield_duration = 3.5
        self.yield_cooldown = 0.0
        self.last_yield_time = 0.0 
        self.is_yielding = False
        self.done_yielding = False

        self.lidar_delay = 0
      
        self.passed_first_light = False
        self.roundabout = False #true for debugging

        #This variable attempts to recover the time lost in the car stopping at mark zones.
        #It gets updated once the car moves at every mark zone. It resets to 0 if the car
        #just has to pass a zone that was NOT selected as pickup, stop, or dropoff. 
        self.compensator = 0
        
        #This variable enables or disables traffic light detection.
        #The light delay was sometimes used because car noticably detected wrong objects as red light.
        #A delay was created before to activate traffic light detection after turning left from start.
        self.activate = False
        self.light_delay = 0

        self.depth_of_stop = Float32()
        self.required_led = Float32()
        self.required_led.data = 0.0
        
        #The red counters considers for how long car stopped at red light.
        self.green_counter = 0
        self.red_counter = 0
        
        self.green_flag = False
        self.red_flag = False
        self.yellow_flag = False
        self.second_green_flag_checker = False
        
        self.yield_flagged = False
        self.stop_flagged = False
        self.yield_countdown = 0.0
        
        #dynamic pid values
        self.declare_parameter('kp', 0.30)
        self.declare_parameter('kd', 0.13)  #pd before stop
        self.declare_parameter('ki', 0.0)
        
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.ki = self.get_parameter('ki').value
        
        self.depths_yield = None
        self.depths = None
        
        self.starting = True #revert to True if not debugging
    
        self.time_to_turn_left_ = False
        
        self.once = False
        self.once_time = time.time()

        self.get_logger().info('Please answer yes or no:')

        # Get user input 
        answer = input('Is coordinate 2 a pickup? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_2_is_pickup = True #IMPORTANT Note: THIS VARIABLE FOR EACH COORDINATE CONTROLS WHETHER THE CAR SHOULD STOP OR NOT WHEN ARRIVING AT THAT POINT.
        else:
            self.coordinate_2_is_pickup = False

        answer = input('Is coordinate 4 a pickup, dropoff, or mark zone? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_4_is_pickup = True

            answer = input('Is coordinate 4 a dropoff then? y/n ').strip().lower() #Remaining questions for each coordinate control color of LED strip. 
            if answer == 'y':
                self.coordinate_4_is_dropoff = True
            else:
                self.coordinate_4_is_dropoff = False
                answer = input('Is coordinate 4 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_4_is_mark = True
                else:
                    self.coordinate_4_is_mark = False
        else:
            self.coordinate_4_is_pickup = False
        


        answer = input('Is coordinate 14 a pickup, dropoff, or mark zone? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_14_is_pickup = True

            answer = input('Is coordinate 14 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_14_is_dropoff = True
            else:
                self.coordinate_14_is_dropoff = False
                answer = input('Is coordinate 14 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_14_is_mark = True
                else:
                    self.coordinate_14_is_mark = False
        else:
            self.coordinate_14_is_pickup = False
  

        answer = input('Is coordinate 20 a pickup, dropoff, or mark zone? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_20_is_pickup = True

            answer = input('Is coordinate 20 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_20_is_dropoff = True
            else:
                self.coordinate_20_is_dropoff = False
                answer = input('Is coordinate 20 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_20_is_mark = True
                else:
                    self.coordinate_20_is_mark = False
        else:
            self.coordinate_20_is_pickup = False

        answer = input('Is coordinate 22 a pickup? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_22_is_pickup = True

            answer = input('Is coordinate 22 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_22_is_dropoff = True
            else:
                self.coordinate_22_is_dropoff = False
                answer = input('Is coordinate 22 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_22_is_mark = True
                else:
                    self.coordinate_22_is_mark = False
        else:
            self.coordinate_22_is_pickup = False
       
        
        self.counter_red = 0
        self.last_red_time = None
        
        self.coordinate_2 = False
        self.coordinate_2_timer = 0
        self.time_used_pickup= 0
        self.passed_2_coordinate = False

        self.coordinate_4 = False
        self.coordinate_4_timer = 0
        self.passed_4_coordinate = False

        self.coordinate_14 = False
        self.coordinate_14_timer = 0
        self.passed_14_coordinate = False

        self.coordinate_20 = False
        self.coordinate_20_timer = 0
        self.passed_20_coordinate = False

        self.coordinate_22 = False
        self.coordinate_22_timer = 0
        self.passed_22_coordinate = False

        self.coordinate_6 = False
        self.coordinate_6_timer = 0
        self.passed_6_coordinate = False

        self.coordinate_8 = False
        self.coordinate_8_timer = 0
        self.passed_8_coordinate = False

        self.lidar_alert = Float32()
        self.lidar_stopped = False

        self.stopseen = Bool()
        self.yieldseen = Bool()
        self.coneseen = Bool()

        
    def flagcone_callback(self, conemsg):
        self.coneseen = conemsg

    def flagstop_callback(self, stopmsg):
        self.stopseen = stopmsg

    def flagyield_callback(self, yieldmsg):
        self.yieldseen = yieldmsg
    
    def calldepth(self, depth_message):
        self.depths = Image()
        self.depths = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding='16UC1')

    def alert_callback(self, alert_message):
        self.lidar_alert.data = alert_message.data
    
    def stop_callback(self, stopdepth):
        if stopdepth is not None:
            self.depth_of_stop.data = stopdepth.data
            #print(self.depth_of_stop.data)
        
    def light_callback(self, lightmsg):
        frame_received = self.bridge.imgmsg_to_cv2(lightmsg, desired_encoding='bgr8')
        if frame_received is None:
            return
            
        #Look for traffic light.
        #frameforlight = frame_received[0:400, 485:730]
        #frameforlight = frame_received[0:400, 250:1100]
        frameforlight = frame_received[0:400, 585:1030]

        #Convert to HSV
        hsv_frame = cv2.cvtColor(frameforlight, cv2.COLOR_BGR2HSV)
        
        #Create masks to avoid random or noisy detections in the sky and ground.   
        red_mask_filter = cv2.inRange(hsv_frame, (0, 100, 100), (10, 255, 255)) + cv2.inRange(hsv_frame, (160, 100, 100), (180, 255, 255))
        yellow_mask_filter = cv2.inRange(hsv_frame, (20, 100, 100), (30, 255, 255))
        green_mask_filter = cv2.inRange(hsv_frame, (40, 100, 100), (70, 255, 255))

        #Combine all masks and mask original image.
        mask = red_mask_filter | yellow_mask_filter | green_mask_filter
        masked_frame = cv2.bitwise_and(frameforlight, frameforlight, mask=mask)
        heightl, widthl = masked_frame.shape[:2]
        
        lower_red1 = np.array([1, 15, 230]) #correct s it was 180 not 15
        upper_red1 = np.array([40, 255, 255])
        lower_red2 = np.array([160, 15, 230]) #correct s
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([21, 120, 245])
        upper_yellow = np.array([35, 190, 255])

        # --- GREEN ---
        lower_green = np.array([89, 2, 254]) #simulated 's' was 150
        upper_green = np.array([91, 4, 255])
        
        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # YELLOW mask
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # GREEN mask
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        #cv2.imshow("Red Mask", red_mask) 
        #cv2.imshow("Green Mask", green_mask) 
        #cv2.waitKey(1)

        if self.activate == True:
            red_pixels = cv2.countNonZero(red_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)
            green_pixels = cv2.countNonZero(green_mask)
            
            if red_pixels >= 80: #beware of flickering red light.
                self.get_logger().info("Red Light is ON! Stopping car!")
                self.red_flag = True
                self.green_flag = False
                self.yellow_flag = False

            elif yellow_pixels > 1000:
                self.get_logger().info("Yellow Light is ON! Slowing down!")
                self.yellow_flag = True
                self.red_flag = False
                self.green_flag = False

            elif green_pixels >= 3:
                self.get_logger().info("Green Light is ON! Accelerating car!")
                if self.passed_first_light == False:
                    self.green_flag = True
                    self.red_flag = False
                    self.yellow_flag = False
                elif self.passed_first_light == True:
                    self.green_flag = True
                    self.red_flag = False
                                               
    
    def detect_yellow_lane_and_curvature(self, image):
        #image = self.bridge.imgmsg_to_cv2(ima, desired_encoding='bgr8')
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Yellow color mask in HSV
        lower_yellow = np.array([18, 94, 140])
        upper_yellow = np.array([48, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        
        coords = np.column_stack(np.where(mask > 0))
        if coords.shape[0] < 100:
            return image, None, False, None

        ys = coords[:, 0]
        xs = coords[:, 1]
        
        roi_bottom = int(image.shape[0] * 0.7)
        maskpoints = ys > roi_bottom
        
        ys_filtered = ys[maskpoints]
        xs_filtered = xs[maskpoints]

        poly_coeffs = np.polyfit(ys, xs, 2)

        y_vals = np.linspace(np.min(ys), np.max(ys), num=100)
        x_vals = np.polyval(poly_coeffs, y_vals)
        
        x_center = int(np.mean(x_vals))

        for x, y in zip(x_vals.astype(int), y_vals.astype(int)):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        A = poly_coeffs[0]
        B = poly_coeffs[1]
        y_eval = np.max(ys)
        curvature = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.abs(2 * A)
        
        aa, bb, cc = poly_coeffs
        y_near = image.shape[0] - 20  # Pick a point near the bottom of the image
        x_near = (aa * (y_near ** 2) + bb * y_near + cc) #+ 120 # was 100 before
        
        image_center_x = image.shape[1] // 2  # center of the image
        yellow_on_left = x_near < image_center_x
        
        cv2.circle(image, (int(x_near), int(y_near)), 5, (0, 0, 255), -1)
        cv2.line(image, (image_center_x, 0), (image_center_x, image.shape[0]), (255, 255, 0), 1)

        #cv2.putText(image, f"Yellow Lane Radius: {int(curvature_radius)} px", (50, 50),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        return image, curvature, yellow_on_left, x_center
        
        
    def time_to_yield(self, yield_x, yield_y):
        if self.depths_yield is None:
            return False

        depth_from_yield_sign = self.depths_yield[yield_y, yield_x] / 1000.0  # convert mm to cm
        if depth_from_yield_sign <= 3.8 and depth_from_yield_sign >= 0 and not self.is_yielding and (time.time() - self.last_yield_time) > self.yield_cooldown:
            if self.done_yielding == False:
                self.get_logger().info("Yield Sign Detected! No cars approaching!")
            self.yield_start_time = time.time()
            self.done_yielding = True
            self.is_yielding = True
            return True
            
        return False
        
        #self.activate = True

    def handle_coordinate_2(self):
        self.throttle = 0.0
        passenger_coordinate_one = time.time()
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        go = time.time()
        time.sleep(3) 
        end = time.time()
        self.time_used_pickup = end - go 
        self.throttle = 0.4
        self.publish_motor_command(-0.15, self.throttle)
        self.coordinate_2 = False
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)

    def handle_coordinate_4(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_4_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_4_is_mark == True:
            self.required_led.data = 4.0
        else: 
            self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        self.throttle = 0.0
        self.publish_motor_command(0.0, self.throttle)
        time.sleep(3)
        self.throttle = 0.4
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)
        self.coordinate_4 = False

    def handle_coordinate_14(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_14_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_14_is_mark == True:
            self.required_led.data = 4.0
        else: 
            self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        self.throttle = 0.0
        self.publish_motor_command(0.0, self.throttle)
        go = time.time()
        time.sleep(3)
        end=time.time()
        self.throttle = 0.4+0.1
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)
        self.coordinate_14 = False
    
    def handle_coordinate_20(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_20_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_20_is_mark == True:
            self.required_led.data = 4.0
        else: 
            self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        self.throttle = 0.0
        self.publish_motor_command(0.0, self.throttle)
        time.sleep(3)
        self.throttle = 0.4
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)
        self.coordinate_20 = False

    def handle_coordinate_22(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_22_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_22_is_mark == True:
            self.required_led.data = 4.0
        else: 
            self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        self.throttle = 0.0
        self.publish_motor_command(0.0, self.throttle)
        time.sleep(3)
        self.throttle = 0.4
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)
        self.coordinate_22 = False

    def handle_coordinate_6(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_6_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_6_is_mark == True:
            self.required_led.data = 4.0
        else: 
            self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        self.throttle = 0.0
        self.publish_motor_command(0.0, self.throttle)
        time.sleep(3)
        self.throttle = 0.4
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)
        self.coordinate_6 = False

    def handle_coordinate_8(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_8_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_8_is_mark == True:
            self.required_led.data = 4.0
        else: 
            self.required_led.data = 2.0
        self.get_led.publish(self.required_led)
        self.throttle = 0.0
        self.publish_motor_command(0.0, self.throttle)
        time.sleep(3)
        self.throttle = 0.4
        self.required_led.data = 1.0
        self.get_led.publish(self.required_led)
        self.coordinate_8 = False
            
	#Main Function is below. All previous functions above that are not subscribers can be called from the one below.
	
    def image_callback(self, msg):
        try:
            steering_angle = 0.28# +0.2 #Initialize steering angle in case lane detection fails to provide one. 0.28
            
            #Capture frame from camera. 
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            #Quality of camera is not perfect. So frames can't be streamed continuously. In that case, return back.
            if frame is None or frame.size == 0:
                #self.get_logger().warn("Received an empty frame!")
                return
            
            # -------------------------------------- Lane Detection using CV2 libraries and features, such as slope and length. ----------------------
            height, width, _ = frame.shape
            roi = frame[int(height / 2):, :]
            
            yellow_viz, radius, left_flag, xnear = self.detect_yellow_lane_and_curvature(roi.copy())
            if xnear is not None:
                cv2.circle(yellow_viz, (xnear, yellow_viz.shape[0] // 2), 5, (255, 0, 0), -1)
            #cv2.imshow("Yellow Lane Detection", yellow_viz)
            #cv2.waitKey(3)

            # Preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
        
            _, asphalt_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            hsv_white = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_asphalt = np.array([0, 0, 0])
            upper_asphalt = np.array([180, 80, 100])
            asphalt_mask = cv2.inRange(hsv_white, lower_asphalt, upper_asphalt)
            
            lower_white = np.array([0, 0, 180])      # Allow slightly darker whites
            upper_white = np.array([180, 50, 255])   # Allow more saturation

            white_mask = cv2.inRange(hsv_white, lower_white, upper_white)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_asphalt = cv2.dilate(asphalt_mask, kernel, iterations=2)
            
            white_on_asphalt = cv2.bitwise_and(white_mask, dilated_asphalt)
            
            hsvy = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_yelloww = np.array([18, 94, 140])
            upper_yelloww = np.array([50, 255, 255])
            masky = cv2.inRange(hsvy, lower_yelloww, upper_yelloww)
            
            yellow_edges = cv2.Canny(masky, 50, 150)

            # Hough Line Detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=120)
           
            #yellowlines = cv2.HoughLinesP(yellow_viz, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
            white_lines = cv2.HoughLinesP(white_on_asphalt, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100) #100 before
            
            frame_width = edges.shape[1]
            min_x_frame = int(frame_width * 0.30)  # 30% from left
            max_x_frame = int(frame_width * 0.91)  # 100% from left

            left_lines = []
            right_lines = []
            
            def average_x(lines):
                if not lines:
                    return None
                xs = [x1 for x1, _, x2, _ in lines] + [x2 for _, _, x2, _ in lines]
                return int(np.mean(xs))
            
            if radius is None: #If detection of yellow lanes by curvature method doesn't exist.
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if None in [x1, y1, x2, y2]:
                            continue
                        if not (min_x_frame < x1 < max_x_frame and min_x_frame < x2 < max_x_frame):
                            continue
                        if x2 == x1:
                            continue
                        slope = (y2 - y1) / (x2 - x1)
                        if abs(slope) < 0.20:  # filter almost horizontal lines
                            continue
                        if slope < 0 and x1 < int(frame_width * 0.5):
                            left_lines.append((x1, y1, x2, y2))
                
                        #else:
                        #    right_lines.append((x1, y1, x2, y2)) #right lanes can't be yellow
            
            if white_lines is not None: #find white lanes
                for line in white_lines:
                    x1w, y1w, x2w, y2w = line[0]
                    if None in [x1w, y1w, x2w, y2w]:
                        continue
                    line_length = np.sqrt((x2w - x1w)**2 + (y2w - y1w)**2)
                    angle = np.degrees(np.arctan2(y2w - y1w, x2w - x1w))
                        #self.get_logger().info(f"angle ={angle}")
                    if not (min_x_frame < x1w < max_x_frame and min_x_frame < x2w < max_x_frame):
                        continue
                    if x2w == x1w:
                        continue
                    if line_length < 100:
                        continue
                    slope = (y2w - y1w) / (x2w - x1w)
                    if abs(slope) < 0.2:
                        continue
                        #if x1w < int(frame_width * 0.5) and x2w < int(frame_width * 0.5):
                    #if slope < 0:
                        #left_lines.append((x1w, y1w, x2w, y2w))
                        #if x1w > int(frame_width * 0.5) and x2w > int(frame_width * 0.5):
                    if slope > 0:
                        right_lines.append((x1w, y1w, x2w, y2w))
            
            if radius is None:
                left_x = average_x(left_lines)
                right_x = average_x(right_lines)

                if left_x is not None and right_x is not None:
                    lane_center = (left_x + right_x) // 2
                elif left_x is not None:
                    lane_center = left_x + 150 
                elif right_x is not None:
                    lane_center = right_x - 150 
                    #if self.passed_2_coordinate == True and self.passed_4_coordinate == False:
                    #    lane_center = lane_center - 150
                else:
                    #self.get_logger().warn("No lane lines detected.")
                    lane_center = None
                
                
                #doesn't work
                #if self.passed_20_coordinate == True and lane_center is not None:
                #    lane_center = lane_center - 200 
                 
            elif radius is not None:
                right_x = average_x(right_lines)
                left_x = xnear if xnear is not None else None #get the x pixel coordinate from curvature function.
                if left_x is not None and right_x is not None:
                    lane_center = (left_x + right_x) // 2
                    if self.once == True:
                        lane_center = lane_center + 250
                elif left_x is not None:
                    if self.passed_2_coordinate == False and self.roundabout == False:
                        lane_center = left_x + 150 + 250 #- 50  #offset was 400
                    else:
                        lane_center = left_x + 150 + 550 
                    if self.roundabout == True and self.passed_4_coordinate == True: 
                        lane_center =  left_x + 150 + 300
                elif right_x is not None and self.is_stopping == True:
                    lane_center = right_x - 150 
                else:
                    #self.get_logger().warn("No lane lines detected.")
                    lane_center = None

                if self.passed_14_coordinate == True: #and self.passed_20_coordinate == False:
                    lane_center = lane_center + 150 + 50 #150 to 200 offset should work.
                    print(time.time() - self.once_time)
                    if self.once == False and time.time() - self.once_time >= (8.8 + self.compensator - 0.2):
                        turn = time.time()
                        while (time.time() - turn) <= 4.0:
                            steering_angle = 0.2
                            self.publish_motor_command(steering_angle, self.throttle)
                        lane_center = lane_center - 200
                        self.once = True
                        self.passed_14_coordinate = False
                        print("check")
                    
                #lane_center = lane_center - 500
                #self.get_logger().info(f"lane center={lane_center}")
               
            vis_image = roi.copy()
            
            if radius is None:
                if left_lines:
                    for x1, y1, x2, y2 in left_lines:
                        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if right_lines:
                for x1, y1, x2, y2 in right_lines:
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if lane_center is not None:
                    #cv2.line(vis_image, (lane_center, 0), (lane_center, vis_image.shape[0]), (255, 0, 0), 2)
                cv2.circle(vis_image, (lane_center, vis_image.shape[0] // 2), 5, (255, 0, 0), -1)
                #cv2.waitKey(1)
                
                #cv2.imshow("Lane Detection", vis_image)
                #cv2.waitKey(3)
            # Smooth lane center using a buffer
                self.image_center_buffer.append(lane_center)
                if len(self.image_center_buffer) < self.image_center_buffer.maxlen:
                    self.get_logger().info("Warming up lane center buffer...")
                    return
                smoothed_center = int(np.mean(self.image_center_buffer))

                # PID control
                image_center = width // 2
                error = smoothed_center - image_center
                abs_error = abs(error)
                # --------------------------------- Lane Detection Ends with a rough estimate of lane center -------------------------
                

                # ---------------------------------- Developing a PI Controller ----------------------------------
                # Live PID parameters
                kp = self.get_parameter('kp').get_parameter_value().double_value
                kd = self.get_parameter('kd').get_parameter_value().double_value
                ki = self.get_parameter('ki').get_parameter_value().double_value


                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time
                #self.get_logger().info(f"error={error}")

                self.integral += error * dt
                derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                    
                self.integral = max(min(self.integral, 100), -100)
                derivative = max(min(derivative, 100), -100)
                    
                self.prev_error = error
                
                #self.get_logger().info(f"Using special PD: kp={self.kp}, kd={self.kd}")
                steering_angle = -(self.kp * error + self.ki * self.integral + self.kd * derivative) / (width / 2)
                steering_angle = max(min(steering_angle, 1.0), -1.0)
                #self.get_logger().info(f"Throttle: {self.throttle:.2f}, Error: {error:.2f}, Steering: {steering_angle:.2f}, kp={self.kp:.2f}, kd={self.kd:.2f}")
                if time.time() - self.start_time < 2.0:
                    steering_angle = 0.0
                    
            #print(self.depth_of_stop.data)
            """
            if self.depth_of_stop.data is not None and self.is_stopping == False:
                if self.depth_of_stop.data <= 8.9 and self.depth_of_stop.data >= 0.0 and (time.time() - self.last_stop_time) > self.stop_cooldown: #2 < x < 7.7  max 1.2 physical
                    self.get_logger().info("Stop sign detected within range. Stopping.")
                    self.throttle = 0.0
                    steering_angle = 0.0
                    self.required_led.data = 4.0
                    self.get_led.publish(self.required_led)
                    self.stop_start_time = time.time()
                    self.is_stopping = True
                    self.publish_motor_command(steering_angle, self.throttle)
                    #self.time_lost_stop_sign += time.time() - self.stop_start_time
                    time.sleep(3)
                    self.last_stop_time = time.time()
                    self.time_lost += (time.time() - self.stop_start_time)
                    self.required_led.data = 1.0
                    self.get_led.publish(self.required_led)
                    self.is_stopping = False
                    self.stop_cooldown = 15
            """
            #Red Light is ON. Assign 0 velocity throughout the execution.
            if self.red_flag == True: 
                if self.last_red_time == None:
                    self.last_red_time = time.time()
                self.throttle = 0.0
                steering_angle = 0.0
                self.publish_motor_command(steering_angle, self.throttle)
                self.required_led.data = 4.0
                self.get_led.publish(self.required_led)
                
            #If red light is off:
            else:
                #If red light was turned on before. Check for how long car stopped.
                if self.last_red_time is not None:
                    self.required_led.data = 1.0
                    self.get_led.publish(self.required_led)
                    #self.red_counter += time.time() - self.last_red_time
                    self.time_lost += (time.time() - self.last_red_time)
                    self.last_red_time = None

                if self.lidar_alert.data == 1.0 and self.lidar_delay >= 5:
                    if not self.lidar_stopped:
                        self.get_logger().info("LIDAR: Obstacle detected. Stopping.")
                        self.throttle = 0.2
                        steering_angle = 0.1
                        self.lidar_stop_start_time = time.time()
                        self.lidar_stopped = True 
                        self.required_led.data = 4.0
                        self.get_led.publish(self.required_led)
                    self.publish_motor_command(steering_angle, self.throttle)
                    return
                
                if self.lidar_stopped == True and self.lidar_alert.data == 0.0:
                    #self.time_lost_lidar = time.time() - self.lidar_stop_start_time  
                    self.time_lost += (time.time() - self.lidar_stop_start_time) 
                    self.lidar_stopped = False
                    self.throttle = 0.4
                    self.required_led.data = 1.0
                    self.get_led.publish(self.required_led)

                #If car did not stop at light because it was never red.     
                else:
                

                    #sequential driving

                    
                    if self.starting == True:
                        self.required_led.data = 5.0
                        self.get_led.publish(self.required_led)
                        time.sleep(8) #set it to 7
                        self.required_led.data = 1.0
                        self.get_led.publish(self.required_led)
                        self.lidar_delay = time.time()
                    	
                    	#Initialize timer and variable to True for turning car left.
                        #self.time_to_turn_left = True #set it back to true
                    	
                    	#These three variables below are set to wrong values deliberately to test the car's throttle at red light
                    	#without having to move it to the starting point. The last publish motor command should be commented out.
                     	#Also, self.time_to_turn_left should be false and self.throttle should be 0.0 for this purpose.
                    	
                    	#self.drive_straight = True
                    	#self.activate = True
                    	#self.straight_drive_countdown = time.time()
                        self.coordinate_2 = True
                        self.coordinate_2_timer = time.time() 
                        self.starting = False
                        
                    # Countdown for turning left set to 3 seconds once car drives.
                    #if self.time_to_turn_left == True:
                    #    if (time.time() - self.left_turn_countdown >= 3.8): #6 then 4
                    #        self.time_to_turn_left = False
                    #        self.turn_left() #Call the function to turn left and start process to drive straight.
                    
                    
                   
                    self.publish_motor_command(steering_angle, self.throttle)
                    
                        
                    #Check status of coordinate 2 indicated by user. 
                    if self.coordinate_2 == True and (time.time() - self.coordinate_2_timer) >= (5 + 2.5 + self.time_lost):  #0.5
                        if self.coordinate_2_is_pickup == True:
                            self.handle_coordinate_2() #if user selected it as pickup, go to the defined stop function.
                            self.compensator = 0.4
                        else:
                            self.coordinate_2 = False
                            self.compensator = 0 #reset time compensator
                        self.time_lost = 0.0
                        if self.passed_2_coordinate == False:
                            self.coordinate_4_timer = time.time() #start countdown to coordinate 13.
                            self.passed_2_coordinate = True
                            self.coordinate_4 = True
                        self.time_lost = 0
                            
                    #Check status of coordinate 4 indicated by user.
                    if self.coordinate_4 == True and (time.time() - self.coordinate_4_timer) >= (3.5+1.7 + self.compensator + 0.15 + self.time_lost):
                        if self.coordinate_4_is_pickup == True:
                            self.handle_coordinate_4()
                            self.compensator = 0.36
                        else:
                            self.coordinate_4 = False
                            self.compensator = 0
                        self.time_lost = 0.0    
                        duration = time.time()
                        while (time.time() - duration <= 1.8 + self.compensator): #drive straight
                            self.throttle = 0.7 
                            steering_angle = -0.09 
                            self.publish_motor_command(steering_angle, self.throttle)
                        self.passed_4_coordinate= True
                        #self.passed_2_coordinate = False
                        self.coordinate_14_timer = time.time() #start countdown to coordinate 19.
                        self.coordinate_14 = True
                        self.roundabout = True
                        self.time_lost = 0
                    
                    #Check status of coordinate 19 indicated by user.
                    if self.coordinate_14 == True and ((time.time() - self.coordinate_14_timer) >= 6.9 - 0.8 + self.time_lost): #10 or 8
                        if self.coordinate_14_is_pickup == True: 
                            self.handle_coordinate_14() #Call this stop function if marked as zone. 
                            self.coordinate_14 = False
                            self.compensator = 0.35
                        else:
                            self.coordinate_14 = False
                            self.compensator = 0
                        self.get_logger().info("Roundabout")
                        print(lane_center)
                        q1_duration = 3.2
                        q2_duration = 2.9
                        q3_duration = 2.5
                        self.throttle = 0.5#0.37
                        prev_time = time.time()
                        steering_angle = -0.18
                        while (time.time() - prev_time) <= (2.0 + 0.6 + self.compensator - 0.3):
                            self.publish_motor_command(steering_angle, self.throttle )
                        prev_time_q2 = time.time()
                        steering_angle = 0.42
                        self.throttle = 0.37
                        while (time.time() - prev_time_q2) <= q2_duration:
                            self.publish_motor_command(steering_angle, self.throttle )
                        self.get_logger().info("q2")
                        prev_time_q3 = time.time()
                        steering_angle = -0.15
                        
                        #while (time.time() - duration_right <= 1):
                        #    self.throttle = 0.3
                         #   steering_angle = -0.6
                          #  self.publish_motor_command(steering_angle, self.throttle)
                        self.passed_14_coordinate = True
                        self.once_time = time.time()
                        self.once = False
                        self.coordinate_20_timer = time.time() #start countdown to coordinate 17.
                        self.coordinate_20 = True
                        self.compensator = 3
                        self.time_lost = 0

                    if self.coordinate_20 == True and ((time.time() - self.coordinate_20_timer) >= (2+self.compensator)):
                        if self.coordinate_20_is_pickup == True:
                            self.handle_coordinate_20() #Call function to stop if marked as a zone by user.
                            #self.compensator = 2
                        else:
                            self.coordinate_20 = False
                            self.compensator = 0
                        
                        self.passed_20_coordinate = True
                        #self.passed_14_coordinate = False
                        self.coordinate_22_timer = time.time() #start countdown to coordinate 15. 
                        self.coordinate_22 = True
                        self.time_lost = 0

                    if self.coordinate_22 == True and (time.time() - self.coordinate_22_timer) >= 9:
                        if self.coordinate_22_is_pickup == True: #Call to stop if coordinate marked as a zone. 
                            self.handle_coordinate_22()
                        else:
                            self.coordinate_22 == False
                        
                        self.passed_22_coordinate = True

                        self.coordinate_6 = True 
                        self.coordinate_6_timer = time.time()

                    if self.coordinate_6 == True and (time.time() - self.coordinate_6_timer) >= 4.5:
                        
                        self.throttle = 0.75#0.37
                        ret1 = time.time()
                        steering_angle = -0.1
                        while (time.time() - ret1) <= (2.5):
                            self.publish_motor_command(steering_angle, self.throttle )

                        ret2 = time.time()
                        steering_angle = 0.1
                        self.throttle = 0.4
                        while (time.time() - ret2) <= 2:
                            self.publish_motor_command(steering_angle, self.throttle ) 
                        
                        ret3 = time.time()
                        steering_angle = -0.1
                        self.throttle = 0.4
                        while (time.time() - ret3) <= 1.5:
                            self.publish_motor_command(steering_angle, self.throttle ) 

                        self.throttle = 0.0
                        self.steering_angle = -0.0
                        j = 1
                        self.required_led.data = 5.0
                        self.get_led.publish(self.required_led)
                        while j == 1:
                            self.publish_motor_command(self.steering_angle, self.throttle)

                                #self.passed_first_light = True
                               
                            #if self.red_counter != 0:
                            #self.activate = False 
                        
                        #if (time.time() - self.turn_left_again_countdown) >= 6: #back to 3
                        #    self.turn_left_again = False
                        #    self.activate = False
                               
                    else: 
                        self.throttle = 0.40 #0.23
                    if self.is_stopping == True:
                        steering_angle = 0.0
                        self.throttle = 0.0  

            #Function that controls speed most of time.    	   
            self.publish_motor_command(steering_angle, self.throttle)
            self.red_flag = False
                #self.get_logger().info(f"Throttle: {self.throttle:.2f}, Steering: {steering_angle:.2f}")
 
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {e}")

    def publish_motor_command(self, throttle, steering_angle):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [throttle, steering_angle]
        self.motor_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
