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
        self.subscription_lidar_alert = self.create_subscription(Float32, '/collision_alert', self.alert_callback, qos)

        self.subscription_stop = self.create_subscription(Float32, '/stop_depth', self.stop_callback, qos)
        #self.subscription_cone_depth = self.create_subscription(Float32, '/cone_depth', self.cone_callback, qos)
        #self.subscription_yieldsign_depth = self.create_subscription(Float32, '/yield_depth', self.yield_callback, qos)

        self.subscription_stopsign_alert = self.create_subscription(Bool, '/stop_flag', self.flagstop_callback, qos)
        #self.subscription_yieldsign_alert = self.create_subscription(Bool, '/yield_flag', self.flagyield_callback, qos)
        #self.subscription_cone_alert = self.create_subscription(Bool, '/cone_flag', self.flagcone_callback, qos)  

        #Initialize publishers
        self.motor_pub = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', qos)

        self.get_led = self.create_publisher(Float32, '/desired_led', qos)
        
        self.redlight = self.create_subscription(Bool, '/traffic_light_red', self.red_call, qos)
        #---------------- Declare Global Variables whose boolean values and magnitudes can be altered throughout the main loop execution. ----------------------
        self.initial_speed = True
        
        self.return_to_hub = False
        self.hub_eta = 0
        #self.car_pose = Pose()

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

        self.lidar_delay = 0

        # Control parameters
        self.throttle = 0.23
        self.image_center_buffer = deque(maxlen=5)
        self.start_time = time.time()
        self.get_logger().info("Lane Detection with PID Initialized")
        
        #Variables for stop sign
        self.stop_start_time = None
        self.stop_duration = 3.5
        self.stop_cooldown = 10
        self.last_stop_time = 0
        self.is_stopping = False
      
        self.passed_first_light = False
        self.lidar_stop_start_time = 0.0
        self.lidar_alert = Float32()
        self.lidar_stopped = False
        
        #This variable enables or disables traffic light detection.
        #The light delay was sometimes used because car noticably detected wrong objects as red light.
        #A delay was created before to activate traffic light detection after turning left from start.
        self.activate = False
        self.light_delay = 0

        self.required_led = Float32()
        self.required_led.data = 0.0
        
        #The red counters considers for how long car stopped at red light.
        self.green_counter = 0
        self.red_counter = 0
        
        self.green_flag = False
        self.red_flag = False
        self.yellow_flag = False
        
        self.yield_flagged = False
        self.stop_flagged = False
        self.yield_countdown = 0.0

        self.time_lost = 0
        self.second_lane_adjuster = 0
        
        #dynamic pid values
        self.declare_parameter('kp', 0.30)
        self.declare_parameter('kd', 0.13)  #pd before stop
        self.declare_parameter('ki', 0.0)
        
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.ki = self.get_parameter('ki').value
        
        self.depths_yield = None
        self.depths = None
        
        self.starting = True #revert to True

        self.lane_adjuster = 0
        
        self.compensator = 0
        self.counter_red = 0
        self.last_red_time = None
        
        self.coordinate_2 = False
        self.coordinate_2_timer = 0
        self.time_used_pickup= 0
        self.passed_2_coordinate = False

        self.coordinate_4 = False
        self.coordinate_4_timer = 0
        self.passed_4_coordinate = False

        self.coordinate_6 = False
        self.coordinate_6_timer = 0
        self.passed_6_coordinate = False

        self.coordinate_8 = False
        self.coordinate_8_timer = 0
        self.passed_8_coordinate = False

        self.coordinate_23 = False
        self.coordinate_23_timer = 0
        self.passed_23_coordinate = False

        self.coordinate_25 = False
        self.coordinate_25_timer = 0
        self.passed_25_coordinate = False

        self.coordinate_24 = False
        self.coordinate_24_timer = 0
        self.passed_24_coordinate = False

        self.new_offset = False

        self.yield_flag = Bool()
        self.stop_flag = Bool()
        self.cone_flag = Bool()

        self.depth_of_stop = Float32()
        self.depth_of_yield = Float32()
        self.depth_of_cone = Float32()

        self.red_light = Bool()

        self.cone_stopped = False
        self.yield_stopped = False
        self.stop_stopped = False

        self.cone_stop_time = 0
        self.yield_stop_time = 0

        self.coordinate_return = False
        self.coordinate_return_timer = 0

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
        


        answer = input('Is coordinate 6 a pickup, dropoff, or mark zone? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_6_is_pickup = True

            answer = input('Is coordinate 6 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_6_is_dropoff = True
            else:
                self.coordinate_6_is_dropoff = False
                answer = input('Is coordinate 6 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_6_is_mark = True
                else:
                    self.coordinate_6_is_mark = False
        else:
            self.coordinate_6_is_pickup = False

        answer = input('Is coordinate 8 a pickup? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_8_is_pickup = True

            answer = input('Is coordinate 8 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_8_is_dropoff = True
            else:
                self.coordinate_8_is_dropoff = False
                answer = input('Is coordinate 8 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_8_is_mark = True
                else:
                    self.coordinate_8_is_mark = False
        else:
            self.coordinate_8_is_pickup = False

        answer = input('Is coordinate 23 a pickup? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_23_is_pickup = True

            answer = input('Is coordinate 23 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_23_is_dropoff = True
            else:
                self.coordinate_23_is_dropoff = False
                answer = input('Is coordinate 23 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_23_is_mark = True
                else:
                    self.coordinate_23_is_mark = False
        else:
            self.coordinate_23_is_pickup = False

        answer = input('Is coordinate 25 a pickup? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_25_is_pickup = True

            answer = input('Is coordinate 25 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_25_is_dropoff = True
            else:
                self.coordinate_25_is_dropoff = False
                answer = input('Is coordinate 25 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_25_is_mark = True
                else:
                    self.coordinate_25_is_mark = False
        else:
            self.coordinate_25_is_pickup = False

        answer = input('Is coordinate 24 a pickup? y/n ').strip().lower()
        if answer == 'y':
            self.coordinate_24_is_pickup = True

            answer = input('Is coordinate 24 a dropoff then? y/n ').strip().lower()
            if answer == 'y':
                self.coordinate_24_is_dropoff = True
            else:
                self.coordinate_24_is_dropoff = False
                answer = input('Is coordinate 24 a mark zone then? y/n ').strip().lower()
                if answer == 'y':
                    self.coordinate_24_is_mark = True
                else:
                    self.coordinate_24_is_mark = False
        else:
            self.coordinate_24_is_pickup = False

        
    def red_call(self, msgg):
        self.red_light = msgg
    
    def flagstop_callback(self, msg):
        self.stop_flag = msg

    def flagyield_callback(self, msg):
        self.yield_flag = msg

    def flagcone_callback(self, msg):
        self.cone_flag = msg
    
    def calldepth(self, depth_message):
        self.depths = Image()
        self.depths = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding='16UC1')
    
    def calldepth_yield(self, depthmessage):
        self.depths_yield = Image()
        self.depths_yield = self.bridge.imgmsg_to_cv2(depthmessage, desired_encoding='16UC1')

    def alert_callback(self, alert_message):
        self.lidar_alert.data = alert_message.data
    
    def stop_callback(self, stopdepth):
        if stopdepth is not None:
            self.depth_of_stop.data = stopdepth.data
            #print(self.depth_of_stop.data)

    def yield_callback(self, yielddepth):
        if yielddepth is not None:
            self.depth_of_yield.data = yielddepth.data
    
    def cone_callback(self, conedepth):
        if conedepth is not None:
            self.depth_of_cone.data = conedepth.data
        
    def light_callback(self, lightmsg):
        frame_received = self.bridge.imgmsg_to_cv2(lightmsg, desired_encoding='bgr8')
        if frame_received is None:
            return
            
        #Look for traffic light.
        #frameforlight = frame_received[0:400, 485:730]
        #frameforlight = frame_received[0:400, 250:1100]
        #frameforlight = frame_received[0:400, 585:1030]
        frameforlight = frame_received[0:400, 985:1700]
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
            
            if red_pixels >= 380 and red_pixels <= 1000: #10:#if red_pixels >= 80: #beware of flickering red light.
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
        self.throttle = 0.7
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

    def handle_coordinate_23(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_23_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_23_is_mark == True:
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
        self.coordinate_23 = False

    def handle_coordinate_25(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_25_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_25_is_mark == True:
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
        self.coordinate_25 = False

    def handle_coordinate_24(self):
        self.throttle = 0.0
        steering_angle = 0.0
        self.publish_motor_command(steering_angle, self.throttle)
        if self.coordinate_24_is_dropoff == True:
            self.required_led.data = 3.0
        elif self.coordinate_24_is_mark == True:
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
        self.coordinate_24 = False
    
            
	#Main Function is below. All previous functions above that are not subscribers can be called from the one below.
	
    def image_callback(self, msg):
        try:
            steering_angle = 0.28 #Initialize steering angle in case lane detection fails to provide one. 0.28
            
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
            #cv2.waitKey(1)

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
                else:
                    #self.get_logger().warn("No lane lines detected.")
                    lane_center = None
                
                        
            elif radius is not None:
                right_x = average_x(right_lines)
                left_x = xnear if xnear is not None else None #get the x pixel coordinate from curvature function.
                
                if left_x is not None and right_x is not None:
                    lane_center = (left_x + right_x) // 2
                    #print("a")
                elif left_x is not None:
                    if self.passed_2_coordinate == False:
                        lane_center = left_x + 150 + 300 #300 works until first light
                    elif self.passed_6_coordinate == True:
                        lane_center = left_x + 550 + 50
                    else:
                        lane_center = left_x + 150 + 600 + 10
                    #print("b")
                elif right_x is not None and self.is_stopping == True:
                    lane_center = right_x - 150 
                else:
                    #self.get_logger().warn("No lane lines detected.")
                    lane_center = None
                    #print("e")
                #if self.coordinate_5 == True and lane_center is not None:
                #    lane_center = lane_center + 150
                
                #Below 5 commands not working
                #if self.coordinate_19 == True and lane_center is not None:
                #    if time.time() - self.lane_adjuster >= 2:
                #        lane_center = lane_center + 200
                #    else:
                #        lane_center = lane_center + 650


                if self.passed_8_coordinate == True and self.passed_23_coordinate == False:
                    if (time.time()-self.second_lane_adjuster>=2.2 and time.time()-self.second_lane_adjuster<=3):
                        lane_center = lane_center + 90
                #self.get_logger().info(f"lane center={lane_center}")
               
            vis_image = roi.copy()
            
            if radius is None:
                if left_lines:
                    for x1, y1, x2, y2 in left_lines:
                        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if right_lines:
                for x1, y1, x2, y2 in right_lines:
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            #print(lane_center)
            if lane_center is not None:
                if self.passed_6_coordinate == True and self.passed_8_coordinate == False and lane_center <= 900:
                    lane_center = lane_center - 250
                    #print("gg")
                if self.passed_8_coordinate == True and self.passed_23_coordinate == False:
                    lane_center = lane_center + 220
                    #print("hh")


            if lane_center is not None:
                    #cv2.line(vis_image, (lane_center, 0), (lane_center, vis_image.shape[0]), (255, 0, 0), 2)
                cv2.circle(vis_image, (lane_center, vis_image.shape[0] // 2), 5, (255, 0, 0), -1)
                #cv2.waitKey(1)
                
                #cv2.imshow("Lane Detection", vis_image)
                #cv2.waitKey(1)
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
                    
            """ The snippet below utilizes Intel RealSense depth camera to estimate distance from yield and road cone, but it doesn't work.
            # Stopping in front of road cone. 
            if self.depth_of_cone.data is not None and self.depth_of_cone.data <= 0.7 and self.depth_of_cone.data > 0.0:
                if not self.cone_stopped and self.cone_flag.data == True:
                    self.throttle = 0.0
                    steering_angle = 0.0
                    self.cone_stopped = True
                    self.cone_stop_time = time.time()
                    self.required_led.data = 4.0
                    self.get_led.publish(self.required_led)
                self.publish_motor_command(steering_angle, self.throttle)
                return

            if self.cone_stopped == True and self.cone_flag.data == False and self.depth_of_cone.data > 0.7: 
                self.time_lost += (time.time() - self.cone_stop_time) 
                self.throttle = 0.4
                self.required_led.data = 1.0
                self.get_led.publish(self.required_led)

            if self.depth_of_yield.data is not None and 0.0 < float(self.depth_of_yield.data) <= 0.7:
                if not self.yield_stopped and self.yield_flag.data is True:
                    # Begin stopping
                    self.throttle = 0.0
                    steering_angle = 0.0
                    self.yield_stopped = True
                    self.yield_stop_time = time.time()
                    self.required_led.data = 4.0
                    self.get_led.publish(self.required_led)
                    self.publish_motor_command(steering_angle, self.throttle)
                    return

            # State: Continue stopping for 3 seconds in front of yield
            if self.yield_stopped:
                yield_duration = time.time() - self.yield_stop_time
                if yield_duration < 3.0:
                    # Stay stopped
                    self.throttle = 0.0
                    steering_angle = 0.0
                    self.publish_motor_command(steering_angle, self.throttle)
                    return
                else:
                    # Done waiting, continue driving
                    self.time_lost += yield_duration
                    self.yield_stopped = False
                    self.required_led.data = 1.0  # Resume LED
                    self.get_led.publish(self.required_led)
                    self.throttle = 0.4
                    steering_angle = 0.0
                    self.publish_motor_command(steering_angle, self.throttle)  
            """
            if self.depth_of_stop.data is not None and self.is_stopping == False:
                if self.depth_of_stop.data <= 0.7 and self.depth_of_stop.data >= 0.30 and (time.time() - self.last_stop_time) > self.stop_cooldown: #2 < x < 7.7  max 1.2 physical
                    self.get_logger().info("Stop sign detected within range. Stopping.")
                    self.throttle = 0.0
                    steering_angle = 0.0
                    self.publish_motor_command(steering_angle, self.throttle)
                    self.required_led.data = 4.0
                    self.get_led.publish(self.required_led)
                    self.stop_start_time = time.time()
                    self.is_stopping = True
                    time.sleep(3)
                    self.last_stop_time = time.time()
                    self.required_led.data = 1.0
                    self.get_led.publish(self.required_led)
                    self.is_stopping = False
            
            if self.lidar_alert.data == 1.0:
                if not self.lidar_stopped:
                    self.get_logger().info("LIDAR: Obstacle detected. Stopping.")
                    self.throttle = 0.0
                    steering_angle = 0.0
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
            
            #Red Light is ON. Assign 0 velocity throughout the execution.
            if self.red_flag == True: #If using the detect.py algorithm, then set this as self.red_flag.data
                if self.last_red_time == None:
                    self.last_red_time = time.time()
                self.throttle = 0.0
                steering_angle = 0.0
                self.counter_red = self.counter_red + 1
                self.publish_motor_command(steering_angle, self.throttle)
                self.required_led.data = 4.0
                self.get_led.publish(self.required_led)
                
            #If red light is off:
            else:
                #If red light was turned on before. Check for how long car stopped.
                if self.last_red_time is not None:
                    self.required_led.data = 1.0
                    self.get_led.publish(self.required_led)
                    self.red_counter += time.time() - self.last_red_time
                    self.time_lost = self.last_red_time
                    self.last_red_time = None

                #If car did not stop at light because it was never red.     
                else:
                

                    #sequential driving

                    
                    if self.starting == True:
                        self.required_led.data = 5.0
                        self.get_led.publish(self.required_led)
                        time.sleep(8) #set it to 7
                        self.required_led.data = 1.0
                        self.get_led.publish(self.required_led)
                    	
                        self.coordinate_2 = True
                        self.coordinate_2_timer = time.time() 
                        self.starting = False
                        
                   
                    self.publish_motor_command(steering_angle, self.throttle)
      
                    #Check status of coordinate 2 indicated by user. 
                    if self.coordinate_2 == True and (time.time() - self.coordinate_2_timer) >= (5 + 2.5 + self.time_lost):  #0.5
                        if self.coordinate_2_is_pickup == True:
                            self.handle_coordinate_2() #if user selected it as pickup, go to the defined stop function.
                            self.compensator = 0.4
                        else:
                            self.coordinate_2 = False
                            self.compensator = 0
                         
                        self.time_lost = 0
                        if self.passed_2_coordinate == False:
                            self.coordinate_4_timer = time.time() #start countdown to coordinate 13.
                            self.passed_2_coordinate = True
                            self.coordinate_4 = True
                        
                            
                    #Check status of coordinate 4 indicated by user.



                    if self.coordinate_4 == True and (time.time() - self.coordinate_4_timer) >= (5.2+ 0.65 + self.time_lost): #0.15 
                        if self.coordinate_4_is_pickup == True:
                            self.handle_coordinate_4()
                            self.compensator = 0.35
                        else:
                            self.coordinate_4 = False
                            self.compensator = 0
                        duration_st = time.time()
                        while (time.time() - duration_st <= 1.2 + self.compensator): #drive straight
                            self.throttle = 0.7 
                            steering_angle = -0.14 #-0.09 
                            self.publish_motor_command(steering_angle, self.throttle)   
                        duration = time.time()
                        self.throttle = 0.4 
                        steering_angle = 0.15 + 0.2 + 0.1
                        while (time.time() - duration <= 4 + 1 - 2):
                            self.publish_motor_command(steering_angle, self.throttle)
                        self.passed_4_coordinate= True
                        self.time_lost = 0
                        self.coordinate_6_timer = time.time() #start countdown to coordinate 6.
                        self.coordinate_6 = True
                        
                        #self.activate = True
                    
                    #Check status of coordinate 6 indicated by user.
                    if self.coordinate_6 == True and (time.time() - self.coordinate_6_timer) >= 0.7+self.compensator+self.time_lost:
                        if self.coordinate_6_is_pickup == True: 
                            self.handle_coordinate_6() #Call this stop function if marked as zone. 
                            self.compensator = 0.33
                        else:
                            self.coordinate_6 = False
                            self.compensator = 0
                        self.passed_6_coordinate = True

                        sp = time.time()
                        self.throttle = 0.69
                        steering_angle = -0.08
                        while (time.time() - sp <= 3.2 - 0.4):
                            self.publish_motor_command(steering_angle, self.throttle)
                        self.time_lost = 0
                        self.coordinate_8_timer = time.time() 
                        self.coordinate_8 = True
                        #self.activate = True

                    if self.coordinate_8 == True and (time.time() - self.coordinate_8_timer) >= 0.4+self.compensator+self.time_lost:
                        if self.coordinate_8_is_pickup == True: 
                            self.compensator = 0.3
                            self.handle_coordinate_8() #Call this stop function if marked as zone. 
                        else:
                            self.coordinate_8 = False
                            self.compensator = 0
                        self.passed_8_coordinate = True
                        self.time_lost = 0
                        self.coordinate_23_timer = time.time()
                        self.coordinate_23 = True
                        self.activate = False
                        self.second_lane_adjuster = time.time()
                       

                    if self.coordinate_23 == True and (time.time() - self.coordinate_23_timer) >= 7.7 + self.compensator:
                        if self.coordinate_23_is_pickup == True: 
                            self.handle_coordinate_23() #Call this stop function if marked as zone. 
                        else:
                            self.coordinate_23 = False
                            self.compensator = 0
                        self.passed_23_coordinate = True

                        g = time.time()
                        self.throttle = 0.4
                        self.steering_angle = -0.99
                        while (time.time() - g) <= 5.3:
                            self.publish_motor_command(self.steering_angle, self.throttle)
                        v = time.time()
                        self.throttle = -0.4
                        self.steering_angle = 0.32
                        while (time.time() - v) <= 2.2:
                            self.publish_motor_command(self.steering_angle, self.throttle)
                        h = time.time()
                        self.throttle = 0.7
                        self.steering_angle = -0.14
                        while (time.time() - h) <= 3.2:
                            self.publish_motor_command(self.steering_angle, self.throttle)
                        self.throttle = 0.0
                        self.steering_angle = -0.0
                        self.publish_motor_command(self.steering_angle, self.throttle)

                        self.time_lost = 0
                        self.coordinate_25 = True
                        self.coordinate_25_timer = time.time()
                        

                    if self.coordinate_25 == True and (time.time() - self.coordinate_25_timer) >= 0:
                        if self.coordinate_25_is_pickup == True: 
                            self.handle_coordinate_25() #Call this stop function if marked as zone. 
                            
                        else:
                            self.coordinate_25 = False
                            self.compensator = 0
                        self.passed_25_coordinate = True

                        hr = time.time()
                        self.throttle = -0.69
                        self.steering_angle = -0.14
                        while (time.time() - hr) <= 4:
                            self.publish_motor_command(self.steering_angle, self.throttle)
                        
                        self.time_lost = 0
                        self.coordinate_24 = True
                        self.coordinate_24_timer = time.time()
                       

                    if self.coordinate_24 == True and (time.time() - self.coordinate_24_timer) >= 0:
                        if self.coordinate_24_is_pickup == True: 
                            self.handle_coordinate_24() #Call this stop function if marked as zone. 
                        else:
                            self.coordinate_24 = False
                        self.passed_24_coordinate = True 

                        vr = time.time()
                        self.throttle = 0.69
                        self.steering_angle = -0.4
                        while (time.time() - vr) <= 3.1:
                            self.publish_motor_command(self.steering_angle, self.throttle)
                
                        gr = time.time()
                        self.throttle = 0.4
                        self.steering_angle = 0.4
                        while (time.time() - gr) <= 2.5:
                            self.publish_motor_command(self.steering_angle, self.throttle)                   
                        self.time_lost = 0

                        self.coordinate_return = True 
                        self.coordinate_return_timer = time.time()

                    if self.coordinate_return == True and (time.time() - self.coordinate_return_timer) >= 2:
                        
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
                               
                    else: 
                        self.throttle = 0.40 #0.23
                    if self.is_stopping == True:
                        steering_angle = 0.0
                        self.throttle = 0.0 
                               
                               
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
