#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

class LaserProcessor(Node):

    def __init__(self):
        super().__init__('laser_processor')

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,  # Volatile (sensor data) history
            depth=10,  # Buffer depth for messages
            reliability=QoSReliabilityPolicy.RELIABLE  # Best Effort Reliability
        )

        self.subscription = self.create_subscription(LaserScan, '/scan', self.reportit, qos)
        
        self.collision_alert = self.create_publisher(Float32, '/collision_alert', qos)
        self.laser = LaserScan()
        self.warning_distance = 0.5  # meters
        self.alert = 0.0
        self.request_to_stop = Float32()
        self.request_to_stop.data = 0.0

    def reportit(self, msg):
        self.laser = msg
        npranges = np.array(msg.ranges)
        npranges[npranges < msg.range_min] = float('nan')
        npranges[npranges > msg.range_max] = float('nan')

        front_angles = []
        front_distances = []
        x_distances = []
        y_distances = []
        
        num_points = len(npranges)
        
        # Define front cone: ±30° (i.e., -pi/6 to +pi/6 radians)
        # front_min_angle = -math.radians(30)
        # front_max_angle = math.radians(30)

        front_min_deg = 150 + 30  # equivalent to -30°
        front_max_deg = 210 - 10  # +30°

        """
        for i in range(num_points):
            angle = msg.angle_min + i * msg.angle_increment

            # self.get_logger().info(f'🚗 Angle: {angle:.2f} ...')
            if front_min_angle <= angle <= front_max_angle:

                distance = npranges[i]
                # self.get_logger().info(f'🚗 Angle: {angle:.2f} ...Distance: {distance:.2f}')
                if math.isnan(distance) or distance == 0.0:
                    continue
                front_angles.append(angle)
                front_distances.append(distance)
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                x_distances.append(x)
                y_distances.append(y)

                # Define front cone in degrees, then normalize
        """

        # Inside loop:
        for i in range(num_points):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle) % 360  # Normalize angle to [0, 360)

            # Check if angle is within the wraparound range
            if front_min_deg <= angle_deg <= front_max_deg:
                distance = npranges[i]
                if math.isnan(distance) or distance == 0.0:
                    continue
                front_angles.append(math.radians(angle_deg))  # Store as radians if needed
                front_distances.append(distance)
                x = distance * math.cos(math.radians(angle_deg))
                y = distance * math.sin(math.radians(angle_deg))
                x_distances.append(x)
                y_distances.append(y)

        #self.get_logger().info('\n🔍 Scanning Front Region...')
        self.alert = 0.0

        if front_distances:
            #self.get_logger().info(f'🚗 front_distances: {front_distances}')
            closest = min(front_distances)
            #self.get_logger().info(f'🚗 Closest Object Ahead: {closest:.2f} meters')

            if closest < self.warning_distance:
                #self.get_logger().warn('Warning: Object too close ahead!')
                self.alert = 1.0
                self.request_to_stop.data = self.alert
                self.collision_alert.publish(self.request_to_stop)
            else:
                self.alert = 0.0
                self.request_to_stop.data = self.alert
                self.collision_alert.publish(self.request_to_stop)


def main(args=None):
    rclpy.init(args=args)
    laser_processor = LaserProcessor()
    rclpy.spin(laser_processor)
    laser_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

