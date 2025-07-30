import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float32
from rclpy.parameter import Parameter as RclpyParameter
from rcl_interfaces.srv import SetParameters

class Led_Controller(Node):
    def __init__(self):
        super().__init__('led_controller')
        self.last_led_value = None

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        self.subscription_green = self.create_subscription(
            Float32, '/desired_led', self.call_led_controller, qos_profile)

        self.param_client = self.create_client(SetParameters, '/qcar2_hardware/set_parameters')
        attempt = 0
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            attempt += 1
            self.get_logger().warn(f"[Attempt {attempt}] /qcar2_hardware/set_parameters not available yet")

        self.get_logger().info("LED Controller ready and service client connected.")

    def call_led_controller(self, msg):
        self.get_logger().info(f"Received LED command: {msg.data}")
        if msg.data == self.last_led_value:
            self.get_logger().info("Duplicate value, skipping.")
            return

        self.last_led_value = msg.data

        value_map = {
            1.0: 1,
            2.0: 2,
            3.0: 3,
            4.0: 0,
            5.0: 5
        }

        if msg.data not in value_map:
            self.get_logger().warn(f"Unknown LED value: {msg.data}")
            return

        led_value = value_map[msg.data]

        param = RclpyParameter('led_color_id', RclpyParameter.Type.INTEGER, led_value)

        # Convert to service request format
        request = SetParameters.Request()
        request.parameters = [param.to_parameter_msg()]

        self.get_logger().info(f"Attempting to set LED to {led_value}...")

        future = self.param_client.call_async(request)

        def done_callback(fut, led_value=led_value):
            try:
                response = fut.result()
                if not response.results:
                    self.get_logger().error("No results received in parameter response!")
                    return
                result = response.results[0]
                if result.successful:
                    self.get_logger().info(f"Set LED to {led_value}")
                else:
                    self.get_logger().error(f"Failed to set LED: {result.reason}")
            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")

        future.add_done_callback(done_callback)

def main(args=None):
    rclpy.init(args=args)
    led_controller = Led_Controller()
    rclpy.spin(led_controller)
    led_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


