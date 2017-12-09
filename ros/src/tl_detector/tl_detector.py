#!/usr/bin/env python
import sys
from collections import namedtuple
from math import pow, sqrt
from threading import Lock

import yaml

import cv2
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from light_classification.tl_classifier import TLClassifier
from sensor_msgs.msg import Image
from styx_msgs.msg import (Lane, TrafficLight, TrafficLightArray,
                           TrafficWaypoint)

STATE_COUNT_THRESHOLD = 3

Point = namedtuple('Point', ['x', 'y'])

LightData = namedtuple('LightData', ['index', 'state'])


def distance(point1, point2):
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2))


def get_light_point(light):
    position = light.pose.pose.position
    return Point(int(position.x), int(position.y))


class Detector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.mutex = Lock()

        self.state = TrafficLight.UNKNOWN
        self.state_count = 0
        self.last_stop_line = None

        self.shared_car_index = None
        self.shared_waypoints = None
        self.shared_stop_lines = None

        self.pub = rospy.Publisher('/traffic_waypoint', TrafficWaypoint,
                                   queue_size=1)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

    def loop(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()

            stop_line = self.get_stop_line()
            if stop_line is None:
                continue

            state = self.get_traffic_light_state()
            if state == TrafficLight.UNKNOWN:
                continue

            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count == STATE_COUNT_THRESHOLD \
                    or self.state_count > STATE_COUNT_THRESHOLD \
                    and self.last_stop_line != stop_line:
                self.last_stop_line = stop_line
                tw = TrafficWaypoint(stop_line, state)
                self.pub.publish(tw)

            self.state_count += 1

    def pose_cb(self, msg):
        with self.mutex:
            if self.shared_waypoints is None:
                return

            if self.shared_car_index is None or self.shared_car_index > self.shared_stop_lines[-1]:
                self.shared_car_index = 0

            current_dist = distance(msg.pose.position,
                                    self.shared_waypoints[self.shared_car_index])

            for i in range(self.shared_car_index + 1, len(self.shared_waypoints)):
                dist = distance(msg.pose.position, self.shared_waypoints[i])
                if dist > current_dist:
                    break
                current_dist = dist
                self.shared_car_index = i

    def waypoints_cb(self, msg):
        with self.mutex:
            config_string = rospy.get_param('/traffic_light_config')
            stop_lines = yaml.load(config_string)['stop_line_positions']

            if len(stop_lines) == 0:
                return

            self.shared_waypoints = []
            for waypoint in msg.waypoints:
                position = waypoint.pose.pose.position
                self.shared_waypoints.append(Point(position.x, position.y))

            self.shared_stop_lines = []
            for position in stop_lines:
                index = self.get_closest_waypoint(
                    Point(position[0], position[1]))
                self.shared_stop_lines.append(index)
            self.shared_stop_lines.sort()

    def get_closest_waypoint(self, point):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            point: point to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.shared_waypoints

        """
        if self.shared_waypoints is None or len(self.shared_waypoints) == 0:
            rospy.logerr('No waypoints in traffic light detector')
            return -1

        closest_dist = sys.maxint

        for i in range(len(self.shared_waypoints)):
            dist = distance(self.shared_waypoints[i], point)
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i

        return closest_index

    def get_stop_line(self):
        with self.mutex:
            if self.shared_car_index is None:
                return None

            for index in self.shared_stop_lines:
                if index > self.shared_car_index:
                    return index

            return None


class ImageDetector(Detector):
    def __init__(self):
        super(ImageDetector, self).__init__()

        variant = rospy.get_param('variant')
        config_string = rospy.get_param('/traffic_light_config')
        camera_info = yaml.load(config_string)['camera_info']

        self.shared_camera_image = None

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(variant, camera_info['image_width'],
                                             camera_info['image_height'])

        rospy.Subscriber('/image_color', Image, self.image_cb)

    def image_cb(self, msg):
        with self.mutex:
            self.shared_camera_image = msg

    def get_traffic_light_state(self):
        with self.mutex:
            if self.shared_camera_image is None:
                return TrafficLight.UNKNOWN
            camera_image = self.shared_camera_image

        image = self.bridge.imgmsg_to_cv2(camera_image, 'rgb8')

        return self.light_classifier.get_classification(image)


class DummyDetector(Detector):
    def __init__(self):
        super(DummyDetector, self).__init__()

        self.shared_traffic_lights = None
        self.tl_map = {}

        rospy.Subscriber('/vehicle/traffic_lights',
                         TrafficLightArray, self.traffic_cb)

    def traffic_cb(self, msg):
        with self.mutex:
            if self.shared_waypoints is None:
                return

            if self.shared_traffic_lights is None:
                self.shared_traffic_lights = {}
                for light in msg.lights:
                    point = get_light_point(light)
                    index = self.get_closest_waypoint(light.pose.pose.position)
                    self.shared_traffic_lights[point] = LightData(
                        index, light.state)
                    self.tl_map[index] = point
            else:
                for light in msg.lights:
                    point = get_light_point(light)
                    index = self.shared_traffic_lights[point].index
                    self.shared_traffic_lights[point] = LightData(
                        index, light.state)

    def get_traffic_light_state(self):
        with self.mutex:
            if self.shared_traffic_lights is None:
                return TrafficLight.UNKNOWN

            tl_index = None

            for index in sorted(self.tl_map.keys()):
                if index > self.shared_car_index:
                    tl_index = index
                    break

            if tl_index == None:
                return TrafficLight.UNKNOWN

            dist = distance(self.shared_waypoints[tl_index],
                            self.shared_waypoints[self.shared_car_index])
            if dist > 150:
                return TrafficLight.UNKNOWN

            return self.shared_traffic_lights[self.tl_map[tl_index]].state


if __name__ == '__main__':
    try:
        if rospy.has_param('/dummy_traffic_light_detector') \
           and rospy.get_param('/dummy_traffic_light_detector'):
            detector = DummyDetector()
        else:
            detector = ImageDetector()
        detector.loop()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
