#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, TrafficWaypoint, Waypoint

from enum import Enum

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200; # Number of waypoints we will publish. You can change this number
FLOAT_EPSILON = 0.00001;
MIN_STOP_TIME = 1;
SPEED_TOLERANCE = 1; # m/s less than the speed limit to follow
KPH_TO_MPS = 0.2777778;


def angleDiff(a, b):
    '''
    Get the difference between two angles
    '''
    diff = math.fmod(b - a + math.pi, math.pi * 2.0);
    if (diff < 0):
        diff += math.pi * 2.0;
    return diff - math.pi;


class WaypointUpdater(object):

    class State(Enum):
        STOP = 0
        GO = 1

    def __init__(self):
        rospy.logdebug('Starting up node...');
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.twist_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        sub1 = rospy.Subscriber("/traffic_waypoint", TrafficWaypoint, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.SPEED_LIMIT = (rospy.get_param("/waypoint_loader/velocity") * KPH_TO_MPS) - SPEED_TOLERANCE;

        # Base list of waypoints
        self.base_waypoints = [];
        # If the index is -1 then no detection has been made
        self.lightindex = -1;
        self.lightstate = TrafficWaypoint.UNKNOWN;

        self.last_velocity = 0;

        self.state = WaypointUpdater.State.STOP;

        rospy.logdebug('Fully initialized, about to spin');
        rospy.spin()

    def update_state(self, distanceToLight):
        failedToDetectLight = (self.lightindex < 0);

        if (self.state == WaypointUpdater.State.GO):
            if (failedToDetectLight):
                pass;
            elif (self.lightstate == TrafficWaypoint.RED):
                self.state = WaypointUpdater.State.STOP;
            elif (self.lightstate == TrafficWaypoint.YELLOW):
                # Go to stop if we have enough time to stop
                if ((abs(self.last_velocity) < FLOAT_EPSILON) or
                    (distanceToLight / self.last_velocity > MIN_STOP_TIME)):
                    self.state = WaypointUpdater.State.STOP;

        elif (self.state == WaypointUpdater.State.STOP):
            lightIsGreen = (self.lightstate == TrafficWaypoint.GREEN);

            if (failedToDetectLight or lightIsGreen):
                self.state = WaypointUpdater.State.GO;

        else:
            rospy.logerr('Unexpected state {}, resetting to STOP'.format (self.state))
            self.state = WaypointUpdater.State.STOP;

    def execute_state(self, lane, numwpts):
        if (self.state == WaypointUpdater.State.GO):
            self.execute_state_go(lane);
        elif (self.state == WaypointUpdater.State.STOP):
            self.execute_state_stop(lane, numwpts);

    def execute_state_go(self, lane):
        self.final_waypoints_pub.publish(lane);

    def execute_state_stop(self, lane, numwpts):
        v = 0.0;
        a = .1;

        # Set velocity for everything after the line to 0
        for i in range(0, LOOKAHEAD_WPS):
            self.set_waypoint_velocity(lane.waypoints, i, 0.0);
        # Ramp down the velocity for everything leading up to the line
        for i in range(numwpts-1, -1, -1):
            if (i < len(lane.waypoints)):
                self.set_waypoint_velocity(lane.waypoints, i, v);

            v += a;

            if (v > self.SPEED_LIMIT):
                v = self.SPEED_LIMIT;

        self.final_waypoints_pub.publish(lane);

    def pose_cb(self, msg):
        '''
        :param msg: Current position of the vehicle of type PoseStamped
        '''

        lane = Lane();

        # Pass through the header info
        lane.header = msg.header

        startwpindex = self.find_closest_waypoint(self.base_waypoints, msg.pose);

        if (startwpindex < 0):
            rospy.logerr('pose_cb: start index invalid');
            return;

        for i in range (LOOKAHEAD_WPS):
            lane.waypoints.append (self.base_waypoints[(startwpindex + i) % len(self.base_waypoints)]);
            self.set_waypoint_velocity(lane.waypoints, i, self.SPEED_LIMIT);

        numwpts = 0;
        distanceToLight = 0;

        if (self.lightindex < 0):
            numwpts = LOOKAHEAD_WPS;
            distanceToLight = 10000;
        elif (startwpindex <= self.lightindex):
            numwpts = self.lightindex - startwpindex;
            distanceToLight = self.distance(lane.waypoints, 0, min(LOOKAHEAD_WPS, numwpts)-1);
        else:
            numwpts = self.lightindex + len(self.base_waypoints) - startwpindex;
            distanceToLight = self.distance(lane.waypoints, 0, min(LOOKAHEAD_WPS, numwpts)-1);

        self.update_state(distanceToLight);
        self.execute_state(lane, numwpts);

        #rospy.logwarn('state {} v0 {:+6.3f} numwpts {} start {} end {}'.format(
        #    self.state, self.get_waypoint_velocity(lane.waypoints[0]),
        #    numwpts, startwpindex, self.lightindex))
        return;

    def twist_cb(self, msg):
        self.last_velocity = msg.twist.linear.x;

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints;

    def traffic_cb(self, msg):
        # This is the index of the nearest upcoming red light
        # Used in pose_cb
        self.lightindex = msg.index;
        self.lightstate = msg.state;

        #if msg.state == TrafficWaypoint.RED:
        #    state = "red"
        #elif msg.state == TrafficWaypoint.YELLOW:
        #    state = "yellow"
        #elif msg.state == TrafficWaypoint.GREEN:
        #    state = "green"
        #else:
        #    state = "unknown"
        #rospy.logdebug('traffic light; waypoint: {}; state: {}'.format(msg.index, state))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        '''
        Computes the distance between two waypoints in a list along the piecewise linear arc
        connecting all waypoints between the two.

        :param waypoints: a list of waypoints
        :param wp1: Index of the first waypoint in the list
        :param wp2: Index of the second waypoint in the list
        '''
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def find_closest_waypoint(self, waypoints, pose):
        '''
        Find the closet waypoint to the given position
        :return: Index of the closest waypoint
        '''

        closestIndex = -1;
        closestDistSquared = 99999999;

        dsquared = lambda a, b: (a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2;
        calcAzimuth = lambda a, b: math.atan2(b.y-a.y, b.x-a.x);
        for i, wp in enumerate (waypoints):
            # Is this waypoint the closest candidate?
            testDistSquared = dsquared(wp.pose.pose.position, pose.position);
            if (testDistSquared >= closestDistSquared):
                continue;

            # Is this waypoint in front of us?
            vehicleToPointAzimuth = calcAzimuth(pose.position, wp.pose.pose.position);
            quaternion = (pose.orientation.x, pose.orientation.y,
                          pose.orientation.z, pose.orientation.w);
            heading = tf.transformations.euler_from_quaternion(quaternion)[2]

            if (math.fabs(angleDiff(vehicleToPointAzimuth, heading)) >= math.pi/2.0):
                continue;

            closestDistSquared = testDistSquared;
            closestIndex = i;

        if (closestIndex < 0):
            rospy.logerr('find_closest_waypoint: no waypoint could be found');

        return closestIndex;


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
