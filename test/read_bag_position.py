import os
import pathlib

path = str(pathlib.Path(__file__).parent.parent)
os.chdir(path)

import rosbag
import numpy as np
import csv

bag = rosbag.Bag('data/RTK_DATA/record1.bag')


position = []
heading = []
fields = ['x', 'y', 'z', 'heading', 'time']

a = 0
for topic, msg, t in bag.read_messages(topics='/novatel/oem7/odom'):
    position.append([msg.pose.pose.position.x,
                    msg.pose.pose.position.y, msg.pose.pose.position.z])

# for pose in msg.poses:
#     position.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.z, pose.header.stamp.secs])

bag.close()

with open('data/RTK_read_data/July24.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(position)
