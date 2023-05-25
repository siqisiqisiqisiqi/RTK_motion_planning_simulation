import os
import pathlib

path = str(pathlib.Path(__file__).parent.parent)
os.chdir(path)

import rosbag
import numpy as np
import csv

bag = rosbag.Bag('data/RTK_DATA/2023-03-05-11-12-29_0.bag')


position = []
heading = []
fields = ['x', 'y', 'z', 'heading','time']

a = 0
for topic, msg, t in bag.read_messages(topics='/trajectory'):
    a = a + 1

for pose in msg.poses:
    position.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.z, pose.header.stamp.secs])

bag.close()

with open('data/RTK_read_data/test.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(position)


