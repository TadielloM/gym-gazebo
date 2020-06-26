import sensor_msgs.point_cloud2  as pcl
for i in vars(pcl):
    print i

# # import sys
# sys.path.insert(1,'/root/catk_ws/src/octomap_msgs/include/octomap_msgs')
# import conversions
# print "CIAO"
# for i in vars(conversions):
#     print i

# import numpy as np

# x = np.array([-2,0,0.2, 6.4, 3.0, 1.6, 12])
# bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
# inds = np.digitize(x, bins)
# print inds