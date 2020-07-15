import octomap
for i in vars(octomap.OcTree):
    if(i.startswith('r')):
        print i

# import sys
# sys.path.insert(1,'/root/catk_ws/src/octomap_msgs/include/octomap_msgs')
# import conversions
# print "CIAO"
# for i in vars(conversions):
#     print i