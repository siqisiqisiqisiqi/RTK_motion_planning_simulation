import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pathlib import Path

path = str(pathlib.Path(__file__).parent.parent)
os.chdir(path)

fp = "data/RTK_read_data/03_05_GPS_small_circle.csv"

trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')

x,y,z = trajectory[:,0], trajectory[:,1], trajectory[:,2]
x = x - x[1]
y = y - y[1]
z = z - z[1]

############################2D plot##########################
plt.plot(x,y)
plt.xlabel("x axis /m")
plt.ylabel("y axis /m")
plt.title('2D plot')
plt.show()


################################z-axis plot########################
plt.plot(list(range(len(z))), z)
plt.xlabel("timestamp")
plt.ylabel("z axis /m")
plt.title('z-axis plot')
plt.show()

##################################3D plot#########################
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z, label='3D trajectory')
ax.legend()
ax.set_xlabel('$X$/m', fontsize=10, rotation=150)
ax.set_ylabel('$Y$/m', fontsize=10)
ax.set_zlabel('$z$/m', fontsize=10, rotation=60)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
ax.set_zticks([-2,-1, 0, 1, 2])
plt.show()

################################2D animation########################

import matplotlib.animation as animation 
plt.style.use('dark_background')

fig = plt.figure() 
ax = plt.axes(xlim=(-25, 20), ylim=(-5, 30)) 
line, = ax.plot([], [], lw=2) 

# initialization function 
def init(): 
	# creating an empty plot/frame 
	line.set_data([], []) 
	return line, 

# lists to store x and y axis points 
xdata, ydata = [], [] 

# # animation function 
def animate(i): 
	# t is a parameter 


    xx = x[2*i+1000]
    yy = y[2*i+1000]

    xdata.append(xx) 
    ydata.append(yy) 
    line.set_data(xdata, ydata) 
    return line, 
	
# setting a title for the plot 
plt.title('RTK Circle Test') 
# hiding the axis details 
# plt.axis('off') 

# call the animator	 
anim = animation.FuncAnimation(fig, animate, init_func=init, 
							frames=900, interval=20, blit=True) 

# save the animation as mp4 video file 
anim.save('tractor_trajectory_small_circle.gif',writer='imagemagick') 



