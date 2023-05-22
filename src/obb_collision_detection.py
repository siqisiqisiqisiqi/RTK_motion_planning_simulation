import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

pi = math.pi

# separating axis theorem 
def calculate_vertice(ob):
    """calculate the two rectangle vertice position

    Parameters
    ----------
    ob : obstacle description [x,y,yaw,width,length]

    Returns
    -------
    vertice position: ndarray
    """

    vertice = np.zeros((4, 2))

    theta = ob[2]
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy1 = np.array([[ob[3] / 2], [-ob[4] / 2]])
    xy1 = np.matmul(rotation_matrix, xy1).T[0]
    xy2 = np.array([[ob[3] / 2], [ob[4] / 2]])
    xy2 = np.matmul(rotation_matrix, xy2).T[0]

    vertice[0,:] = np.array([ob[0]+xy1[0], ob[1]+xy1[1]])
    vertice[1,:] = np.array([ob[0]+xy2[0], ob[1]+xy2[1]]) 
    vertice[2,:] = np.array([ob[0]-xy1[0], ob[1]-xy1[1]]) 
    vertice[3,:] = np.array([ob[0]-xy2[0], ob[1]-xy2[1]])

    return vertice

def edges_of(vertices):
    """
    Return the vectors for the edges of the polygon p.

    p is a polygon.
    """
    edges = []
    N = 4

    for i in range(2):
        edge = vertices[(i + 1)%N] - vertices[i]
        edges.append(edge)

    return edges

def orthogonal(v):
    """
    Return a 90 degree clockwise rotation of the vector v.
    """
    return np.array([-v[1], v[0]])

def is_separating_axis(o, p1, p2):
    """
    Return True and the push vector if o is a separating axis of p1 and p2.
    Otherwise, return False and None.
    """
    min1, max1 = float('+inf'), float('-inf')
    min2, max2 = float('+inf'), float('-inf')

    for v in p1:
        projection = np.dot(v, o)

        min1 = min(min1, projection)
        max1 = max(max1, projection)

    for v in p2:
        projection = np.dot(v, o)

        min2 = min(min2, projection)
        max2 = max(max2, projection)

    if max1 >= min2 and max2 >= min1:
        return False, None
    else:
        d = abs(max1 + min1 - (max2 + min2))/2 
        return True, d

def collide(p1, p2):
    '''
    Return True and the MPV if the shapes collide. Otherwise, return False and
    None.

    p1 and p2 are lists of ordered pairs, the vertices of the polygons in the
    counterclockwise direction.
    '''

    p1 = [np.array(v, 'float64') for v in p1]
    p2 = [np.array(v, 'float64') for v in p2]

    edges = edges_of(p1)
    edges += edges_of(p2)
    orthogonals = [orthogonal(e) for e in edges]
    d_max = -1
    for o in orthogonals:

        separates, d = is_separating_axis(o, p1, p2)

        if separates:
            # they do not collide and there is no push vector
            if d > d_max:
                d_max = d

    if d_max > 0:
        return False, d_max
    else:
        return True, None


if __name__ == "__main__":
    # define two rectangle ob = [x,y,yaw,width,length]
    ob1 = [4,8,pi/2,1.0,2.0]
    ob2 = [7,8,pi*2/6,2.0,3.0]
    ob = np.array([ob1,ob2])

    #calculate the vertice
    v1 = calculate_vertice(ob1)
    v2 = calculate_vertice(ob2)

    detection, dist = collide(v1, v2)
    print(dist)

    # visualize the two bounding box
    ax = plt.gca()

    plt.plot(ob[:, 0], ob[:, 1], "xk")

    for i in range(ob.shape[0]):

        theta = ob[i,2]
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = np.array([[ob[i, 3] / 2], [ob[i, 4] / 2]])
        xy = np.matmul(rotation_matrix, xy)
        xy = np.array([[ob[i, 0]], [ob[i, 1]]]) - xy
        rect = Rectangle((xy[0, 0], xy[1, 0]), ob[i,3], ob[i,4], angle=ob[i,2]*180/pi
                        , linewidth=2, edgecolor='k', facecolor='c')
        ax.add_patch(rect)

    for i in range(4):
        plt.plot(v1[i,0], v1[i,1], "xr")
        plt.plot(v2[i,0], v2[i,1], "xr")

    plt.legend()
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(False)
    plt.show()


