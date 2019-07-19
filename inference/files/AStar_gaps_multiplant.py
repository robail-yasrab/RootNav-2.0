import time
import cv2
from PIL import Image
import numpy as np
from FibonacciHeap import FibHeap
from priority_queue import HeapPQ
import math

def AStar_pri_m(start, goal, neighbor_nodes, distance, cost_estimate, weights):
    width, height = 512, 512 
    weights = weights.reshape((512 * 512)).tolist()
    def idx(pos):
        return pos[1] * width + pos[0]

    total_size = width * height
    infinity = float("inf")
    distances = [infinity] * total_size

    visited = [False] * total_size
    prev = [None] * total_size

    unvisited = HeapPQ();

    node_index = [None] * total_size;

    distances[idx(start)] = 0

    start_node = FibHeap.Node(0, start)
    node_index[idx(start)] = start_node
    unvisited.insert(start_node)

    count = 0
    completed = False
    plant_id = -1
    final_goal_position = None

    while len(unvisited) > 0:
        n = unvisited.removeminimum()

        upos = n.value
        uposindex = idx(upos)

        if distances[uposindex] == infinity:
            break

        if upos in goal:
            completed = True
            plant_id = goal[upos]
            final_goal_position = upos
            break

        for v in neighbor_nodes(upos):
            vpos = v[0]
            vposindex = idx(vpos)

            if is_blocked_edge(vpos):
                continue

            if visited[vposindex]:
                continue

            # Calculate distance to travel to vpos
            d = weights[vpos[1] * width + vpos[0]]

            new_distance = distances[uposindex] + d * v[1]

            if new_distance < distances[vposindex]:
                vnode = node_index[vposindex]

                if vnode is None:
                    vnode = FibHeap.Node(new_distance, vpos)
                    unvisited.insert(vnode)
                    node_index[vposindex] = vnode
                    distances[vposindex] = new_distance
                    prev[vposindex] = upos
                else:
                    unvisited.decreasekey(vnode, new_distance)
                    distances[vposindex] = new_distance
                    prev[vposindex] = upos

        visited[uposindex] = True

    if completed:
        from collections import deque
        path = deque()
        current = final_goal_position
        while current is not None:
            path.appendleft(current)
            current = prev[idx(current)]

        return path, plant_id
    else:
        return []

rt2 = math.sqrt(2)

def von_neumann_neighbors(p):
    x, y = p
    return [((x-1, y-1),rt2),((x-1, y),1), ((x, y-1),1), ((x+1, y),1), ((x, y+1),1),((x-1, y+1),rt2),((x+1, y-1),rt2),((x+1, y+1),rt2)]

def manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def is_blocked_edge(p):
    x, y = p
    return not (x >= 0 and y >= 0 and x < 512 and y < 512)





##############################  DIS_MAP   ############################################
'''bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)


def distance_to_weights(d):
    mx = 3
    epsilon = 0.01
    d = np.clip(d, 0, mx)
    cv2.normalize(d, d, 0, 1.0, cv2.NORM_MINMAX)
    d = 1 - d
    d = np.maximum(d, epsilon)
    return d

weights = distance_to_weights(dist)

dist = weights

cv2.imshow('Distance Transform Image', dist)
dist = dist*255
cv2.imwrite('DIS_MAP.png',dist)'''
#####################################################################################

#path_img = Image.fromarray(np.uint8(img))
#path_pixels = path_img.load()

'''
distance = manhattan
heuristic = manhattan
for i in start:
    path = AStar(512, 512, i, goal, von_neumann_neighbors, distance, heuristic, weights)
    if path == []:
        print ("not found")
    else:
        print ("found on plant")
    for position in path:
        x,y = position
        path_pixels[x,y] = (255,0,0) # red

path_img.save(argv[1])
t1 = time.time()
total = t1-t0
print ("Time elapsed:", total, "\n")

path_img = np.array(path_img)
cv2.imshow('DIY convolution', path_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''



