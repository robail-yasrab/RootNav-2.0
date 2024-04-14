from .FibHeapQueue import FibHeap, HeapPQ
import math

def AStar_Pri(start, goal, neighbor_nodes, cost_estimate, weights, max_path_length):
    width, height = 512, 512 
    astar_weight = 0.4

    multi_plant = len(goal) > 1
    goal_pos = list(goal)[0]
    
    weights = weights.reshape((512*512)).tolist()

    def idx(pos):
        return pos[1] * width + pos[0]

    total_size = width * height
    infinity = float("inf")
    distances = [infinity] * total_size

    visited = [False] * total_size
    prev = [None] * total_size

    unvisited = HeapPQ()

    node_index = [None] * total_size

    distances[idx(start)] = 0

    start_node = FibHeap.Node(0, start)
    node_index[idx(start)] = start_node
    unvisited.insert(start_node)

    count = 0
    aa = 0
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
            d = weights[vposindex]

            new_distance = distances[uposindex] + d * v[1]     

            if new_distance < distances[vposindex]:
                aa = distances[vposindex]
                vnode = node_index[vposindex]

                if vnode is None:
                    if multi_plant:
                        vnode = FibHeap.Node(new_distance, vpos)
                    else:
                        remaining = astar_weight * cost_estimate(vpos, goal_pos)
                        vnode = FibHeap.Node(new_distance + remaining, vpos)
                    unvisited.insert(vnode)
                    node_index[vposindex] = vnode
                    distances[vposindex] = new_distance
                    prev[vposindex] = upos
                    aa = distances[vposindex]
                else:
                    if multi_plant:
                        unvisited.decreasekey(vnode, new_distance)
                    else:
                        remaining = astar_weight * cost_estimate(vpos, goal_pos)
                        unvisited.decreasekey(vnode, new_distance + remaining)
                    distances[vposindex] = new_distance
                    prev[vposindex] = upos
                    aa = distances[vposindex]

        visited[uposindex] = True

    if completed and aa <= max_path_length:
        from collections import deque
        path = deque()
        current = final_goal_position
        while current is not None:
            path.appendleft(current)
            current = prev[idx(current)]

        return path, plant_id
    else:
        return [], []

def AStar_Lat(start, goal, neighbor_nodes, weights, max_path_length):
    width, height = 512, 512 

    weights = weights.reshape((512*512)).tolist()

    def idx(pos):
        return pos[1] * width + pos[0]

    total_size = width * height
    infinity = float("inf")
    distances = [infinity] * total_size

    visited = [False] * total_size
    prev = [None] * total_size

    unvisited = HeapPQ()

    node_index = [None] * total_size;

    distances[idx(start)] = 0

    start_node = FibHeap.Node(0, start)
    node_index[idx(start)] = start_node
    unvisited.insert(start_node)

    count = 0
    aa= 0 ## to make sure not get too long roots
    
    completed = False
    plant_id = -1
    primary_id = -1
    final_goal_position = None

    while len(unvisited) > 0:
        n = unvisited.removeminimum()

        upos = n.value
        uposindex = idx(upos)

        if distances[uposindex] == infinity:
            break

        if upos in goal:
            completed = True
            #plant_id = goal[upos]
            final_goal_position = upos

            if isinstance(goal,dict):
                primary_id = goal[upos]
            #print (final_goal_position)
            break

        for v in neighbor_nodes(upos):
            vpos = v[0]
            vposindex = idx(vpos)

            if is_blocked_edge(vpos):
                continue

            if visited[vposindex]:
                continue

            # Calculate distance to travel to vpos
            d = weights[vposindex]

            new_distance = distances[uposindex] + d * v[1]

            if new_distance < distances[vposindex]:
                aa= distances[vposindex]
                vnode = node_index[vposindex]

                if vnode is None:
                    vnode = FibHeap.Node(new_distance, vpos)
                    unvisited.insert(vnode)
                    node_index[vposindex] = vnode
                    distances[vposindex] = new_distance
                    prev[vposindex] = upos
                    aa= distances[vposindex]
                else:
                    unvisited.decreasekey(vnode, new_distance)
                    distances[vposindex] = new_distance
                    prev[vposindex] = upos
                    aa= distances[vposindex]

        visited[uposindex] = True

    if completed and aa <= max_path_length:
        from collections import deque
        path = deque()
        current = final_goal_position
        while current is not None:
            path.appendleft(current)
            current = prev[idx(current)]

        return path, primary_id
    else:
        return [],[]


rt2 = math.sqrt(2)

def von_neumann_neighbors(p):
    x, y = p
    return [((x-1, y-1),rt2),((x-1, y),1), ((x, y-1),1), ((x+1, y),1), ((x, y+1),1),((x-1, y+1),rt2),((x+1, y-1),rt2),((x+1, y+1),rt2)]

def manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def is_blocked_edge(p):
    x, y = p
    return not (x >= 0 and y >= 0 and x < 512 and y < 512)
