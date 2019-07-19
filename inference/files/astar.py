import time
import cv2
import numpy as np
import scipy.interpolate
#from pycubicspline import * 
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc as misc

def AStar2(start, goal, neighbor_nodes, distance, cost_estimate, img2):
    path_img=Image.fromarray(np.uint8(img2))
    global path_pixels_lat
    path_pixels_lat = path_img.load()

    def reconstruct_path(came_from, current_node):
        path = []
        if came_from is not None:
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
        if len(path) >=2:
            return list(path)
        else:
            return []

    g_score = {start: 0}
    f_score = {start: g_score[start] }
    openset = {start}
    closedset = set()
    came_from = {start: None}

    while openset:
        current = min(openset, key=lambda x: f_score[x])
        if is_blockedB(current) == True:
            goal = current
            #print 'GOT SEC TIP'
        if current == goal:
            return reconstruct_path(came_from, goal)
        openset.remove(current)
        closedset.add(current)
        for neighbor in neighbor_nodes(current):
            if neighbor in closedset:
                continue
            if neighbor not in openset:
                openset.add(neighbor)
            tentative_g_score = g_score[current] + distance(current, neighbor)
            if tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score
    return []


def AStar(start, goal, neighbor_nodes, distance, cost_estimate, decoded):

    path_img=Image.fromarray(np.uint8(decoded))
    global path_pixels
    path_pixels = path_img.load()
    


    def reconstruct_path(came_from, current_node):
        path = []
        while current_node is not None:
            path.append(current_node)
            current_node = came_from[current_node]
        return list(reversed(path))
    g_score = {start: 0}
    f_score = {start: g_score[start] + cost_estimate(start, goal)}
    openset = {start}
    closedset = set()
    came_from = {start: None}
    while openset:
        current = min(openset, key=lambda x: f_score[x])
        if current == goal:
            return reconstruct_path(came_from, goal)
        openset.remove(current)
        closedset.add(current)
        for neighbor in neighbor_nodes(current):
            if neighbor in closedset:
                continue
            if neighbor not in openset:
                openset.add(neighbor)
            tentative_g_score = g_score[current] + distance(current, neighbor)
            if tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + cost_estimate(neighbor, goal)

    return []
def von_neumann_neighbors(p):
    x, y = p
    neighbors = [(x-1, y-1),(x-1, y), (x, y-1), (x+1, y), (x, y+1),(x-1, y+1),(x+1, y-1),(x+1, y+1)]
    #neighbors = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]
    #print (neighbors)
    return [p for p in neighbors if not is_blocked(p)]

def von_neumann_neighborsA(p):
    x, y = p
    neighbors = [(x-1, y-1),(x-1, y), (x, y-1), (x+1, y), (x, y+1),(x-1, y+1),(x+1, y-1),(x+1, y+1)]
    #neighbors = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]
    #print (neighbors)
    return [p for p in neighbors if not is_blockedA(p)]
def manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
def squared_euclidean(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def is_blocked(p):
    x,y = p
    #print (x,y)
    pixel = path_pixels[x,y]
    if any(c < 1 for c in pixel):
        return True
def is_blockedA(p):
    x,y = p
    pixel = path_pixels_lat[x,y]
    if any(c < 1 for c in pixel ):
        return True
def is_blockedB(p):
    x,y = p
    #print x,y
    pixel = path_pixels_lat[x,y]
    #print pixel
    if any(c == 128 for c in pixel):
        return True

