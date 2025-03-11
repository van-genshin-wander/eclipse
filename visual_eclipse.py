import numpy as np
import math
from config import *


def line_inter_earth(center:np.ndarray, direct: np.ndarray, earth_pos:np.ndarray):
    '''根据光线中心和方向向量返回与地球两个交点或None'''
    center_earth = earth_pos - center
    # norm(t * axis - center_earth) ** 2 == R_EARTH ** 2
    A = direct @ direct
    B = direct @ center_earth
    C = center_earth @ center_earth - R_EARTH ** 2
    delta = B ** 2 - A * C 
    if delta >= 0:
        t1 = (B + math.sqrt(delta))/ A
        t2 = (B - math.sqrt(delta))/ A
        return (center + t1 * direct, center + t2 * direct)
    return None

if __name__ == "__main__":
    
    print(line_inter_earth(np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([0, 0, 100])))