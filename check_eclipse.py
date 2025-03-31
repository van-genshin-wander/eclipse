import numpy as np
import math
from config import *

def get_homothetic_center(r_sun, r_target, pos_target:np.ndarray):
    #内外位似中心，用于计算光锥顶点
    inner = r_sun / (r_sun + r_target) * pos_target
    outer = r_sun / (r_sun - r_target) * pos_target
    return (inner, outer)

def check_cone_inner(center:np.ndarray, pos_moon:np.ndarray, r_moon, pos_earth:np.ndarray, r_earth, same:bool=True):
    #检测地球与内位似中心的光锥是否有交导致日食
    #same为要求交点与月球是否在锥顶同侧
    moon_direct = pos_moon - center
    earth_direct = pos_earth - center
    norm_earth_direct = np.linalg.norm(earth_direct)
    norm_moon_direct = np.linalg.norm(moon_direct)
    if norm_earth_direct <= r_earth:
        return False
    if not same: moon_direct = -moon_direct
    if same:    #防止地球在月球与太阳之间也被判定为与光锥相交
        if abs(earth_direct @ (moon_direct)) < norm_moon_direct ** 2:
            return False
    angle_0 = math.asin(r_moon/norm_moon_direct)
    angle_earth = math.acos(moon_direct@(earth_direct)/(norm_earth_direct * norm_moon_direct))
    angle_1 = math.asin(r_earth/norm_earth_direct)
    if abs(angle_earth) < angle_0:
        return True
    if max(angle_earth - angle_1, 0) < angle_0:
        return True
    return False

def check_cone_outer(center:np.ndarray, pos_moon:np.ndarray, r_moon, pos_earth:np.ndarray, r_earth):
    #检测地球与外位似中心的光锥是否有交导致日食
    #same为要求交点与月球是否在锥顶同侧
    moon_direct = pos_moon - center
    earth_direct = pos_earth - center
    norm_earth_direct = np.linalg.norm(earth_direct)
    norm_moon_direct = np.linalg.norm(moon_direct)
    ans_same = False; ans_diff = False
    if norm_earth_direct <= r_earth:
        # if same: return True
        # else: return False
        return (True, False)
    # if same:    
    # if earth_direct @ (moon_direct) > norm_moon_direct**2:
    #     ans_same = False
    # if not same: moon_direct = -moon_direct
    angle_0 = math.asin(r_moon/norm_moon_direct)
    angle_earth = math.acos(moon_direct@(earth_direct)/(norm_earth_direct * norm_moon_direct))
    angle_1 = math.asin(r_earth/norm_earth_direct)
    if abs(angle_earth) < angle_0:
        ans_same = True
    if max(angle_earth - angle_1, 0) < angle_0:
        ans_same = True
    if earth_direct @ (moon_direct) > norm_moon_direct**2: #防止地球在月球与太阳之间也被判定为与光锥相交
        ans_same = False
    angle_earth = np.pi - angle_earth
    if abs(angle_earth) < angle_0:
        ans_diff = True
    if max(angle_earth - angle_1, 0) < angle_0:
        ans_diff = True
    return (ans_same, ans_diff)

def check_solar_eclipse(pos_moon, pos_earth, r_moon=R_MOON, r_earth=R_EARTH, r_sun=R_SUN):
    inner, outer = get_homothetic_center(r_sun, r_moon, pos_moon)
    ans = set()
    c1, c2 = check_cone_outer(outer, pos_moon, r_moon, pos_earth, r_earth)
    if c1:
        ans.add("日全食")
    if c2:
        ans.add("日环食")
    if check_cone_inner(inner, pos_moon, r_moon, pos_earth, r_earth, same=True):
        ans.add("日偏食")
    return ans

def check_lunar_eclipse(pos_moon, pos_earth, r_moon=R_MOON, r_earth=R_EARTH, r_sun=R_SUN):
    pre_ans = check_solar_eclipse(pos_moon=pos_earth, pos_earth=pos_moon, r_moon=r_earth, r_earth=r_moon, r_sun=r_sun)
    ans = set(s.replace('日', '月') for s in pre_ans)
    return ans

# def check_eclipse_point(pos_moon, pos_earth, pos_observer, r_moon=R_MOON, r_earth=R_EARTH):
#     ans = set()
#     # if pos_observer @ pos_earth > pos_earth @ pos_earth:
#     #     return ans
#     inner, outer = get_homothetic_center(R_SUN, R_MOON, pos_moon)
#     inner_moon = pos_moon - inner
#     inner_earth = pos_earth - inner
#     inner_observer = pos_observer - inner
#     norm_inner_moon = np.linalg.norm(inner_moon)
#     angle_0 = math.asin(r_moon / norm_inner_moon)
#     angle_1 = math.acos(inner_observer @ inner_moon/(norm_inner_moon * np.linalg.norm(inner_observer)))
#     if angle_1 < angle_0:
#         if inner_observer @ inner_moon > norm_inner_moon ** 2 and inner_observer @ inner_earth > inner_observer @ inner_observer:
#             ans.add("日偏食")
    
#     outer_moon = pos_moon - outer
#     outer_earth = pos_earth - outer
#     outer_observer = pos_observer - outer
#     norm_outer_moon = np.linalg.norm(outer_moon)
#     angle_0 = math.asin(r_moon / norm_outer_moon)
#     angle_1 = math.acos(outer_observer @ inner_moon/(norm_outer_moon * np.linalg.norm(outer_observer)))
#     if angle_1 < angle_0:
#         if outer_observer @ outer_moon < norm_outer_moon ** 2 and outer_observer @ outer_earth < outer_observer @ outer_observer:
#             ans.add("日全食")
#     angle_1 = np.pi - angle_1
#     if angle_1 < angle_0:
#         if outer_observer @ outer_earth < outer_observer @ outer_observer:
#             ans.add("日环食")
#     return ans

def check_eclipse_point(pos_moon, pos_earth, pos_observer, r_moon=R_MOON, r_earth=R_EARTH):
    ans = set()
    ob_moon = pos_moon - pos_observer
    ob_sun = - pos_observer
    ob_earth = pos_earth - pos_observer
    norm_om = np.linalg.norm(ob_moon)
    norm_os = np.linalg.norm(ob_sun)
    norm_oe = np.linalg.norm(ob_earth)
    angle_moon = math.asin(R_MOON / norm_om)
    angle_sun = math.asin(R_SUN / norm_os)
    if math.acos(ob_earth @ ob_sun / (norm_oe * norm_os)) < 0.5 * np.pi:
        return ans
    angle_0 = math.acos(ob_moon @ ob_sun / (norm_om * norm_os))
    if angle_0 < angle_moon + angle_sun:
        ans.add("日偏食")
    if angle_0 + angle_moon < angle_sun:
        ans.add("日环食")
    if angle_0 + angle_sun < angle_moon:
        ans.add("日全食")
    return ans


if __name__ == '__main__':
    for i in range(400):
        theta = 2 * math.pi * i / 400
        delta = np.array([3.844e8 * math.cos(theta), 3.844e8 * math.sin(theta), 0])
        pos_earth = np.array([1.496e11, 0, 0])
        pos_moon = pos_earth + delta
        # print(pos_moon, pos_earth)
        inner, outer = get_homothetic_center(R_SUN, R_MOON, pos_moon)
        # print(inner, outer)
        # print(check_cone_inner(inner, pos_moon, R_MOON, pos_earth, R_EARTH, same=False))
        print(check_solar_eclipse(pos_moon, pos_earth))