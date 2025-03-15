import numpy as np
import math

from config import *
from caculate import *

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation

def get_pos(earth_pos, t, lat, lon):
    theta = OMEGA * t + lat / 180 * np.pi
    phi = lon / 180 * np.pi
    c_t, s_t = math.cos(theta), math.sin(theta)
    c_p, s_p = math.cos(phi), math.sin(phi)
    x, y, z = init_000pos[:]
    xx = c_p * (c_t * x - s_t * y)
    yy = c_p * (c_t * y + s_t * x)
    zz = R_EARTH * s_p
    # print(xx, yy, zz)
    return np.array([xx, yy, zz]) + earth_pos

def check_earth(earth_pos, moon_pos, t):
    ans = [[None for i in range(180)] for j in range(360)]
    for i in range(360):
        for j in range(180):
            pos = get_pos(earth_pos, t, i - 180, j - 90)
            ans[i][j] = check_eclipse_point(moon_pos, earth_pos, pos)
    return ans

def draw(t):
    t_span = (0, t)
    t_eval = np.linspace(0, t, t//60)
    _, _, state = get_solution(t_span, t_eval, state0)
    earth_pos = state[0:3]; moon_pos = state[6:9]
    ans = check_earth(earth_pos, moon_pos, t)
    # print(ans)
    print(earth_pos)
    print(get_pos(earth_pos, t, 30, 30) - earth_pos)

    # 创建地图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 添加地图特征
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # 设置地图范围
    ax.set_global()

    for i in range(360):
        for j in range(180):
            if ans[i][j]:
                polygon = np.array([
                    [i-180,j-90],   # 点1
                    [i-180,j-90+1],  # 点2（跨越日期变更线）
                    [i-180+1,j-90+1], # 点3
                    [i-180+1,j-90],  # 点4
                    [i-180,j-90]    # 闭合多边形
                ])
                # 初始化多边形绘制
                polygon_plot, = ax.plot(polygon[:, 0], polygon[:, 1], color='red', linewidth=2, marker=None, transform=ccrs.PlateCarree())
                polygon_fill = ax.fill(polygon[:, 0], polygon[:, 1], color='red', alpha=0.3, transform=ccrs.PlateCarree())
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=20) 

    draw(2112000)   #3月29日10:38

    # t_span = (0, 86400)
    # t_eval = np.linspace(0, 86400, 86400) 

    # _, _, state = get_solution(t_span, t_eval, state0)
    # pos_earth = state[0:3].copy()
    # print(earth_pos0 + init_000pos)
    # print(get_pos(pos_earth, 86400, 0, 0) )
    # print(line_inter_earth(np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([0, 0, 100])))