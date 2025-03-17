import numpy as np
import math

from config import *
from caculate import *

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

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

def generate_data(t, earth_traj, moon_traj):
    earth_pos = earth_traj[:, t]
    moon_pos = moon_traj[:, t]
    ans = check_earth(earth_pos, moon_pos, t)
    for i in range(360):
        for j in range(180):
            if "日环食" in ans[i][j]: ans[i][j] = 3
            elif "日全食" in ans[i][j]: ans[i][j] = 2
            elif "日偏食" in ans[i][j]: ans[i][j] = 1
            else: ans[i][j] = 0
    arr = np.array(ans, dtype=int)
    arr_1 = arr[:180, :].copy(); arr_2 = arr[180:, :].copy()
    return np.concatenate((arr_2, arr_1))

# 初始化地图和颜色映射
def setup_map():
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 添加地理特征
    ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    
    # 创建自定义颜色映射（区域显示为红色半透明，背景透明）
    cmap = mcolors.ListedColormap(['none', 'red', 'blue', 'yellow'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    
    return fig, ax, cmap, norm

def generate_ani(earth_traj, moon_traj, start, end):
    # 设置经纬度网格
    lon_res = 1  # 经度分辨率（度）
    lat_res = 1  # 纬度分辨率（度）
    lon_grid = np.arange(-180, 180, lon_res)
    lat_grid = np.arange(-90, 90, lat_res)

    # 初始化地图
    fig, ax, cmap, norm = setup_map()
    time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, 
                        ha='right', va='top', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))

    # 初始数据
    initial_data = generate_data(0, earth_traj, moon_traj)
    mesh = ax.pcolormesh(lon_grid, lat_grid, initial_data.T,  # 需要转置
                        cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree(),
                        alpha=0.5, shading='auto')

    # 动画更新函数
    def update(frame):
        # 生成新数据
        new_data = generate_data(frame, earth_traj, moon_traj)
        
        # 更新网格数据
        mesh.set_array(new_data.T.ravel())  # 需要转置并展平
        
        # 更新时间
        current_time = start_date + timedelta(seconds=frame * 60 +start - 60)
        time_text.set_text(current_time.strftime("%Y-%m-%d %H:%M"))
        
        return mesh, time_text

    # 创建动画
    ani = FuncAnimation(fig, update, frames=(end - start)//60,  # 3小时数据（每秒一帧）
                    interval=50, blit=True)
    # plt.show()
    # 导出为GIF
    print("Exporting GIF...")
    ani.save(f"dynamic_region_{start}_{end}.gif", writer="pillow", fps=10, dpi=100)
    print("GIF saved as dynamic_region.gif")

def main(start, end):
    t_span = (0, end)
    t_eval = np.linspace(start, end, (end - start)//60)
    earth_traj, moon_traj, _ = get_solution(t_span, t_eval, state0)
    generate_ani(earth_traj, moon_traj, start, end)


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

    # draw(2112000)   #3月29日10:38
    main(2112000, 2118000)
    # t_span = (0, 86400)
    # t_eval = np.linspace(0, 86400, 86400) 

    # _, _, state = get_solution(t_span, t_eval, state0)
    # pos_earth = state[0:3].copy()
    # print(earth_pos0 + init_000pos)
    # print(get_pos(pos_earth, 86400, 0, 0) )
    # print(line_inter_earth(np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([0, 0, 100])))