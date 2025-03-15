import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory

from check_eclipse import *
import time
from datetime import datetime, timedelta
from memory_profiler import profile
from config import *
# 天文常数


def num2date(first_date: datetime, time):
    #把tdb时间first_date之后time秒的utc时间算出来
    time = time - 69
    delta = timedelta(seconds=time)
    end_date = first_date + delta
    return f"{end_date.year}-{end_date.month}-{end_date.day} {end_date.hour}:{end_date.minute}:{end_date.second}"


# 微分方程组
def n_body(t, state):
    # 解析状态变量
    earth_pos = state[0:3]
    earth_vel = state[3:6]
    moon_pos = state[6:9]
    moon_vel = state[9:12]
    
    # 计算加速度
    def acceleration(pos, main_gm):
        r = np.linalg.norm(pos)
        # r_safe = max(r, 1e-10)
        return - main_gm * pos / r**3
    
    def get_j2_acceleration(pos_moon):#传入之前已处理为相对位置
        # """计算 J2 项对月球的加速度"""
        r = np.linalg.norm(pos_moon)  # 月球到地球的距离
        z = pos_moon[-1] 
        r_squared = r**2
        r_fifth = r**5
        
        # J2 项加速度公式
        a_j2 = -1.5 * J2 * GM_EARTH * R_EARTH**2 / r_fifth
        term1 = (5 * z**2 / r_squared - 1) * pos_moon
        term2 = -2 * z * np.array([0, 0, 1])
        return -a_j2 * (term1 + term2)
    
    # 地球加速度（主要来自太阳）
    earth_acc = acceleration(earth_pos, GM_SUN) - acceleration(moon_pos - earth_pos, GM_MOON)
    
    # 月球加速度（来自太阳和地球）
    moon_acc_sun = acceleration(moon_pos, GM_SUN)
    moon_acc_earth = acceleration(moon_pos - earth_pos, GM_EARTH) +  get_j2_acceleration(moon_pos - earth_pos)
    moon_acc = moon_acc_sun + moon_acc_earth 
    
    return np.concatenate([earth_vel, earth_acc, moon_vel, moon_acc])

def get_solution(t_span, t_eval, state0):

# 数值积分
    solution = solve_ivp(n_body, t_span, state0, t_eval=t_eval, 
                        method='DOP853', rtol=1e-12, atol=1e-14)

    # 提取结果
    earth_traj = solution.y[0:3, :].copy()  # 地球轨迹
    moon_traj = solution.y[6:9, :].copy()   # 月球轨迹
    final_state = solution.y[:, -1].copy()
    del solution
    return earth_traj, moon_traj, final_state

# 转换为天文单位

def draw(earth_traj, moon_traj):
    earth_traj_au = earth_traj / AU
    moon_traj_au = moon_traj / AU

    # print(earth_traj[:, -1])

    # 可视化
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制太阳
    ax.scatter([0], [0], [0], color='yellow', s=200, label='Sun')

    # 绘制地球轨道
    ax.plot(earth_traj_au[0], earth_traj_au[1], earth_traj_au[2], 
            color='blue', alpha=0.7, label='Earth Orbit')

    # 绘制月球轨道（放大显示）
    scale_factor = 50  # 放大倍数以显示月球轨道细节
    moon_relative_au = (moon_traj - earth_traj) / AU * scale_factor
    ax.plot(earth_traj_au[0] + moon_relative_au[0], 
            earth_traj_au[1] + moon_relative_au[1], 
            earth_traj_au[2] + moon_relative_au[2], 
            color='gray', alpha=0.3, label=f'Moon Orbit (x{scale_factor})')

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.legend()
    ax.set_title('Earth and Moon Trajectories (20 Years, Moon Orbit Scaled)')
    plt.show()

def merge_ans(ans, lis):
    if not lis: pass
    if not ans: ans.append(lis)
    elif ans[-1][1] == lis[0] - 1:
        ans[-1] = [ans[-1][0], lis[1], ans[-1][2].union(lis[2])]
    else:
        ans.append(lis)

def check_lis(start_id, end_id, shape):
    ans = []
    shm_earth = shared_memory.SharedMemory(name='earth', create=False)
    shm_moon = shared_memory.SharedMemory(name='moon', create=False)
    earth_traj = np.ndarray(shape, np.float64, shm_earth.buf)
    moon_traj = np.ndarray(shape, np.float64, shm_moon.buf)
    for i in range(start_id, end_id):
        flag = check_solar_eclipse(moon_traj[:, i], earth_traj[:, i])
        if flag:
            merge_ans(ans, [i, i, flag])
    shm_earth.close()
    shm_moon.close()
    return ans

def check(earth_traj:np.ndarray, moon_traj:np.ndarray, max_workers = 16, d = None):
    print(earth_traj.shape)
    _, length = earth_traj.shape
    ans = []
    if d == None:
        d = length//max_workers
    shm_earth = shared_memory.SharedMemory(name='earth', create=True, size=earth_traj.nbytes)
    array_earth = np.ndarray(earth_traj.shape, dtype=earth_traj.dtype, buffer=shm_earth.buf)
    array_earth[:] = earth_traj
    shape = earth_traj.shape
    del earth_traj
    shm_moon = shared_memory.SharedMemory(name='moon', create=True, size=moon_traj.nbytes)
    array_moon = np.ndarray(moon_traj.shape, dtype=moon_traj.dtype, buffer=shm_moon.buf)
    array_moon[:] = moon_traj
    del moon_traj
    print("shared")
    ans = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = [executor.submit(check_lis, i, min(i+d, length), shape) for i in range(0, length, d)]
        # ans = []
        # for result in (results):
        #     ans.append(result.result())
        ans = []
        for result in results:
            for lis in result.result():
                merge_ans(ans, lis)
        del array_earth, array_moon
        shm_earth.close()
        shm_earth.unlink()
        shm_moon.close()
        shm_moon.unlink()
        print("closed")
        return ans

def main(total_time=50 * 365 * 86400, batch_time=50 * 365 * 86400):
    earth_pos0 = np.array([-1.428462224208084E+11, 3.680535107737571E+10, 1.595510688398423E+10], dtype=np.float64)  # 地球初始位置
    earth_vel0 = np.array([-8.529116798515936E+03, -2.642262762651281E+04, -1.145454880138407E+04], dtype=np.float64)  # 地球初始速度

    moon_pos0 = np.array([-1.426212712320577E+11, 3.706060929347127E+10, 1.609611255177771E+10], dtype=np.float64)  
    moon_vel0 = np.array([-9.345614418844045E+03, -2.582241418742911E+04, -1.113011779797258E+04], dtype=np.float64)  

    state0 = np.concatenate([earth_pos0, earth_vel0, moon_pos0, moon_vel0])
    ans = []
    # print(0.5 * GM_EARTH * (earth_vel0 @ earth_vel0) + 0.5 * GM_MOON * (moon_vel0 @ moon_vel0)) 
    # print(- GM_SUN * GM_EARTH /np.linalg.norm(earth_pos0) - GM_SUN * GM_MOON /np.linalg.norm(moon_pos0) - GM_MOON * GM_EARTH /np.linalg.norm(earth_pos0 - moon_pos0))
    # print(0.5 * GM_EARTH * (earth_vel0 @ earth_vel0) + 0.5 * GM_MOON * (moon_vel0 @ moon_vel0)- GM_SUN * GM_EARTH /np.linalg.norm(earth_pos0) - GM_SUN * GM_MOON /np.linalg.norm(moon_pos0) - GM_MOON * GM_EARTH /np.linalg.norm(earth_pos0 - moon_pos0))

    print("start")
    for i in range(0, total_time, batch_time):
        batch = min(batch_time, total_time - i)
        t_span = (0, batch)
        t_eval = np.linspace(0, batch, batch//60) 
        earth_traj, moon_traj, final_state = get_solution(t_span, t_eval, state0)
        earth_vel0 = final_state[3:6]; moon_vel0 = final_state[9:12]
        earth_pos0 = final_state[0:3]; moon_pos0 = final_state[6:9]
        # draw(earth_traj, moon_traj)
        # print(0.5 * GM_EARTH * (earth_vel0 @ earth_vel0) + 0.5 * GM_MOON * (moon_vel0 @ moon_vel0)) 
        # print(- GM_SUN * GM_EARTH /np.linalg.norm(earth_pos0) - GM_SUN * GM_MOON /np.linalg.norm(moon_pos0) - GM_MOON * GM_EARTH /np.linalg.norm(earth_pos0 - moon_pos0))
        # print(0.5 * GM_EARTH * (earth_vel0 @ earth_vel0) + 0.5 * GM_MOON * (moon_vel0 @ moon_vel0)- GM_SUN * GM_EARTH /np.linalg.norm(earth_pos0) - GM_SUN * GM_MOON /np.linalg.norm(moon_pos0) - GM_MOON * GM_EARTH /np.linalg.norm(earth_pos0 - moon_pos0))
        print("solved")
        state0 = final_state
        for lis in check(earth_traj, moon_traj):
            lis[0] += i; lis[1] += i
            merge_ans(ans, lis)
        earth_traj, moon_traj = None, None
        print("deleted")
        state0 = final_state
        

    for lis in ans:
        lis[0] = num2date(start_date, 60 * lis[0])
        lis[1] = num2date(start_date, 60 * lis[1])
        with open("eclipse", 'a', encoding='utf-8') as f:
            f.write(str(lis) + '\n')
    print(ans)



if __name__ == '__main__':
    np.set_printoptions(precision=20) 

    main()
    # earth_pos0 = np.array([-1.428462224208084E+11, 3.680535107737571E+10, 1.595510688398423E+10], dtype=np.float64)  # 地球初始位置
    # earth_vel0 = np.array([-8.529116798515936E+03, -2.642262762651281E+04, -1.145454880138407E+04], dtype=np.float64)  # 地球初始速度

    # moon_pos0 = np.array([-1.426212712320577E+11, 3.706060929347127E+10, 1.609611255177771E+10], dtype=np.float64)  
    # moon_vel0 = np.array([-9.345614418844045E+03, -2.582241418742911E+04, -1.113011779797258E+04], dtype=np.float64)  

    # state0 = np.concatenate([earth_pos0, earth_vel0, moon_pos0, moon_vel0])
    # t_span = (0, 256 * 86400)
    # t_eval = np.linspace(0, 256 * 86400, 256 * 86400) 

    # earth_traj, moon_traj = get_solution(t_span, t_eval, state0)

    # earth, moon = earth_traj[:,2110000], moon_traj[:, 2110000]
    # inner, _ = get_homothetic_center(R_SUN, R_MOON, moon)
    # direct_earth = earth - inner; direct_moon = moon - inner
    # print(np.linalg.norm(direct_moon)**2)
    # print(direct_moon@direct_earth)
    # print(check_cone_inner(inner, moon, R_MOON, earth, R_EARTH))
    # print(check_eclipse(moon, earth))

    # s = time.time()
    # ans = check(earth_traj, moon_traj)
    # e = time.time()
    # print(ans)
    # print(e - s)