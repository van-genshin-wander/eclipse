from caculate import *
from config import *

class GmSun:
    data: float
gmsun = GmSun()
gmsun.data = GM_SUN

def bi_search(l, r, f, tol):
    while r - l > tol:
        m = (l + r) / 2
        ff = f(m)
        if ff > 0: r = m
        elif ff < 0: l = m
        else: return m
        print(l, m, r, ff)
    return (l + r) / 2

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
    earth_acc = acceleration(earth_pos, gmsun.data) - acceleration(moon_pos - earth_pos, GM_MOON)
    
    # 月球加速度（来自太阳和地球）
    moon_acc_sun = acceleration(moon_pos, gmsun.data)
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


def load_data():
    start_time = []
    end_time = []
    form = r"%Y %b %d at %H:%M:%S"
    with open('./training_data/start_time.txt', 'r') as f:
        for l in f:
            start_time.append(datetime.strptime(l.strip(), form))
    with open('./training_data/end_time.txt', 'r') as f:
        for l in f:
            end_time.append(datetime.strptime(l.strip(), form))
    return start_time, end_time

def evaluation(gm_sun):
    gmsun.data = gm_sun
    ans = []
    tim = 30 * 365 * 86400
    print("start")

    earth_pos0 = np.array([-1.428462224208084E+11, 3.680535107737571E+10, 1.595510688398423E+10], dtype=np.float64)  # 地球初始位置
    earth_vel0 = np.array([-8.529116798515936E+03, -2.642262762651281E+04, -1.145454880138407E+04], dtype=np.float64)  # 地球初始速度

    moon_pos0 = np.array([-1.426212712320577E+11, 3.706060929347127E+10, 1.609611255177771E+10], dtype=np.float64)  
    moon_vel0 = np.array([-9.345614418844045E+03, -2.582241418742911E+04, -1.113011779797258E+04], dtype=np.float64)  

    state0 = np.concatenate([earth_pos0, -earth_vel0, moon_pos0, -moon_vel0])
    for i in range(0,tim , tim):
        batch = min(tim, tim - i)
        t_span = (0, batch)
        t_eval = np.linspace(0, batch, batch//60) 
        earth_traj, moon_traj, final_state = get_solution(t_span, t_eval, state0)
        earth_vel0 = final_state[3:6]; moon_vel0 = final_state[9:12]
        earth_pos0 = final_state[0:3]; moon_pos0 = final_state[6:9]
        print("solved")
        state0 = final_state
        for lis in check(earth_traj, moon_traj):
            lis[0] += i; lis[1] += i
            merge_ans(ans, lis)
        earth_traj, moon_traj = None, None
        print("deleted")
    score = 0
    start_lis, end_lis = load_data()
    for i, lis in enumerate(ans):
        lis[0] = start_date - timedelta(seconds=60 * lis[0] + 69)
        lis[1] = start_date - timedelta(seconds=60 * lis[1] + 69)
        temp = int((lis[0] - end_lis[i]).total_seconds()/60) 
        temp += int((lis[1] - start_lis[i]).total_seconds()/60)
        ab = abs((lis[0] - end_lis[i]).total_seconds()) + abs((lis[1] - start_lis[i]).total_seconds())
        if ab > 2000:
            print(end_lis[i])
        score += temp
    return -score

def train(origin_err, tol):
    gmsun.data = bi_search(gmsun.data - origin_err, gmsun.data + origin_err, evaluation, tol)
    return gmsun.data

if __name__ == '__main__':
    print(earth_pos0)
    state0 = np.concatenate([earth_pos0, -earth_vel0, moon_pos0, -moon_vel0])
    # print(evaluation(gmsun.data + 1e15, state0))
    train(1e15, 1e10)