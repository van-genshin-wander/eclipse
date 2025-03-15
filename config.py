from datetime import datetime
import numpy as np

G = 6.67349e-11 #m3kg-1s-2
GM_EARTH = 3.98600435436e14
GM_MOON = 4.902800066e12
GM_SUN = 1.3271244004193938e20
OMEGA = 7.292115e-5
J2 = 1.08263e-3  # 地球扁率项
AU = 1.4959787e11  # 天文单位 (m)
YEAR = 365.25 * 86400  # 年 (秒)
start_date = datetime(2025, 3, 5)
init_000pos = np.array([-6.079021099365261E+06, 1.930262876977291E+06, 1.480844368335324E+04])

R_SUN = 6.95700e8
R_MOON = 1.73753e6
R_EARTH = 6.37101e6

earth_pos0 = np.array([-1.428462224208084E+11, 3.680535107737571E+10, 1.595510688398423E+10], dtype=np.float64)  # 地球初始位置
earth_vel0 = np.array([-8.529116798515936E+03, -2.642262762651281E+04, -1.145454880138407E+04], dtype=np.float64)  # 地球初始速度

moon_pos0 = np.array([-1.426212712320577E+11, 3.706060929347127E+10, 1.609611255177771E+10], dtype=np.float64)  
moon_vel0 = np.array([-9.345614418844045E+03, -2.582241418742911E+04, -1.113011779797258E+04], dtype=np.float64)  

state0 = np.concatenate([earth_pos0, earth_vel0, moon_pos0, moon_vel0])