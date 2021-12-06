import numpy as np
import pandas as pd
from scipy import signal
import os
import matplotlib.pyplot as plt
import time

time_start = time.perf_counter()
path = 'D:\\Document\\python'
dirs = os.listdir(path)

D = 0.02
L = 0.325
fn = 1.88
c1 = 0.256426
c2 = 0.238191
fs = 2000
cutf = 8
p = 6
t = np.linspace(0, 25, 50000)
wn = 2*cutf/fs

txt = []
AD = []
FLL = []
FDD = []
FL_mean = []
FL_rms = []
FD_mean = []
FD_rms = []

for file in dirs:
    if os.path.splitext(file)[1] == ".txt":
        txt.append(file)
    if os.path.splitext(file)[1] == ".xlsx":
        xlsx = file

for i in range(0, len(txt), 2):
    file1 = os.path.join(path, txt[i+1])
    disp = pd.read_csv(file1, delimiter='\t')
    disp = disp['E1'][:-1]
    d, c = signal.butter(p, wn, 'lowpass')
    disp = signal.filtfilt(d, c, disp, axis=0)
    disp -= disp.mean(axis=0)
    disp = disp*0.005
    # b = (sum((disp/D)**2)/len(disp)*2)**0.5
    b = np.sqrt(np.var(disp/D)*2)
    AD.append(b)

    dbdt = np.gradient(disp)/np.gradient(t)
    ddbdt = np.gradient(dbdt)/np.gradient(t)

    file2 = os.path.join(path, txt[i])
    E = pd.read_csv(file2, delimiter='\t')
    E = np.array(E[['E1', 'E2']][:-1])
    FL = signal.filtfilt(d, c, E[:, 0])/c1+0.557*ddbdt
    FD = signal.filtfilt(d, c, E[:, 1])/c2
    # FLL.append(FL)
    # FDD.append(FD)
    FL_mean.append(FL.mean(axis=0))
    FL_rms.append(np.sqrt(np.var(FL)))
    FD_mean.append(FD.mean(axis=0))
    FD_rms.append(np.sqrt(np.var(FD)))

f = np.array(AD).reshape(-1, 3)
f = f.mean(axis=1)
g = np.array(FL_mean).reshape(-1, 3)
g = g.mean(axis=1)  # Lift_time_mean
h = np.array(FL_rms).reshape(-1, 3)
h = h.mean(axis=1)  # Lift_fluctuate
m = np.array(FD_mean).reshape(-1, 3)
m = m.mean(axis=1)  # Drag_time_mean
n = np.array(FD_rms).reshape(-1, 3)
n = n.mean(axis=1)  # Drag_fluctuate

o = pd.read_excel(os.path.join(path, xlsx))
u = o['u_p 4'][1:]
ur = u/fn/D
uu = 0.5*1000*D*L*u*u

CL_mean = (g[1:]-g[0])/uu
CL_fluctuate = h[1:]/uu
CD_mean = -(m[1:]-m[0])/uu
CD_fluctuate = n[1:]/uu

# print(f)
# print(CL_mean)
# print(CD_mean)
# print(CL_fluctuate)
# print(CD_fluctuate)

# plt.plot(t,disp)
# plt.show()

data = {'ur': ur, 'A/D': f[1:], 'CL': CL_mean,
        "CL'": CL_fluctuate, 'CD': CD_mean, "CD'": CD_fluctuate}
output = pd.DataFrame(data)
outputpath = os.path.join(path, 'output.csv')
output.to_csv(outputpath, index=False)


time_end = time.perf_counter()
print('Running time: %s Seconds' % (time_end-time_start))
