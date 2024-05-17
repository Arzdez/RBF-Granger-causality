import rbf_net as rbf
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from functools import partial


def _processing(path: str,*, N_centre = 15, Tau = 100, step = 20000):
    
    path = str(path)
    print("Обрабатывается файл:" , path)
    
    chanels =[
        "time",
        "Cd R",
        "Cd L",
        "Cx occip R",
        "Cx occip L",
    ]
    
    Data = np.loadtxt(path, comments="#")
    Master_1 = np.vstack((Data[:, 3], Data[:, 0])).T
    Slave_1 = np.vstack((Data[:, 4], Data[:, 0])).T
    
    print(f"Размер - {np.shape(Master_1)}\n Колличество центров - {N_centre}\n Дальность прогноза - {Tau}\n Шаг - {step}")
    PI = []
    PI_revers = []
    Time_line = []



    if len(Data) < 610000:
        L = len(Data)
    else:
        L = 610000
    for i in range(0, L - step, step):
#В одну сторону
#-------------------------------------------------------------------------
        
        model = rbf.Rbf_net(Slave_1[i : i + step], Master_1[i : i + step], N_centre)
        model.solo_pred(Tau)
        model.couple_pred(Tau)
        
        pi_step = 1 - ( model.err_cup / model.err_solo)
        if pi_step < 0:
            pi_step = 0
        PI.append(pi_step)
#-------------------------------------------------------------------------

#В другую сторону
#-------------------------------------------------------------------------
        model = rbf.Rbf_net(Master_1[i : i + step], Slave_1[i : i + step], N_centre)
        model.solo_pred(Tau)
        model.couple_pred(Tau)
        
        pi_revers_step = 1 - ( model.err_cup / model.err_solo)
        if pi_revers_step < 0:
            pi_revers_step = 0
        PI_revers.append(pi_revers_step)
#-------------------------------------------------------------------------
        Time_line.append(Data[i,0])
        

    PI = np.asarray(PI)
    PI_revers = np.asarray(PI_revers)
    
    Sum_PI = np.vstack((PI, PI_revers)).T
    print(np.shape(Sum_PI))
    print(f"{path} - Сохраняется")
    
    np.savetxt(path.replace(".txt", f"_casuality_N_cen={N_centre}.txt"), Sum_PI)
    np.savetxt(path.replace(".txt", "_casuality Cx_R on Cx_L.txt"), PI)
    np.savetxt(path.replace(".txt", "_casuality Cx_L on Cx_R.txt"), PI_revers)
    
    plt.figure(figsize=(18,9))
    plt.title(f"Chanel Cx_R and Cx_L\n window width = {step/1000} ")
    plt.plot(Time_line, PI, label = "Cx_R -> Cx_L", marker = 'o', linestyle = '--', color = "green",)
    plt.plot(Time_line, PI_revers, label = "Cx_R <- Cx_L", marker = 'o', linestyle = '--', color = "red")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path.replace(".txt", f" N = {N_centre}.pdf")))
    plt.close()

    
    
    

def granger_parallel(path,*, N_centr = 10, Tau = 100, step = 20000, names = "rd"):
    
    path_tuple = []
    
    
    if not os.path.exists(path):
        raise FileExistsError("invalid path")
    os.chdir(path)
    
    if names == "rd":
        names = "RD_Cx_L+_clear.txt"
        print("Считаю для рд")
    else:
        names = "baseline_clear.txt"
        print("Считаю для фона")


    for root, dirs, files in os.walk(".", topdown=False):
            for name in files:
                if names in name:
                    path_tuple.append(os.path.join(root, name))

    
    part_processing = partial(_processing, N_centre = N_centr, Tau = Tau, step = step)
    print(len(path_tuple))
    if len(path_tuple) > 10:
        threads = 10
    else:
        threads = len(path_tuple)

    with mp.Pool(processes = threads) as worker:
        worker.map(part_processing, path_tuple)



