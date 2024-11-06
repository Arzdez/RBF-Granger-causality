import numpy as np
import matplotlib.pyplot as plt
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1,r"E:\Work\Программы\eeg_filter\pgc")
import _txt_edit.coma_to_dot as ctd
from rbf_net import Rbf_net
from sklearn import metrics

def rbf_calc(kardio_date,
             n_kernel,
             tau,
             shell_number,
             step = 4000):

    time = kardio_date[:,0]
    x_date = kardio_date[:,shell_number]
    date_len = len(kardio_date)
    for_predict= np.vstack((x_date,time)).T
    
    err_progression=[]

    for i in range(1,for_predict.shape[0]):
        for j in range(for_predict.shape[1]):
            for_predict[i][j] = (for_predict[i][j] * ( 3 / 65536) - 1.5)
            
    for i in range(1, date_len - step, step):
        print("New 5000")
        net = Rbf_net(for_predict[i : i + step],
                      for_predict[i : i + step],
                      n_kernel)
        recun = net.solo_pred(tau=tau)
        err_progression.append(np.sqrt(metrics.mean_squared_error(for_predict[i : i + step-tau-1, 0], recun[1:,0])))
        print("Done")

    return 0


if __name__ == "__main__":
    step=5000
    n_kernel=10
    tau = 10
    shell_number=3
    ctd(r"E:\Work\Малюски\Моллюски\Данные тиомочевина 10 гл\28txt\2020_2_28_13_05.dat_R12.txt",comment_line=0)
    shellfish_ECG = np.loadtxt(r"E:\Work\Малюски\Моллюски\Данные тиомочевина 10 гл\28txt\2020_2_28_13_05.dat_R12.txt")

    rbf_calc(shellfish_ECG,
             n_kernel=n_kernel,
             tau=tau,
             shell_number=shell_number,
             step=step,)
