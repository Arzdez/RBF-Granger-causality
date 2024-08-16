import numpy as np
import matplotlib.pyplot as plt
import os
from granger import granger_parallel
import time

def meaner(path_to_data, path_to_baseline, n_cen):
    

    baseline_files = []
    data_files = []
    Time = np.linspace(10, 600, 30, endpoint=False,)

    supertime = np.vstack([Time ,Time ,Time ,Time, Time, Time ,Time, Time, Time, Time ])

    mean_rd_l_to_r = np.empty(30)
    mean_rd_r_to_l = np.empty(30)

    max_rd_l_to_r = np.empty(30)
    min_rd_l_to_r = np.empty(30)

    max_rd_r_to_l = np.empty(30)
    min_rd_r_to_l = np.empty(30)

    
    
    os.chdir(path_to_baseline)
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if f"baseline_clear_casuality_N_cen={n_cen}" in name:
                baseline_files.append(os.path.join(root, name))
    
    base_l_to_r_mean = 0 
    base_l_to_r_mean1 = 0

    base_r_to_l_mean = 0
    base_r_to_l_mean1 = 0

    base_mean_l_to_r = []
    base_mean_r_to_l = []

    count11 = 0
    count22 = 0
    for j, i in enumerate(baseline_files):
        tmp_data = np.loadtxt(i)

        count1=0
        base_l_to_r_mean = 0
        for i in tmp_data[:,1]:
            if i < 0.04:
                base_l_to_r_mean += i
                count1+=1
                
                base_l_to_r_mean1 += i
                count11+=1

        base_mean_l_to_r.append(base_l_to_r_mean/count1)

        count2=0
        base_r_to_l_mean = 0

        for i in tmp_data[:,0]:
            if i < 0.04:
                base_r_to_l_mean += i
                count2+=1

                base_r_to_l_mean1 += i
                count22+=1

        
        base_mean_r_to_l.append(base_r_to_l_mean/count2)


    base_l_to_r_mean1 /= count11
    base_r_to_l_mean1 /= count22

    print(base_mean_l_to_r)
    print(base_mean_r_to_l)


    
    os.chdir(path_to_data)
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if f"RD_Cx_L+_clear_casuality_N_cen={n_cen}.txt" in name:
                data_files.append(os.path.join(root, name))
                

    data_l_to_r = np.empty((30, len(data_files)))
    data_r_to_l = np.empty((30, len(data_files)))
    index_of_min = []
    index_of_max = []
    name_of_min = []
    name_of_max = []
    
    for j, i in enumerate(data_files):
        tmp_data = np.loadtxt(i)

        print(base_mean_l_to_r[j])
        data_l_to_r[:,j] = tmp_data[:,1] / base_mean_l_to_r[j]
        data_r_to_l[:,j] = tmp_data[:,0] / base_mean_r_to_l[j]

    

    for i in range(data_l_to_r.shape[0]):

        max_rd_l_to_r[i] =  data_l_to_r[i,:].max()
        min_rd_l_to_r[i] =  data_l_to_r[i,:].min()


        max_rd_r_to_l[i] = data_r_to_l[i,:].max()
        min_rd_r_to_l[i] = data_r_to_l[i,:].min()

        index_of_min.append(data_r_to_l[i,:].argmin())
        index_of_max.append(data_r_to_l[i,:].argmax())
        #index_of_min.append('\n')

    for i in index_of_min[4:14]:
        name_of_min.append(f"{data_files[i]}\n")
    
    for i in index_of_max[4:14]:
        name_of_max.append(f"{data_files[i]}\n")
    
    print(name_of_min)
    ful_name_min =''.join(name for name in name_of_min)
    ful_name_max =''.join(name for name in name_of_max)


    with open(f"name_of_mean_N={n_cen}", 'w') as file:
        file.write(ful_name_min)
        file.close()

    with open(f"name_of_max_N={n_cen}", 'w') as file:
        file.write(ful_name_max)
        file.close()

    for i in range(data_l_to_r.shape[0]):

        mean_rd_l_to_r[i] = data_l_to_r[i,:].sum() / data_l_to_r.shape[1]
        mean_rd_r_to_l[i] = data_r_to_l[i,:].sum() / data_l_to_r.shape[1]

    #ЭТО НУЖНО ЧТОБЫ МИН-МАКСЫ РИСОВАЛИСЬ А НЕ ОТКЛОНЕНИЯ
    #---------------------------------------------------------
    a =  mean_rd_r_to_l -  min_rd_r_to_l
    b =  max_rd_r_to_l -  mean_rd_r_to_l

    c =  mean_rd_l_to_r - min_rd_l_to_r
    d =  max_rd_l_to_r - mean_rd_l_to_r
    #---------------------------------------------------------

    print(data_l_to_r.shape)
    print(data_r_to_l.shape)
    print(Time.shape)
    plt.figure(figsize=(18,9))

    plt.subplot(2,1,1)
    plt.xlabel("Time, с")
    plt.ylabel("PI")
    plt.title(f"Polinomaial Granger causality average values.\n Window width = 20 c.")
    plt.scatter(supertime , data_r_to_l.T, marker='x', color='red')   
    plt.errorbar(Time, mean_rd_r_to_l, yerr =[a, b], label = "Cx_R->Cx_L", marker = 'o', linestyle = '-', color = "black", capsize=10,)
    plt.axhline(base_r_to_l_mean1, color = "black", linestyle = '--', alpha=0.5,label = "Cx_R->Cx_L BASE")
    #plt.xlim(0,600)
    #plt.ylim(0,0.075)

    plt.legend(fontsize=16)
    plt.minorticks_on()
    plt.grid(which='major')
    plt.tight_layout()


    plt.subplot(2,1,2)
    plt.xlabel("Time, с")
    plt.ylabel("PI")
    plt.scatter(supertime , data_l_to_r.T, marker='x', color='red' )
    plt.errorbar(Time, mean_rd_l_to_r, yerr = [c, d], label = "Cx_L->Cx_R", marker = 'o', linestyle = '-', color = "black", capsize=10)
    plt.axhline(base_l_to_r_mean1, color = "black", linestyle = '--', alpha=0.5,label = "Cx_L->Cx_R BASE")
    #plt.xlim(0,600)
    #plt.ylim(0,0.075)

    plt.legend(fontsize=16)
    plt.minorticks_on()
    plt.grid(which='major')
    plt.tight_layout()


    plt.savefig(f"Pi n_cen = {n_cen}.pdf")
    #plt.show()
    #time.sleep(5)
    #plt.close()

    
    

if __name__ == "__main__":
    Data = r"C:\Users\insec\Desktop\mouse Cx_RD+\Filtered\282-18.07"
    casual_281 = r"E:\Work\DIplom\Крысы\For article\Rat 281\rd"
    base_281 = r"E:\Work\DIplom\Крысы\For article\Rat 281\base"

    RD_casual = r"/home/arzdez/Desktop/Крысы/For article/Rd in Cx casual"
    Base_casual = r"/home/arzdez/Desktop/Крысы/For article/Baseline casual"


    granger_parallel(RD_casual, N_centr=11, Tau=100, names="rd", step=20000)
    granger_parallel(Base_casual , N_centr=11, Tau=100, names="base", step=20000)
    meaner(RD_casual, Base_casual, n_cen=11)

