#b = ['1_1','1_2',"2_1","2_2","3_1","3_2"]
#root = 'phm-ieee-2012-data-challenge-dataset/Learning_set/Bearing%s'
#b = ['1_1','1_2',"2_1","2_2","3_1","3_2"]
b = ['1_3','1_4','1_5','1_6','1_7','2_3','2_4','2_5','2_6','2_7','3_3']
root = 'phm-ieee-2012-data-challenge-dataset/Test_set/Bearing%s'



import pandas as pd
import matplotlib.pyplot as pplot
import numpy as np
import os



if __name__ == '__main__':
    #for b_ in ['1_1','1_2',"2_1","2_2","3_1","3_2"]:

    for b_ in b:
        bearing_dir = root%b_

        dir_files = os.listdir(bearing_dir)
        AccDF = None
        TempDF = None
        acc_columns = ['h','m','s','se-6','h_acc','v_acc']
        kk = 0;
        for file in dir_files:
            v = pd.read_csv(os.path.join(bearing_dir, file))
            v = pd.DataFrame(v)
            bearing_conditions = b[0][0]
            exper_index = b[0][-1]
            meas_type = file[0:3]

            if meas_type == 'acc':
                v.columns = acc_columns 

                v['bearing_conditions'] = bearing_conditions
                v['exper_index'] = exper_index
                v['meas_type'] = file[0:3]
                v['block'] = file[4:9]
                if AccDF is None:
                    AccDF = v
                else:
                    AccDF = pd.concat([AccDF, v])
            kk += 1
            if kk % 1000 == 0:
                print("Working on b_:%s...%i"%(b_,kk))
        #v1 = pd.read_csv(os.path.join(bearing_dir, dir_files[100]))
        #v2 = pd.read_csv(os.path.join(bearing_dir, dir_files[213]))
        AccDF.to_pickle("AccB_%s.pickle"%b_)


