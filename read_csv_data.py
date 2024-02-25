import csv
import numpy as np

def Read_csv_data(filename, Ngrab = 3):
    data_grab = []
    dc = np.zeros(Ngrab)
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=",")   
        for row in reader:
            try:
                if (row[0][0] != '#'): # ignore comments
                    for ii in range(Ngrab):
                        dc[ii] = float(row[ii])
                    if (len(data_grab) == 0):
                        data_grab = dc
                    else:
                        data_grab = np.r_['0,2,1', data_grab, dc]
                else:
                    try:
                        print(row[0], end =" ")
                        print ('%s' % ', '.join(map(str, row[1:])))
                    except:
                        print(" ")
            except:
                pass
    return data_grab