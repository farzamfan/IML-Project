import numpy as np 
import pandas as pd

def perc_patients_nodata_tests(data,v=False):
#calculates per test in data (columns except pid, Age, Time) the percentage 
#   of patients that have no recordings of that test at all (nan); v indicates
#   whether to print status updates
#returns a data frame with the percentage per test

    num_pat=np.size(data.index.unique())
    perc_nodata = pd.DataFrame(index=['perc_nodata'],columns=data.columns.drop(['Age','Time']))

    tests_handled = 0

    for col in data.columns.drop(['Age','Time']):
        num_pat_nodata_col = 0 #counter for how many patients have no data for that col
        pids_handled = 0
        if v==True:
            print(' ... handling test '+col+' (no. '+str(tests_handled+1)+' out of '+str(np.size(data.columns.drop(['Age','Time'])))+')')
        for pid in data.index.unique():
            pid_col = data.loc[data.index==pid,col]
            if pd.isnull(pid_col).all(): #check if no data for column col in patient pid
                num_pat_nodata_col += 1
            pids_handled += 1
            if v==True and pids_handled%1000==0: #print output every 1000 pids
                print('  ... handled '+str(pids_handled)+' pids out of '+str(num_pat))
        perc_nodata_col = num_pat_nodata_col/num_pat
        perc_nodata[col] = perc_nodata_col

        tests_handled += 1

    if v==True:
        print(' DONE!')

    return perc_nodata

#TOO SLOW.. not usable
def concat_pid_rows_tooslow(data):
#concatenates all rows from one pid

    tests = data.columns.drop(['pid','Time','Age'])
    tests_hours = [test+'_'+str(i) for i in range(1,13) for test in tests]
    col_data_concat = ['pid', 'Time', 'Age'] + tests_hours
    data_concat = pd.DataFrame(columns=col_data_concat)

    for pid in data['pid'].unique():

        data_pid = data[data['pid']==pid]
        data_pid.sort_values('Time',ascending=True,inplace=True)

        if len(data_pid.index)!=12:
            #ToDo: change to throw an error
            raise Exception('The pid '+str(pid)+' doesn\'t have 12 time points.')

        data_concat = data_concat.append(data_pid.loc[data_pid.index[0],['pid','Time','Age']].append(pd.Series(index=tests_hours)),ignore_index=True)

        t = 1
    for _, row in data_pid.iterrows():
        tests_t = [test+'_'+str(t) for test in tests]
        data_concat[tests_t] = row[row.index.drop(['pid', 'Time', 'Age'])].values
        t += 1

    return

def concat_pid_rows(data):
#concatenates all rows from one pid without having to loop through the rows

    data_asctime = data.groupby(data.index).apply(lambda x: x.sort_values('Time',ascending=True)) #sort data in each pid group in ascending order (in time), necessary for cumcount to work correctly
    data_asctime.index = data_asctime.index.droplevel()
    data_tests_concat = data_asctime[data_asctime.columns.drop(['Time','Age'])].set_index([data_asctime.index,data_asctime.groupby(data_asctime.index).cumcount()+1]).unstack().sort_index(level=1, axis=1)
    data_tests_concat.columns = data_tests_concat.columns.map('{0[0]}_{0[1]}'.format)

    #reinsert age
    age = data.groupby(data.index).apply(lambda x: x.head(1))
    age.index = age.index.droplevel()
    age = age.loc[:,'Age']

    data_concat = data_tests_concat.merge(age,how='inner',left_index=True,right_index=True)

    return data_concat