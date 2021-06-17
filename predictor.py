#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np

import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[46]:


venue_encoder = LabelEncoder()
team_encoder = LabelEncoder()

def processed_data(df):
  df.fillna(0)
  relevant_cols = ["venue" ,"innings", 'batting_team', 'bowling_team','striker', 'bowler' , 'total_runs']
  ipl_data = df[relevant_cols]
  
  # encoding
  ipl_data_1= ipl_data.copy()
  ipl_data_1['venue'] = venue_encoder.fit_transform(ipl_data_1['venue'])
  ipl_data_1['batting_team']= team_encoder.fit_transform(ipl_data_1['batting_team']) 
  ipl_data_1['bowling_team']= team_encoder.fit_transform(ipl_data_1['bowling_team']) 

  return ipl_data_1

path = os.getcwd()
filename = "final_modified_data.csv"

file = pd.read_csv(os.path.join(path,filename), index_col = None)
ipl_data_1 = processed_data(file)


# In[47]:


anarray = ipl_data_1.to_numpy()

x = anarray[:,:6]
y = anarray[:,6]




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


LinearRegressor = LinearRegression()
LinearRegressor.fit(x_train, y_train)


# In[48]:


file_players_striker = pd.read_csv("players_striker_rate.csv", index_col = None)
file_bowler_performance = pd.read_csv("bowling_performance.csv", index_col = None)

file_striker = file_players_striker[['striker','striker_rate']].set_index('striker')
file_bowler = file_bowler_performance[["bowler",'bowler_rate']].set_index('bowler')

dict_striker = file_striker.T.to_dict()
dict_bowler = file_bowler.T.to_dict()


# In[49]:


def make_players_performance_in_numbers(data, dictionary, rate):
    list_performance = []
    for players_list in data:
        count = 0
        sum = 0
        for players in players_list:
            if players in dictionary:
                count = count+1
                sum = sum + dictionary[players][rate]
            else:
                count = count
                sum = sum
        if rate == "striker_rate":
            average = sum * 10
        else:
            average = sum /10
        
        list_performance.append(round(average,3))

    data = pd.Series(list_performance)
    return data


# In[53]:


def predictRuns(test_file):
    
    #path_1 = os.getcwd()
    #filename_1 = test_file
    inputFile = pd.read_csv(test_file, index_col = None)

    inputFile_1 = inputFile[["venue", 'innings', "batting_team", "bowling_team","batsmen", "bowlers"]]
    
    test_file = inputFile_1.copy()
    test_file['batsmen'][0] = test_file['batsmen'][0].split(',')
    test_file['bowlers'][0] = test_file['bowlers'][0].split(',')
    test_file['batsmen'] = make_players_performance_in_numbers(test_file['batsmen'], dict_striker, "striker_rate")
    test_file['bowlers'] = make_players_performance_in_numbers(test_file['bowlers'], dict_bowler, "bowler_rate")
    
    test_file['venue'] = venue_encoder.transform(test_file['venue']) 
    test_file['batting_team']= team_encoder.transform(test_file['batting_team'])  
    test_file['bowling_team']= team_encoder.transform(test_file['bowling_team'])
    
    testarray = test_file.to_numpy()
    powerplay_run = LinearRegressor.predict(testarray)


    return int(powerplay_run)



#run = predictRuns("19_inn2.csv")
#run

