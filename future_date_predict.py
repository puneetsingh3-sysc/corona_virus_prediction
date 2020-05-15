"""
 @ future_date_predict.py

 Contains the future forecast function which will add the dates which is given by the user to predict the number of cases which can be possible using different machine learning algorithms.

"""

"""
Including Python Packages
"""
import numpy as np
import datetime as dt

class future_forecast():
    def days_add(punjas,dates,col):
        while True:
            days_choice = input("\n\t\tEnter days between 0 to 20 day for predicting the coronavirus: ")
	    
            try:
                if int(days_choice)>0 and int(days_choice)<21:
                    break 
                else:
                    print('\t\tIncorrect choice')

            except ValueError:
                print('\t\t Invalid character')
	
        # Add dates in this list

        future_forecast = np.array([n_days for n_days in range(len(dates)+int(days_choice))]).reshape(-1,1)
        days_choice = int(days_choice)
        new_dates = future_forecast[:-days_choice]
            
        start = col[-1]+"20"
        start_dates = dt.datetime.strptime(start, "%m/%d/%Y")
        future_dates = []
        temp_date = []
        for i in range(days_choice+1):
            if (i != 0):
                future_dates.append((start_dates + dt.timedelta(days=i)).strftime("%m/%d/%Y"))
            else:
                temp_date.append((start_dates + dt.timedelta(days=i)).strftime("%m/%d/%Y"))

        # returning number of days for future pediction, future date generated, future dates added in the list, and new_dates is added dates with existing dates
        return (days_choice,future_dates,future_forecast,new_dates)

	    
