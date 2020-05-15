"""
 @ main_corona.py

 Contains the main function which calls the corona_virus class and their methods are used to predict the future date cases using Support Vector Machine and Linear Regression machine learning algorithms.
"""

"""
Including Python Packages and some packages were installed with the help of requirements.txt
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import operator
import random
import math
import time
import datetime
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from future_date_predict import *
import seaborn as sns


# Fetching the raw data from Github
url_confirm_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_recover_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

# Starting of the corona_virus class
class corona_virus(object):

    def __init__(self,choice):
        self.choice = choice
    
    def main_method_call(self):
        methods = 'corona_' + str(self.choice)
        method_names = getattr(self, methods, lambda:'Invalid')
        print("\tChoice {0} and {1} method is selected".format(choice,methods))
        return method_names()

    def corona_1(self):
        confirmed_global = pd.read_csv(url_confirm_global, error_bad_lines=False)
	
        # fix region names
        confirmed_global['Country/Region']= confirmed_global['Country/Region'].str.replace("Mainland China", "China")
        confirmed_global['Country/Region']= confirmed_global['Country/Region'].str.replace("US", "Unites States")
        return confirmed_global

    def corona_2(self):
        deaths_global = pd.read_csv(url_deaths_global, error_bad_lines=False)
	
        # fix region names
        deaths_global['Country/Region']= deaths_global['Country/Region'].str.replace("Mainland China", "China")
        deaths_global['Country/Region']= deaths_global['Country/Region'].str.replace("US", "Unites States")
        return deaths_global

    def corona_3(self):
        recover_global = pd.read_csv(url_recover_global, error_bad_lines=False)
	
        # fix region names
        recover_global['Country/Region']= recover_global['Country/Region'].str.replace("Mainland China", "China")
        recover_global['Country/Region']= recover_global['Country/Region'].str.replace("US", "Unites States")
        return recover_global
	
    def country_or_states_cases(self,ld,coun,flag):
        area_cases = []
        area_wo_cases = []
	
        for i in coun:
            if (flag == 1):
                cases = ld[self.data['Country/Region']==i].sum()
            else:
                cases = ld[self.data['Province/State']==i].sum()

            if cases > 0:
                area_cases.append(cases)
            else:
                area_wo_cases.append(i)

        for i in area_wo_cases:
            coun.remove(i)

        coun = [sort_zip for sort_zip, temp in sorted(zip(coun, area_cases),key=operator.itemgetter(1), reverse = True)]
        for i in range(len(coun)):
            area_cases[i] = ld[self.data['Country/Region']==coun[i]].sum()
        
        if (flag == 1):
            print("\tConfirmed Positive cases by Country/Regions are: \n")
        else:
            print("\tConfirmed Positive cases by Province/States are: \n")

        for i in range(len(coun)):
            print(f"\t{coun[i]}: {area_cases[i]} cases")

        return (coun,area_cases,area_wo_cases)

    def main(self,data,cols):
        self.data = data
        sum_list = []
        dates = self.data.loc[1:,cols[4]:cols[-1]]
        dates_final = dates.keys()

        print(self.data)
        # Choice 'Yes' will print the dataset with start date, last updated date and all dates. Choice 'No' will execute the following code without printing the dates.
        date_choice = input("\n\t\tDo you want to see the starting and last updated date (yes/no)? ")
        if (date_choice == "yes"):
            print("\tDataset for corona virus commenced on :",cols[4])
            print("\tLast updated on :",cols[-1])
            print("\tDates selected are: ",dates_final)

        else:
            print("\tExecuting preceding code because not written 'yes'")

        data_sum = 0 
        for i in dates_final:
            data_sum = dates[i].sum()
            sum_list.append(data_sum)
        
        print ("\n\tSum of the cases is:",data_sum)
        
        all_days = np.array([i for i in range(len(dates_final))]).reshape(-1,1)
        sum_list = np.array(sum_list).reshape(-1,1)

        # Calls the future_forecast function from future_date_predict file
        future = future_forecast()
        days_choice, add_dates, future_fct, new_dates = future.days_add(all_days,cols)

        latest_data = self.data[dates_final[-1]]
        countries = list(self.data['Country/Region'].unique())

        country_choice = input("\n\t\tDo you want to see the list of countries affected by Corona (yes/no)? ")
        if (country_choice == "yes"):
            print (countries)

        else:
            print("\tDid not enter 'yes', so moving forward")

        (countries,countries_cases,country_wo_cases) = self.country_or_states_cases(latest_data,countries,1)

        uniq_states = list(self.data['Province/State'].unique())
        (states,states_cases,states_wo_cases) = self.country_or_states_cases(latest_data,uniq_states,2)

        non_value = []

        for i in range(len(uniq_states)):
            if type(uniq_states[i]) == float:
                non_value.append(i)
        
        uniq_states = list(uniq_states)
        states_cases = list(states_cases)
        
        for i in non_value:
            uniq_states.pop(i)
            states_cases.pop(i)

        plt.figure(figsize=(100,100))
        plt.barh(countries, countries_cases,height=0.8)
        plt.title("Cases of COVID-2019 in Countries")
        plt.xlabel("Number of COVID-2019 Cases", fontsize=10)
        plt.ylabel("Countries affected by COVID-2019",fontsize=10)
        plt.show()

        china_data = latest_data[self.data['Country/Region']=='China'].sum()
        outside_china = np.sum(countries_cases) - china_data

        china_choice = input("\n\t\tDo you want to see the China and outside data (yes/no)? ")
        if (china_choice == "yes"):
            print("\n\tOutside Mainland China {} cases: ".format(outside_china))
            print("\tMainland China {} cases".format(china_data))
            print("\tTotal: {} cases".format(china_data+outside_china))
            plt.figure(figsize=(30,30))
            plt.barh('Mainland China',china_data)
            plt.barh('Outside Mainland China',outside_china)
            plt.title('Number of Covid-2019 cases in China')
            plt.show()
        
        else:
            print("\n\tDid not enter 'yes', so moving forward")


        imp_countries = []
        imp_cases = []
        others = np.sum(countries_cases[10:])
        for i in range(len(countries_cases[:10])):
            imp_countries.append(countries[i])
            imp_cases.append(countries_cases[i])

        imp_countries.append('Others')
        imp_cases.append(others)

        affect_choice = input("\n\t\tDo you want to see the top 10 affected countries (yes/no)? ")
        if (affect_choice == "yes"):
            plt.figure(figsize=(30,30))
            plt.barh(imp_countries,imp_cases)
            plt.title("Number of Covid-2019 cases in Countries/Regions",size=20)
            plt.show()
	
        else:
            print("\tDid not enter 'yes', so moving forward")

        pie_choice = input("\n\t\tTop 10 affected countries with pie chart (yes/no)? ")
        if (pie_choice == "yes"):
            c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(countries))
            plt.figure(figsize=(100,100))
            plt.title("Covid-2019 cases per country")
            plt.pie(imp_cases,colors=c)
            plt.legend(imp_countries,loc='best')
            plt.show()

        else:
            print("\tDid not enter 'yes', so moving forward")


        kernel_type = ['poly','sigmoid','rbf']
        regularization = [0.01, 0.1, 1, 10]
        kernel_coeff = [0.01, 0.1, 1]
        loss_fn = [0.01, 0.1, 1]
        shrink_bool = [True,False]
        svm_grid_param= {'kernel': kernel_type,'C': regularization,'gamma': kernel_coeff,'epsilon': loss_fn ,'shrinking':shrink_bool}
       
        # Creating an object of svm algorithm from SVR as svm
        svm = SVR()
        svm_search = RandomizedSearchCV(svm,svm_grid_param,scoring='neg_mean_squared_error',cv=4,return_train_score=True, n_jobs=-1,n_iter=17,verbose=1)

        # Splitting the data taken from the csv file
        xtrain,xtest,ytrain,ytest = train_test_split(all_days, sum_list, test_size=0.2,random_state=2,shuffle=False)
        svm_search.fit(xtrain, np.ravel(ytrain))

        svm_select_params = svm_search.best_estimator_
        print ('\n\tSVM parameters selected are: ',svm_select_params)

        svm_prediction = svm_select_params.predict(future_fct)
        print ('\n\tSVM Prediction: ',svm_prediction)

        svm_test_predict = svm_select_params.predict(xtest)	

        plt.plot(svm_test_predict)
        plt.plot(ytest)
        print ('\n\tMean Absolute Error is:',mean_absolute_error(svm_test_predict, ytest))
        print ('\tMean Squared Error is:',mean_squared_error(svm_test_predict, ytest))
       
        # Total number of Cases till now
        plt.figure(figsize=(100,100))
        plt.plot(new_dates, sum_list)
        plt.title('Number of COVID-2019 cases over time',size=30)
        plt.xlabel('Days onwards 22nd January 2019',size=30)
        plt.ylabel('Number of Cases',size=30)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.show()
       
        # COVID Cases vs predicted cases
        plt.figure(figsize=(100,100))
        plt.plot(new_dates, sum_list)
        plt.plot(future_fct,svm_prediction, linestyle='dashed', color='blue')
        plt.title('COVID-2019 cases using SVM Predictions',size=30)
        plt.xlabel('Days onwards 22nd January 2019',size=30)
        plt.ylabel('Number of Cases',size=30)
        plt.legend(['Confirmed Cases of COVID-2019', 'SVM Predictions'])
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.show()
        

        # Prediction for the future days given above
        print('\n\tFuture dates prediction using SVM:')
        print(set(zip(add_dates[-days_choice:],svm_prediction[-days_choice:])))
	
	
	# Predicting the COVID cases using Linear Regression model
        lr_model = LinearRegression(normalize=True, fit_intercept=True)
        lr_model.fit(xtrain,np.ravel(ytrain))
        lr_test_predict = lr_model.predict(xtest)
        lr_predict = lr_model.predict(future_fct)

        plt.plot(lr_test_predict)
        plt.plot(ytest)
        print ('\n\tMean Absolute Error is:',mean_absolute_error(lr_test_predict, ytest))
        print ('\tMean Squared Error is:',mean_squared_error(lr_test_predict, ytest))
       
        # Total number of Cases till now
        plt.figure(figsize=(100,100))
        plt.plot(new_dates, sum_list)
        plt.plot(future_fct, lr_predict, linestyle='dashed', color='red')
        plt.title('COVID-2019 cases using Linear regression Predictions',size=30)
        plt.xlabel('Days onwards 22nd January 2019',size=30)
        plt.ylabel('Number of Cases',size=30)
        plt.legend(['Confirmed Cases of COVID-2019', 'Linear Regression Predictions'])
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.show()

        print('\n\tFuture predictions using Linear Regression:')
        print(lr_predict[-days_choice:])

# End of the class


# Code will commence from here that is main function
if __name__ == "__main__":

    print("\t\tPress 1 to see data for confirmed global cases")
    print("\t\tPress 2 to see data for global deaths cases")
    print("\t\tPress 3 to see data for recovered global cases\n")

    while True:
        choice = input("\tEnter your choice: ")
        try:
            if int(choice)>0 and int(choice)<4:
                # initializing object and calling __init__ method
                cv = corona_virus(choice)
                # using object to call other methods of the class
                data = cv.main_method_call()
                break
            else:
                print('\t\tIncorrect choice')

        except ValueError:
            print('\t\t Invalid character')

    # Extracting columns of the dataset
    columns = data.keys()

    choice = input("\n\t\tDo you want to see the name of columns (yes/no)? ")
    if (choice == "yes"):
        print("\t",columns)

    else :
        print("\tDid not enter 'yes', so moving forward")

    cv.main(data,columns)

