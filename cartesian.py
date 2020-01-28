import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from datetime import datetime
import pickle
import math
from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle
from sklearn import metrics
import os.path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support as score


df = pd.read_csv("Datatset/prs_dataset.csv", sep=',', encoding='utf-8')

df = df[(df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')]


project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes', 'angular.js', 'laravel', 'opencv',
                 ]

multi_class_model_path = 'Results/CART/3_labels/'
SSF_folder_path = 'Results/Baseline/SSF/'
FIFO_folder_path = 'Results/Baseline/FIFO/'



# One hot encoding
def encode_labels(df1, column_name):
    encoder = preprocessing.LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return  one_hot_vector


#Creating the dependent variable class
factor = pd.factorize(df['label'])
df.label = factor[0]
definitions = factor[1]
print(df.label.head())
print(definitions)


df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')


df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']

# Selected features
predictors = ['num_comments', 'Contributor', 'Participants_Count', 'line_comments_count', 'Deletions_Per_Week', 'Additions_Per_Week',
         'Project_Accept_Rate', 'Mergeable_State', 'User_Accept_Rate', 'first_response', 'Project_Domain', 'latency_after_first_response',
         'comments_reviews_words_count', 'Wait_Time', 'Team_Size', 'Stars', 'Language', 'Assignees_Count', 'Sunday', 'Contributor_Num',
         'Watchers', 'Last_Comment_Mention', 'Contributions', 'Saturday', 'Wednesday', 'Label_Count', 'Commits_PR', 'PR_Latency',
         'Comments_Per_Merged_PR', 'Organization_Core_Member', 'Comments_Per_Closed_PR'
          ]


start_date = '2017-09-01'
end_date = '2018-02-28'

target = 'label'

X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]


   df_balanced_test = pd.DataFrame()
    for project in project_list:
        # print(project)
        # df_proj = X_train_total.loc[[project]]
        df_proj = df_[df_.Project_Name == project]
        if given_value == 0:
            alternate_value = 1
            given_value_count = len(df_proj[df_proj[based_on] == given_value])
            alternate_value_count = len(df_proj[df_proj[based_on] == alternate_value])
            if given_value_count > alternate_value_count:
                X_train_neg = df_proj[df_proj[based_on] == given_value].sample(n=alternate_value_count, replace=True)
                X_train_pos = df_proj[df_proj[based_on] == alternate_value]
                X_total = X_train_pos.append(X_train_neg)
                df_balanced_test = df_balanced_test.append(X_total)
            else:
                X_train_neg = df_proj[df_proj[based_on] == given_value]
                X_train_pos = df_proj[df_proj[based_on] == alternate_value].sample(n=given_value_count, replace=True)
                X_total = X_train_pos.append(X_train_neg)
                df_balanced_test = df_balanced_test.append(X_total)
        else:
            alternate_value = 0
            given_value_count = len(df_proj[df_proj[based_on] == given_value])
            alternate_value_count = len(df_proj[df_proj[based_on] == alternate_value])
            if given_value_count > alternate_value_count:
                X_train_neg = df_proj[df_proj[based_on] == given_value].sample(n=alternate_value_count, replace=True)
                X_train_pos = df_proj[df_proj[based_on] == alternate_value]
                X_total = X_train_pos.append(X_train_neg)
                df_balanced_test = df_balanced_test.append(X_total)
            else:
                X_train_neg = df_proj[df_proj[based_on] == given_value]
                X_train_pos = df_proj[df_proj[based_on] == alternate_value].sample(n=given_value_count, replace=True)
                X_total = X_train_pos.append(X_train_neg)
                df_balanced_test = df_balanced_test.append(X_total)


    return df_balanced_tes

def CART_Model(df_test_PR, folder_path, file_name):
    pd.options.mode.chained_assignment = None
    with open('../Models/Saved_Models/3_labels/XGBoost.pickle.dat', 'rb') as f:
        xgb_model = pickle.load(f)
        y_pred_accept = xgb_model.predict(df_test_PR[predictors])
        df_test_PR['Score'] = y_pred_accept
        print(df_test_PR[['Pull_Request_ID', 'label', 'Score']].head(10))
        df_test_PR.sort_values(by=['Score'], ascending=True).to_csv(
            folder_path + file_name, sep=',', encoding='utf-8', index=False)

        return df_test_PR.sort_values(by=['Score'], ascending=True)

    pd.options.mode.chained_assignment = None
    with open('../Accept/Models_21_11_19/accept_xgb.pickle.dat', 'rb') as f:
        accept_model = pickle.load(f)
        # y_pred_accept = accept_model.predict(df_test_PR[predictors_a])
        y_pred_accept = accept_model.predict_proba(df_test_PR[predictors_a])[:,1]
        df_test_PR['Result_Accept'] = y_pred_accept

    # print(df_test_PR[['Pull_Request_ID', 'PR_accept', 'Result_Accept']].head(10))

    with open('../Response/Models_21_11_19/response_xgb.pickle.dat', 'rb') as f:
        response_model = pickle.load(f)
        # y_pred_response = response_model.predict(df_test_PR[predictors_r])
        y_pred_response = response_model.predict_proba(df_test_PR[predictors_r])[:,1]
        df_test_PR['Result_Response'] = y_pred_response

    # print(df_test_PR[['Pull_Request_ID', 'PR_response', 'Result_Response']].head(10))

    # df_test_PR['Score'] = (df_test_PR['Result_Accept'] + df_test_PR['Result_Response'])/2

    df_test_PR['Score'] = df_test_PR['Result_Accept'].apply(np.exp) + df_test_PR['Result_Response'].apply(np.exp)
    # result = [math.exp(y_pred_accept[index]) + math.exp(y_pred_response[index]) for index in range(len(y_pred_accept))]
    # result = [math.exp(y_pred_accept_prob[index]) + math.exp(y_pred_response_prob[index]) for index in range(len(y_pred_accept))]
    # result =[(y_pred_accept[index] + y_pred_response[index])/2 for index in range(len(y_pred_accept))]
    # result = {'result': result}
    # result = pd.DataFrame(result)
    # df_test_PR['Score_2'] = result
    print(df_test_PR[['Pull_Request_ID', 'Result_Accept', 'Result_Response', 'Score']].head(10))
    # print(result['result'])
    df_test_PR.sort_values(by=['Score'], ascending=False).to_csv(
        algo_folder_path+'cart_results.csv', sep=',', encoding='utf-8', index=False)
    return df_test_PR
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', '2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    top_k_list = [5, 10, 20]
    df = df.set_index('Project_Name')
    df_algo_results = pd.DataFrame(columns=['Project', 'MAP_5'])
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for project in project_list:
            total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                # print(date_start, date_end)
                # total_result = []
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                # print(last_date)
                for i in range(1, last_date+1):
                    MAP_accept_response = 0
                    positive_count_accept_response = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if counter >top_k: break
                            if row['label'] == 0 or row['label'] == 1:
                            #if row['PR_response'] == 1: # For baseline model
                                positive_count_accept_response += 1
                                MAP_accept_response += positive_count_accept_response/(counter)

                    total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                    MAP_day = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
                # print('Project {0} in month {1} have MAP {2:.3f}'.format(project, date_start.split('-')[0]+'-'+date_start.split('-')[1],
                #                                                          MAP_day))
            if top_k > 5:
                df_MAP = df_MAP.append({'MAP_'+str(top_k): np.mean(total_result)}, ignore_index=True)
                print(df_MAP.shape)
            else:
                df_algo_results = df_algo_results.append(
                    {'Project': project, 'MAP_' + str(top_k): np.mean(total_result)}, ignore_index=True)
                print(df_algo_results.shape)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.sort_values(by=['MAP_5', 'MAP_10', 'MAP_20'], ascending=False).to_csv(
        folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)


    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', '2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    df = df.set_index('Project_Name')
    top_k_list = [5, 10, 20]
    df_algo_results = pd.DataFrame(columns=['Project', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for project in project_list:
            total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                # total_result = []
                # print(df_month.shape)
                # print(date_start, "-", date_end)
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                # print(last_date)
                for i in range(1, last_date+1):
                    top_recall_num = 0
                    total_recall_num = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if row['label'] == 0 or row['label'] == 1:
                            # if row['PR_response'] == 1: # for baseline model
                                if counter <= top_k:
                                    top_recall_num += 1
                                total_recall_num += 1
                    total_result.append(top_recall_num/total_recall_num if total_recall_num !=0 else 0)
                    AR_day = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                print('Project {} in month {} have average recall {}'.format(
                    project, date_start.split('-')[0] + '-' + date_start.split('-')[1], AR_day))

            if top_k > 5:
                df_AR = df_AR.append({'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
            else:
                df_algo_results = df_algo_results.append(
                    {'Project': project, 'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    # print(df_algo_results.head())
    df_algo_results.sort_values(by=['AR_5', 'AR_10', 'AR_20'], ascending=False).to_csv(
        folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=Fal
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }

    df = df.set_index('Project_Name')
    day_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'Day', 'MAP'])
    month_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'MAP'])
    for project in project_list:
        df_project = df.loc[[project]]
        for date_start, date_end in months_tested.items():
            df_month = df_project.loc[
                (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
            print(project)
            # print(date_start, date_end)
            total_result = []
            last_date = datetime.strptime(date_end, "%Y-%m-%d").day
            print(last_date)
            for i in range(1, last_date+1):
                MAP_accept_response = 0
                positive_count_accept_response = 0
                counter = 0
                top_k = 0
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            top_k += 1
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        counter += 1
                        if counter > top_k: break
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                        #if row['PR_response'] == 1: # For baseline model
                            positive_count_accept_response += 1
                            MAP_accept_response += positive_count_accept_response/(counter)

                total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                day_MAP = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
                day_wise_result = day_wise_result.append({'Project':project,
                                                          'Year-Month':date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                          'Day':i, 'MAP': day_MAP}, ignore_index=True)
                print('Project {0} in month {1} and day {2} have MAP {3:.3f}'.format(project, date_start.split('-')[0] + '-' +
                                                                         date_start.split('-')[1], i, day_MAP))
            print('Project {0} in month {1} have MAP {2:.3f}'.format(project, date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                                     np.mean(total_result)))
            month_wise_result = month_wise_result.append({'Project':project, 'Year-Month':date_start.split('-')[0]+'-'+date_start.split('-')[1],
                                                          'MAP':np.mean(total_result)}, ignore_index=True)
    # print(df_algo_results.head())
    day_wise_result.to_csv(folder_path+'MAP_day_results_1.csv', sep=',', encoding='utf-8', index=False)
    month_wise_result.to_csv(folder_path + 'MAP_month_results_1.csv', sep=',', encoding='utf-8', index=False)
    calculate_average_dynamic_map_for_months(month_wise_result, folder_pa
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30',
                     '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31', #'2018-02-01': '2018-02-28',  #'2018-03-01':'2018-03-31'
                     }
    df = df.set_index('Project_Name')
    day_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'Day', 'AR'])
    month_wise_result = pd.DataFrame(columns=['Project', 'Year-Month', 'AR'])
    for project in project_list:
        df_project = df.loc[[project]]
        for date_start, date_end in months_tested.items():
            df_month = df_project.loc[
                (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
            print(project)
            total_result = []
            last_date = datetime.strptime(date_end, "%Y-%m-%d").day
            for i in range(1, last_date+1):
                top_recall_num = 0
                total_recall_num = 0
                counter = 0
                top_k = 0
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            top_k+=1
                for index, row in df_month.iterrows():
                    if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                        counter += 1
                        if row['PR_accept'] == 1 and row['PR_response'] == 1:
                            if counter <= top_k:
                                top_recall_num += 1
                            total_recall_num += 1
                total_result.append(top_recall_num/total_recall_num if total_recall_num !=0 else 0)
                day_AR = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                day_wise_result = day_wise_result.append({'Project': project,
                                                      'Year-Month': date_start.split('-')[0] + '-' +
                                                                    date_start.split('-')[1],
                                                      'Day': i, 'AR': day_AR}, ignore_index=True)
                print(
                'Project {0} in month {1} and day {2} have AR {3:.3f}'.format(project, date_start.split('-')[0] + '-' +
                                                                               date_start.split('-')[1], i, day_AR))
            print('Project {0} in month {1} have AR {2:.3f}'.format(project, date_start.split('-')[0] + '-' + date_start.split('-')[1],
                                                               np.mean(total_result)))
            month_wise_result = month_wise_result.append({'Project': project, 'Year-Month': date_start.split('-')[0] + '-' + date_start.split('-')[1],
             'AR': np.mean(total_result)}, ignore_index=True)
        # print(df_algo_results.head())
    day_wise_result.to_csv(folder_path + 'AR_day_results.csv', sep=',', encoding='utf-8', index=False)
    month_wise_result.to_csv(folder_path + 'AR_month_results.csv', sep=',', encoding='utf-8', index=False)
    calculate_average_dynamic_recall_for_months(month_wise_result, folder

def Randomly_select_PRs(df_, number_of_PRs, total_samples):
    df_total = pd.DataFrame()
    for i in range(total_samples):
        df_sample= df_.loc[df_.label == i].sample(n=number_of_PRs, replace=True)
        df_total = df_total.append(df_sample)

    print(df_total.shape)
    print(df_total.label.value_counts())

    return df_total.to_csv('Random_samples/381_samples.csv', sep=',', encoding='utf-8', index=False)


def predict_labels_for_3_models_on_random_samples(folder_path):
    index = 0
    for file in listdir(folder_path):
        if os.path.isfile(folder_path + file):
            df = pd.read_csv(folder_path + file, encoding='utf-8')
            df_MCM = df.sort_values(by=['Score'], ascending=True)  
            df_FIFO = df.sort_values(by=['PR_Date_Created_At', 'PR_Time_Created_At'], ascending=True)
            df['src_churn'] = df['Additions'] + df['Deletions']
            df_SSF = df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)

            
            df_MCM[['Pull_Request_ID', 'label', 'Score']].to_csv(
                'Random_samples/CART/sample_'+str(index)+'.csv', sep=',', encoding='utf-8', index=False)
            df_SSF[['Pull_Request_ID', 'label', 'src_churn', 'Files_Changed']].to_csv(
                'Random_samples/SBM/sample_'+str(index)+'.csv', sep=',', encoding='utf-8', index=False)
            df_FIFO[['Pull_Request_ID', 'label', 'PR_Date_Created_At', 'PR_Time_Created_At']].to_csv(
                'Random_samples/FIFO/sample_'+str(index)+'.csv', sep=',', encoding='utf-8', index=False)
                
            df_MCM.to_csv('Random_samples/CART/3_labels/all_columns/sample_' + str(index) + '.csv',
                                                                 sep=',', encoding='utf-8', index=False)
            df_SSF.to_csv('Random_samples/SSF/3_labels/all_columns/sample_' + str(index) + '.csv',
                          sep=',', encoding='utf-8', index=False)
            df_FIFO.to_csv('Random_samples/FIFO/3_labels/all_columns/sample_' + str(index) + '.csv',
                           sep=',', encoding='utf-8', index=False)


            index+=1


def split_samples(df, folds):
    df = shuffle(df)
    df_split = np.array_split(df, folds)
    for index in range(len(df_split)):
        df_sub_sample = df_split[index]
        df_sub_sample.to_csv('Random_samples/10_parts/sample_'+str(index)+'.csv', encoding='utf-8', index=None)


def get_top_n_MAP_for_samples_list(model, folder_path, result_folder_path, file_name):
    top_k_list = [5, 10, 20, 30]
    df_algo_results = pd.DataFrame(columns=['Model','Sample', 'MAP_5'])
    count = 0
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for file in listdir(folder_path):
            if os.path.isfile(folder_path+file):
                df = pd.read_csv(folder_path + file, encoding='utf-8')
                # print(df.head())
                if model == 'CART':
                    sample_df = df.sort_values(by=['Score'], ascending=True) # for MCM
                elif model == 'FIFO':
                    sample_df = df.sort_values(by=['PR_Date_Created_At', 'PR_Time_Created_At'], ascending=True) # for FIFO
                elif model == 'Accept':
                    sample_df = df.sort_values(by=['Score'], ascending=True)
                else:
                    # for SSF
                    df['src_churn'] = df['Additions'] + df['Deletions']
                    sample_df = df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)
                MAP_accept_response = 0
                positive_count_accept_response = 0
                counter = 0
                for index, row in sample_df.iterrows():
                    counter += 1
                    if counter >top_k: break
                    if row['label'] == 0 or row['label'] == 1:
                        positive_count_accept_response += 1
                        MAP_accept_response += positive_count_accept_response/(counter)
                result = MAP_accept_response / positive_count_accept_response if positive_count_accept_response != 0 else 0
                print(result)
                if top_k > 5:
                    df_MAP = df_MAP.append({'MAP_'+str(top_k): result}, ignore_index=True)
                    print(df_MAP.shape)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Model':model,'Sample': 'Sample_'+str(count), 'MAP_' + str(top_k): result}, ignore_index=True)
                    count += 1
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.to_csv(result_folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)

def get_top_n_recall_for_samples_list(model, folder_path, result_folder_path, file_name):
    top_k_list = [5, 10, 20, 30]
    count = 0
    df_algo_results = pd.DataFrame(columns=['Model', 'Sample', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for file in listdir(folder_path):
            if os.path.isfile(folder_path+file):
                df = pd.read_csv(folder_path + file, encoding='utf-8')
                # print(df.head())
                if model == 'CART':
                    sample_df = df.sort_values(by=['Score'], ascending=True) # for CART
                elif model == 'FIFO':
                    sample_df = df.sort_values(by=['PR_Date_Created_At', 'PR_Time_Created_At'], ascending=True) # for FIFO
                elif model == 'Accept':
                    sample_df = df.sort_values(by=['Score'], ascending=True)  # for FIFO
                else:
                    # for SSF
                    df['src_churn'] = df['Additions'] + df['Deletions']
                    sample_df = df.sort_values(by=['src_churn', 'Files_Changed'], ascending=True)
                top_recall_num = 0
                total_recall_num = 0
                counter = 0
                for index, row in sample_df.iterrows():
                    counter += 1
                    if row['label'] == 0 or row['label'] == 1:
                        if counter <= top_k:
                            top_recall_num += 1
                        total_recall_num += 1
                result = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                print(result)
                if top_k > 5:
                    df_AR = df_AR.append({'AR_' + str(top_k): result}, ignore_index=True)
                else:
                    df_algo_results = df_algo_results.append(
                        {'Model':model,'Sample': 'Sample_'+str(count), 'AR_' + str(top_k): result}, ignore_index=True)
                    count += 1
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.to_csv(result_folder_path+file_name+'.csv', sep=',', encoding='utf-8', index=False)


def extract_metric_from_report(report):
    report = list(report.split("\n"))
    report = report[-2].split(' ')
    # print(report)
    mylist = []
    for i in range(len(report)):
        if report[i] != '':
            mylist.append(report[i])

    return mylist[3], mylist[4], mylist[5]

def calculate_results_samples(folder_path):
    results = pd.DataFrame(columns=['Sample', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy'])
    for file in listdir(folder_path):
        if os.path.isfile(folder_path + file):
            print(file.split('.')[0])
            df = pd.read_csv(folder_path + file, encoding='utf-8')
            y_test = df['label']
            y_pred = df['Score']
            # Print model report:
            print("\nModel Report")
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))

            test_accuracy = metrics.accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = score(y_test, y_pred)

            print(pd.crosstab(y_test, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test, y_pred, digits=3))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test, y_pred, digits=3))
            results = results.append(
                {'Sample': file.split('.')[0],
                 'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
                 'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
                 'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                 'Test_Accuracy': test_accuracy},
                ignore_index=True)

        results.to_csv('Random_samples/381_samples_results.csv', sep=',', encoding='utf-8', index=False)

def All_Model_Results(df_test_PR, folder_path):
    pd.options.mode.chained_assignment = None
    results = pd.DataFrame(columns=['Project', 'Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy'])
    models_list = ['XGBoost', 'DT', 'KNN', 'LogisticRegression', 'LinearSVC', 'NaiveBayes',
                   'RandomForest']
    for model in models_list:
        with open('../Saved_Models//'+model+'.pickle.dat', 'rb') as f:
            xgb_model = pickle.load(f)
            y_pred_accept = xgb_model.predict(df_test_PR[predictors])
            df_test_PR['Score'] = y_pred_accept
            
            for project in project_list:
                print(project)
                df_proj = df_test_PR.loc[df_test_PR.Project_Name == project]
                y_test = df_proj['label']
                y_pred = df_proj['Score']
                # Print model report:
                print("\nModel Report")
                print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
                print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))

                test_accuracy = metrics.accuracy_score(y_test, y_pred)
                precision, recall, fscore, support = score(y_test, y_pred)

                print(pd.crosstab(y_test, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

                print(metrics.classification_report(y_test, y_pred, digits=3))

                precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                    metrics.classification_report(y_test, y_pred, digits=3))
                try:
                    results = results.append(
                        {'Project': project, 'Model': model,
                         'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
                         'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
                         'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                         'Test_Accuracy': test_accuracy},
                        ignore_index=True)
                except IndexError:
                    continue

    results.to_csv('../Results/project_wise_results.csv', sep=',', encoding='utf-8', index=False)



if __name__ == '__main__':

    print('Processing')

    # CART model
    cart_result = CART_Model(X_test, multi_class_model_path, 'test_DS_results.csv')
    print(cart_result[['Pull_Request_ID', 'label', 'Score']].head(100))

    ## Generate random sample
    rand_sample = Randomly_select_PRs(cart_result, 127, 3)

    ## Split the ramdom sample to 10 parts
    split_samples(df, 10)
    

    get_top_n_MAP_for_samples_list('CART', 'Random_samples/10_parts/', 'Random_samples/', 'MAP_CART_results')
    get_top_n_recall_for_samples_list('CART', 'Random_samples/10_parts/', 'Random_samples/', 'AR_CART_results')
    get_top_n_MAP_for_samples_list('SBM', 'Random_samples/10_parts/', 'Random_samples/', 'MAP_SBM_results')
    get_top_n_recall_for_samples_list('SBM', 'Random_samples/10_parts/', 'Random_samples/', 'AR_SBM_results')
    get_top_n_MAP_for_samples_list('FIFO', 'Random_samples/10_parts/', 'Random_samples/', 'MAP_FIFO_results')
    get_top_n_recall_for_samples_list('FIFO', 'Random_samples/10_parts/', 'Random_samples/', 'AR_FIFO_results')


    predict_labels_for_3_models_on_random_samples('Random_samples/10_parts/')
    
    #All_Model_Results(cart_result, 'Random_samples/10_parts/')








