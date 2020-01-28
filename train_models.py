import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
# plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from statistics import mean
import xgboost as xgb
import operator
from xgboost import plot_importance
import pickle
from sklearn.utils import shuffle

# import seaborn as sns
# sns.set(style="white")
# sns.set(style='whitegrid', color_codes=True)

df = pd.read_csv("Datatset/prs_dataset.csv", sep=',', encoding='utf-8')

# 'cmssw'
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes', 'angular.js', 'laravel', 'opencv',
                ]


df = df[(df.Project_Name != 'githubschool') & (df.Project_Name != 'curriculum')]

print(df.shape)


def get_classifiers():
    return {
        'RandomForest': RandomForestClassifier(n_jobs=4, bootstrap=True, class_weight='balanced', n_estimators=500, max_depth = 15,
                                               random_state=42, oob_score=True, min_samples_split=7, min_samples_leaf=3),
        'LinearSVC': LinearSVC(max_iter=2000),
        'LogisticRegression': LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1200),
        'XGBoost': xgb.XGBClassifier(**params),  # {'max_depth': 9, 'min_child_weight': 5}
        'DT': DecisionTreeClassifier(max_depth=5), # max_depth=5
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=7) # n_neighbors=7
    }



params = {
    'objective': 'multi:softprob',
    'num_class': 4,
    'eta': 0.08,
    'colsample_bytree': 0.886,
    'min_child_weight': 1.1,
    'max_depth': 9,
    'subsample': 0.886,
    'gamma': 0.1,
    'lambda': 10,
    'verbose_eval': True,
    'eval_metric': 'auc',
    'scale_pos_weight': 6,
    'seed': 201703,
    'missing': -1
}


def encode_labels(df1, column_name):
    encoder = LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')

#Creating the dependent variable class
factor = pd.factorize(df['label'])
df.label = factor[0]
definitions = factor[1]
# print(df.label.head())
print(definitions)

# {0: 'directly_accepted', 1: 'response_required', 2: 'rejected',}


df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']

# df = df[['Closed_Num_Rate', 'Label_Count', 'num_comments', 'Following', 'Stars', 'Contributions', 'Merge_Latency', #'Rebaseable',
#           'Followers',  'Workload', 'Wednesday', 'PR_accept', 'Closed_Num', 'Public_Repos',
#           'Deletions_Per_Week', 'Contributor', 'File_Touched_Average', 'Forks_Count', 'Organization_Core_Member',
#           'Monday', 'Contain_Fix_Bug', 'src_churn', 'Team_Size', 'Last_Comment_Mention', 'Sunday',
#           'Thursday', 'Project_Age', 'Open_Issues', 'Intra_Branch', 'Saturday', 'Participants_Count',
#           'Comments_Per_Closed_PR', 'Watchers', 'Project_Accept_Rate', 'Point_To_IssueOrPR', 'Accept_Num', 'Close_Latency',
#           'Contributor_Num', 'Commits_Average', 'Assignees_Count', 'Friday', 'Commits_PR', 'Wait_Time', 'line_comments_count',
#           'Prev_PRs', 'Comments_Per_Merged_PR', 'Files_Changed', 'Day', 'Churn_Average', 'Language', 'Tuesday',
#           'Mergeable_State', 'Additions_Per_Week', 'User_Accept_Rate', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6',
#           'X1_7', 'X1_8', 'X1_9', 'PR_Latency', 'Project_Name', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
#           'PR_Time_Closed_At', 'first_response_time', 'first_response', 'latency_after_first_response',
#           'title_words_count', 'body_words_count', 'comments_reviews_words_count',
#           'Project_Domain', 'label']]

# Selected Features
 df['src_churn'] = df['Additions'] + df['Deletions']
 df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']

df = df[['num_comments', 'Contributor', 'Participants_Count', 'line_comments_count', 'Deletions_Per_Week', 'Additions_Per_Week',
         'Project_Accept_Rate', 'Mergeable_State', 'User_Accept_Rate', 'first_response', 'Project_Domain', 'latency_after_first_response',
         'comments_reviews_words_count', 'Wait_Time', 'Team_Size', 'Stars', 'Language', 'Assignees_Count', 'Sunday', 'Contributor_Num',
         'Watchers', 'Last_Comment_Mention', 'Contributions', 'Saturday', 'Wednesday', 'Label_Count', 'Commits_PR', 'PR_Latency',
         'Comments_Per_Merged_PR', 'Organization_Core_Member', 'Comments_Per_Closed_PR', 'PR_Time_Created_At', 'PR_Date_Closed_At',
        'PR_Time_Closed_At', 'PR_Date_Created_At',
         'Project_Name', 'label', 'PR_accept']]



df = df.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)

target = 'label'
start_date = '2017-09-01'
end_date = '2018-02-28'

X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]
y_test = X_test[target]
X_train = df.loc[(df['PR_Date_Created_At'] < start_date)]
X_train = X_train
y_train = X_train[target]

# print('Test dataset')
# print(X_test[target].value_counts())
# accepted, rejected = X_test.PR_accept.value_counts()
# print('Percentage of accepted PRs {}'.format((accepted*100)/X_test.shape[0]))
# print('Percentage of rejected PRs {}'.format((rejected*100)/X_test.shape[0]))

predictors = [x for x in df.columns if x not in [target, 'PR_accept', 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name']]


X_train = X_train[predictors]
X_test = X_test[predictors]


print("Total Train dataset size: {}".format(X_train[predictors].shape))
print("Total Test dataset size: {}".format(X_test[predictors].shape))

# print(df.columns)

# Scale the training dataset: StandardScaler
def scale_data_standardscaler(df_):
    scaler_train =StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled, columns=predictors)

    return df_scaled

def extract_metric_from_report(report):
    report = list(report.split("\n"))
    report = report[-2].split(' ')
    # print(report)
    mylist = []
    for i in range(len(report)):
        if report[i] != '':
            mylist.append(report[i])

    return mylist[3], mylist[4], mylist[5]

def extract_each_class_metric_from_report(report):
    report = list(report.split("\n"))

    mydict2 = {}
    mydict = {}
    index = 0
    for line in range(len(report)):
        if report[line] != '':
            values_list = report[line].split(' ')
            mydict[index] = values_list
            index+=1
    count=0
    for value in mydict:
        mylist = []
        if value != 0:
            for item in range(len(mydict[value])):
                if mydict[value][item] != '':
                    mylist.append(mydict[value][item])
            mydict2[count] = mylist
            count+=1
    return mydict2[0], mydict2[1], mydict2[2], mydict2[3]

    
def train_XGB_model(clf, x_train, y_train, x_test, name=None):
    clf = clf.fit(x_train, y_train, verbose=11)
    importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    f_gain = clf.get_booster().get_score(importance_type='gain')
    importance = sorted(f_gain.items(), key=operator.itemgetter(1))
    print(importance)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('Results/features_fscore.csv', encoding='utf-8', index=True)

    # Save the model
    with open('Saved_Models/XGBoost.pickle.dat', 'wb') as f:
        pickle.dump(clf, f)

    # Load the model
    # with open('response_xgb_16.pickle.dat', 'rb') as f:
    #     load_xgb = pickle.load(f)
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]

    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def train_SVM_model(clf, x_train, y_train, x_test, name=None):
    clf.fit(x_train, y_train)
    svm = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    svm.fit(x_train, y_train)

    with open('Saved_Models/'+name+'.pickle.dat', 'wb') as f:
        pickle.dump(clf, f)
    # train
    y_pred_train = svm.predict(x_train)
    y_predprob_train = svm.predict_proba(x_train)[:, 1]
    # test
    y_pred = svm.predict(x_test)
    y_predprob = svm.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob

def train_RF_LR_model(clf, x_train, y_train, x_test, name=None):
    clf.fit(x_train, y_train)
    with open('Saved_Models/'+name+'.pickle.dat', 'wb') as f:
        pickle.dump(clf, f)
    # train
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    # test
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]

    return y_pred_train, y_predprob_train, y_pred, y_predprob


def XGB_features_ranking(x_train, y_train):

    for i in range(50):
        print("processing model number {}".format(i))
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        f_gain = model.get_booster().get_score(importance_type='gain')
        importance = sorted(f_gain.items(), key=operator.itemgetter(1))
        print(importance)
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df.to_csv('Results/features_selection/features_'+str(i)+'.csv', encoding='utf-8', index=True)


def train_XGB_model_feature_selection_2(x_train, y_train, x_test, y_test):
    df = pd.read_csv("Results/features_fscore.csv", sep=",")
    df = df.sort_values(by=['fscore'], ascending=False)
    thresholds = [200, 100, 50, 30, 25, 20, 15, 10, 5, 3, 2]
    results = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])

    for thresh in thresholds:
        # select features using threshold
        features_set = df.feature[df.fscore>=thresh]
        print(thresh)
        print(len(features_set))

        model = xgb.XGBClassifier(**params)
        model.fit(x_train[list(features_set)], y_train)
        # eval model
        y_pred_train = model.predict(x_train[list(features_set)])
        y_pred_test = model.predict(x_test[list(features_set)])
        # Print model report:
        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred_test)))

        test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
        train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

        precision, recall, fscore, support = score(y_test, y_pred_test)

        reversefactor = dict(zip(range(3), definitions))
        y_test_1 = np.vectorize(reversefactor.get)(y_test)
        y_pred = np.vectorize(reversefactor.get)(y_pred_test)

        print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

        print(metrics.classification_report(y_test_1, y_pred, digits=3))

        precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
            metrics.classification_report(y_test_1, y_pred, digits=3))

        results = results.append(
            {'Model': thresh,
             'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
             'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
             'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
             'Test_Accuracy': test_accuracy,
             'Train_Accuracy': train_accuracy},
            ignore_index=True)

    results.to_csv('Results/feature_selection_results.csv', sep=',', encoding='utf-8', index=False)


def start_train_models():
    results = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA','P_R', 'R_RR', 'R_DA','R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()

    X_train_scaled = scale_data_standardscaler(X_train[predictors])
    X_test_scaled = scale_data_standardscaler(X_test[predictors])


    for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_XGB_model(clf, X_train, y_train, X_test, name)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_SVM_model(clf, X_train_scaled, y_train, X_test_scaled, name)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train, X_test_scaled, name)
            else:
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train, y_train, X_test, name)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred_test)))

            test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

            precision, recall, fscore, support = score(y_test, y_pred_test)

            reversefactor = dict(zip(range(3), definitions))
            y_test_1 = np.vectorize(reversefactor.get)(y_test)
            y_pred = np.vectorize(reversefactor.get)(y_pred_test)

            print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test_1, y_pred, digits=3))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test_1, y_pred, digits=3))

            results = results.append(
                {'Model': name,
                 'P_RR': precision[1], 'P_DA': precision[0],'P_R': precision[2], 'R_RR': recall[1],
                 'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0],'f1_R': fscore[2],
                 'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                 'Test_Accuracy': test_accuracy,
                 'Train_Accuracy': train_accuracy},
                ignore_index=True)

    results.to_csv('Results/cross_project_results.csv', sep=',', encoding='utf-8', index=False)


def start_10_fold_validation(df_):
    results = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
    df = df_.sort_values(by=['PR_Date_Closed_At', 'PR_Time_Closed_At'], ascending=True)
    df_split = np.array_split(df, 11)
    print(df.shape)
    for index in range(len(df_split) - 1):
        train = pd.DataFrame()
        for i in range(index + 1):
            train = train.append(df_split[i])

        # print(f"Train dataset shape: {train.shape}")
        test = df_split[index + 1]
        # print(f"Test dataset shape: {test.shape}")

        X_train = train
        y_train = X_train[target]
        X_test = test
        y_test = X_test[target]

        X_train = X_train[predictors]
        X_test = X_test[predictors]

        X_train_scaled = scale_data_standardscaler(X_train)
        X_test_scaled = scale_data_standardscaler(X_test)


        classifiers = get_classifiers()

        for name, value in classifiers.items():
            clf = value
            print('Classifer: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_XGB_model(clf, X_train, y_train,
                                                                                     X_test)
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                     X_test_scaled)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train_scaled, y_train,
                                                                                       X_test_scaled)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob = train_RF_LR_model(clf, X_train,
                                                                                       y_train, X_test)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))


            test_accuracy = metrics.accuracy_score(y_test, y_pred)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

            precision, recall, fscore, support = score(y_test, y_pred)

            reversefactor = dict(zip(range(3), definitions))
            y_test_1 = np.vectorize(reversefactor.get)(y_test)
            y_pred_test = np.vectorize(reversefactor.get)(y_pred)

            print(pd.crosstab(y_test_1, y_pred_test, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test_1, y_pred_test, digits=3))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test_1, y_pred_test, digits=3))

            results = results.append(
                {'Model': name,
                 'P_RR': precision[1], 'P_DA': precision[0], 'P_R': precision[2], 'R_RR': recall[1],
                 'R_DA': recall[0], 'R_R': recall[2], 'f1_RR': fscore[1], 'f1_DA': fscore[0], 'f1_R': fscore[2],
                 'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                 'Test_Accuracy': test_accuracy,
                 'Train_Accuracy': train_accuracy},
                ignore_index=True)
    results.to_csv('Results/10_folds_results.csv', sep=',', encoding='utf-8', index=False



def start_each_project_model(df):
    results = pd.DataFrame(columns=['Model', 'Project', 'P_AAR', 'P_DA', 'P_DR', 'P_RAR', 'R_AAR', 'R_DA', 'R_DR', 'R_RAR',
                                    'f1_AAR', 'f1_DA', 'f1_DR', 'f1_RAR', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()

    for project in project_list:
        df_project = df.loc[df.Project_Name == project]
        print('Project {} is under processing'.format(project))

        X_test = df_project.loc[(df_project['PR_Date_Created_At'] >= start_date) & (df_project['PR_Date_Created_At'] <= end_date)]
        y_test = X_test[target]
        X_train = df_project.loc[(df_project['PR_Date_Created_At'] < start_date)]
        y_train = X_train[target]

        print("Total Train dataset size: {}".format(X_train[predictors].shape))
        print("Total Test dataset size: {}".format(X_test[predictors].shape))
        X_train, y_train = get_smote_under_sampled_dataset(X_train[predictors], y_train)
        X_train = X_train[predictors]
        X_test = X_test[predictors]
        X_train_scaled = scale_data_standardscaler(X_train[predictors])
        X_test_scaled = scale_data_standardscaler(X_test[predictors])

        for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_XGB_model(clf, X_train, y_train, X_test)
                # train_XGB_model_feature_selection(clf, X_train[predictors], y_train, X_test[predictors])
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_SVM_model(clf, X_train_scaled, y_train,
                                                                                          X_test_scaled)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train_scaled,
                                                                                            y_train, X_test_scaled)
            else:
                y_pred_train, y_predprob_train, y_pred_test, y_predprob = train_RF_LR_model(clf, X_train, y_train,
                                                                                            X_test)

            # Print model report:
            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred_test))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred_test)))

            test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
            train_accuracy = metrics.accuracy_score(y_train, y_pred_train)

            precision, recall, fscore, support = score(y_test, y_pred_test)

            reversefactor = dict(zip(range(4), definitions))
            y_test_1 = np.vectorize(reversefactor.get)(y_test)
            y_pred = np.vectorize(reversefactor.get)(y_pred_test)

            print(pd.crosstab(y_test_1, y_pred, rownames=['Actual PRs'], colnames=['Predicted PRs']))

            print(metrics.classification_report(y_test_1, y_pred, digits=4))

            precision_avg, recall_avg, fscore_avg = extract_metric_from_report(
                metrics.classification_report(y_test_1, y_pred, digits=4))
            # AAR, DA, DR, RAR = extract_each_class_metric_from_report(metrics.classification_report(y_test_1, y_pred, digits=4))
            try:
                results = results.append(
                    {'Model': name, 'Project': project,
                     'P_AAR': precision[1], 'P_DA': precision[0], 'P_DR': precision[3], 'P_RAR': precision[2],
                     'R_AAR': recall[1],
                     'R_DA': recall[0], 'R_DR': recall[3], 'R_RAR': recall[2], 'f1_AAR': fscore[1], 'f1_DA': fscore[0],
                     'f1_DR': fscore[3],
                     'f1_RAR': fscore[2],
                     'Avg_Pre': precision_avg, 'Avg_Recall': recall_avg, 'Avg_f1_Score': fscore_avg,
                     'Test_Accuracy': test_accuracy,
                     'Train_Accuracy': train_accuracy},
                    ignore_index=True)
            except IndexError:
                continue

    results.to_csv('Results/within_project_results.csv', sep=',', encoding='utf-8', index=False)


def calcuate_average_of_10_folds_1():
    df = pd.read_csv('Results/10_folds_results.csv')
    avg_result = pd.DataFrame(columns=['Model', 'P_RR', 'P_DA', 'P_R', 'R_RR', 'R_DA', 'R_R', 'f1_RR', 'f1_DA',
                                    'f1_R', 'Avg_Pre', 'Avg_Recall', 'Avg_f1_Score',
                                    'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        model_result = df.loc[df.Model == name]
        avg_result = avg_result.append(
            {'Model': name,
             'P_RR': model_result['P_RR'].mean(), 'P_DA': model_result['P_DA'].mean(), 'P_R': model_result['P_R'].mean(),
             'R_RR': model_result['R_RR'].mean(), 'R_DA': model_result['R_DA'].mean(), 'R_R': model_result['R_R'].mean(),
             'f1_RR': model_result['f1_RR'].mean(), 'f1_DA': model_result['f1_DA'].mean(), 'f1_R': model_result['f1_R'].mean(),
             'Avg_Pre': model_result['Avg_Pre'].mean(), 'Avg_Recall': model_result['Avg_Recall'].mean(),
             'Avg_f1_Score': model_result['Avg_f1_Score'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/10_folds_average.csv', sep=',', encoding='utf-8', index=False)



if __name__ == '__main__':

    print('Processing')

   
    # XGB_features_ranking(X_train, y_train)

    # train_XGB_model_feature_selection_2(X_train, y_train, X_test, y_test)

    start_train_models()

    # start_10_fold_validation(df)

    # start_each_project_model(df)

    # calcuate_average_of_10_folds_1()