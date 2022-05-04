import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier




def split_keep_string(df,name):
    ser = df[name].copy()
    for i in range(len(ser)):
        curr_val = str(ser.iloc[i])
        tmp_l = curr_val.split(" ")
        ser.iloc[i] = tmp_l[0]
    df[name] = ser

def split_conv_float(df,name):
    ser = df[name].copy()
    for i in range(len(ser)):
        curr_val = str(ser.iloc[i])
        tmp_l = curr_val.split(" ")
        ser.iloc[i] = float(tmp_l[0])
    df[name] = ser
def clean_duration(df):
    ser = df["duration"].copy()
    for i in range(len(ser)):
        curr_val = str(ser.iloc[i])
        tmp_l = float(curr_val) / 60
        ser.iloc[i] = tmp_l
    df["duration"] = ser
def clean_date(df):
    ser = df["Date"].copy()
    for i in range(len(ser)):
        curr_val = str(ser.iloc[i])
        tmp_l = curr_val.split("/")
        ser.iloc[i] = tmp_l[2] + "-" + tmp_l[0] + "-" + tmp_l[1]
    df["Date"] = ser

def clean_weather(df):
    ser = df["activityType"].copy()
    for i in range(len(ser)):
        curr_val = str(ser.iloc[i])
        if curr_val == 'TraditionalStrengthTraining':
            ser.iloc[i] = 'Y'
        else:
            ser.iloc[i] = 'N'
    return ser

def mean_ser(ser,name):
    mean = ser.mean()
    if name == "Duration":
        print(name,"mean:",round(mean,2), " mins")
    elif name == "Total Energy Burned":
        print(name,"mean:",round(mean,2), "Kcals")
    elif name == "HKAverageMETs":
        print(name,"mean:",round(mean,2), "Kcals/hr * kg")

def std_ser(ser,name):
    std = ser.std()
    if name == "Duration":
        print(name,"standard derivative:",round(std,2))
    elif name == "Total Energy Burned":
        print(name,"standard derivative:",round(std,2))
    elif name == "HKAverageMETs":
        print(name,"standard derivative:",round(std,2))

def bar_chart_example(x_ser,y_ser,title,x,y):
    plt.figure() # to create a new current figure
    plt.bar(x_ser, y_ser)
    plt.xticks(rotation = 25, ha="right")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid()

def bar_chart_example_work(x_ser,y_ser,title,x,y):
    plt.figure() # to create a new current figure
    plt.bar(x_ser, y_ser)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid()

def pie_chart_example(x_ser,y_ser,title):
    plt.figure()
    plt.pie(y_ser,labels=x_ser, autopct="%.2f%%")
    plt.savefig("pie_example.png")

def get_total_counts_workouts(df):
    ser = pd.Series(dtype=float)
    for group_name,group_df in df:
        group_count = group_df["totalEnergyBurned"].count()
        print(group_name,"Workout Number:",group_df["totalEnergyBurned"].count())
        ser[group_name] = group_count
    return ser
def get_total_counts_weather(df,name):
    ser = pd.Series(dtype=float)
    for group_name,group_df in df:
        group_count = group_df[name].count()
        ser[group_name] = group_count
    return ser

def mean_of_attribute_in_group(df,name):
    ser = pd.Series(dtype=float)
    for group_name,group_df in df:
        group_mean = group_df[name].mean()
        ser[group_name] = group_mean
    return ser

def knn_classifier(merged_df):
    knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    scaler = MinMaxScaler()
    X = merged_df[["tavg","prcp","wspd"]].copy()
    y = merged_df["Workout"]
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.25, stratify=y)
    print(y_test.value_counts())
    knn_clf.fit(X_train, y_train)
    y_predicted = knn_clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_predicted)
    return accuracy


def decision_tree(merged_df):
    scaler = MinMaxScaler()
    tree_clf = DecisionTreeClassifier(random_state=0)
    X = merged_df[["tavg","prcp","wspd"]].copy()
    y = merged_df["Workout"]   
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.25, stratify=y)
    tree_clf.fit(X_train,y_train)
    y_predicted = tree_clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_predicted)
    return accuracy

def ttest_ind(ser1,ser2,alpha_val):
    t, pval = stats.ttest_ind(ser1,ser2)
    # divide by two because 1 rejection region
    print("t:", t, "pval:", pval)
    alpha = alpha_val
    if pval < alpha:
        print("reject H0")
    else:
        print("do not reject H0")

#utils.bar_chart_example(mean_groupby_data_ser.index,mean_groupby_data_ser,"Average Total Energy Burned","Months","Kcals")