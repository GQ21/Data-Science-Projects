import numpy as np
import re
import pandas as pd
pd.set_option("display.float_format", lambda x: "%.5f" % x)
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.ensemble import IsolationForest
RANDOM_STATE = 0 

def plot_count(
    df: pd.DataFrame, feature: str, label_rotation=False, fig_size=(12, 6)
) -> None:
    """Plots count plot of selected feature."""
    count = df[feature].value_counts()
    df_count = pd.DataFrame({feature: count.index, "Contracts": count.values})

    feature_perc = df[[feature, "TARGET"]].groupby([feature], as_index=False).mean()
    feature_perc.sort_values(by="TARGET", ascending=False, inplace=True)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=fig_size)
    plot_bar_a = sns.barplot(ax=ax1, x=feature, y="Contracts", data=df_count)
    plot_bar_b = sns.barplot(
        ax=ax2, x=feature, y="TARGET", order=feature_perc[feature], data=feature_perc
    )
    if label_rotation:
        plot_bar_a.set_xticklabels(plot_bar_a.get_xticklabels(), rotation=90)
        plot_bar_b.set_xticklabels(plot_bar_b.get_xticklabels(), rotation=90)
    plt.ylabel("Percent of Target = 1", fontsize=10)
    fig.suptitle(f"{feature} feature credit contract counts")


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Takes dataframe and returns dataframe which shows how many missing values every column has"""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    df_mis_val = pd.concat([mis_val, mis_val_percent], axis=1)
    df_mis_val_table_ren_columns = df_mis_val.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    df_mis_val_table_ren_columns = (
        df_mis_val_table_ren_columns[df_mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )
    return df_mis_val_table_ren_columns

def impute_missing_cat(
    train_data: pd.DataFrame, test_data: pd.DataFrame, categorical: list
) -> tuple:
    """Impute missing categorical variables with most frequent value"""
    train_data = train_data.copy()
    test_data = test_data.copy()
    # Impute train data
    imp_missing_cat = SimpleImputer(strategy="most_frequent")
    df_missing_train_cat = pd.DataFrame(
        imp_missing_cat.fit_transform(train_data[categorical]), columns=categorical
    )
    train_data_joined = pd.concat(
        [
            train_data.drop(columns=categorical).reset_index(drop=True),
            df_missing_train_cat,
        ],
        axis=1,
    )
    # Impute test data
    df_missing_test_cat = pd.DataFrame(
        imp_missing_cat.transform(test_data[categorical]), columns=categorical
    )
    test_data_joined = pd.concat(
        [
            test_data.drop(columns=categorical).reset_index(drop=True),
            df_missing_test_cat,
        ],
        axis=1,
    )

    return train_data_joined, test_data_joined

def impute_missing_num(
    train_data: pd.DataFrame, test_data: pd.DataFrame, numerical: list
) -> tuple:
    """Impute missing numerical features with median value"""
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Impute train data
    imp_missing_num = SimpleImputer(strategy="median")
    df_missing_train_num = pd.DataFrame(
        imp_missing_num.fit_transform(train_data[numerical]), columns=numerical
    )
    train_data_joined = pd.concat(
        [train_data.drop(columns=numerical).reset_index(drop=True), df_missing_train_num],
        axis=1,
    )

    # Impute test data
    df_missing_test_num = pd.DataFrame(
        imp_missing_num.transform(test_data[numerical]), columns=numerical
    )
    test_data_joined = pd.concat(
        [test_data.drop(columns=numerical).reset_index(drop=True), df_missing_test_num],
        axis=1,
    )

    return train_data_joined, test_data_joined

def encode_categorical(
    train_data: pd.DataFrame, test_data: pd.DataFrame, categorical: list
) -> tuple:
    """Takes train and test dataframes with categorical feature list. Encodes both dataframes with OneHotEncoder"""
    train_data = train_data.copy()
    test_data = test_data.copy()

    enc = OneHotEncoder(drop="first")
    train_data_encoded = pd.DataFrame(
        enc.fit_transform(train_data[categorical]).toarray(),
        columns=enc.get_feature_names(categorical),
    )
    train_data_joined = pd.concat(
        [train_data.drop(columns=categorical).reset_index(drop=True), train_data_encoded],
        axis=1,
    )

    test_data_encoded = pd.DataFrame(
        enc.transform(test_data[categorical]).toarray(),
        columns=enc.get_feature_names(categorical),
    )
    test_data_joined = pd.concat(
        [test_data.drop(columns=categorical).reset_index(drop=True), test_data_encoded],
        axis=1,
    )

    return train_data_joined, test_data_joined

def scale_minmax(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Takes train and test datasets, scales them with minmax scaler."""
    scaler = MinMaxScaler()

    train_data_bscale = train_data.drop(columns=["SK_ID_CURR", "TARGET"])
    test_data_bscale = test_data.drop(columns="SK_ID_CURR")

    train_data_scaled = pd.DataFrame(
        scaler.fit_transform(train_data_bscale),
        columns=train_data_bscale.columns,
    )
    test_data_scaled = pd.DataFrame(
        scaler.transform(test_data_bscale),
        columns=test_data_bscale.columns,
    )

    train_data_scaled["TARGET"] = train_data["TARGET"]
    train_data_scaled["SK_ID_CURR"] = train_data["SK_ID_CURR"]

    test_data_scaled["SK_ID_CURR"] = test_data["SK_ID_CURR"]

    return train_data_scaled, test_data_scaled

def get_binary_predictions(model,X_train:pd.DataFrame,X_val:pd.DataFrame) -> tuple:
    """With given model gets binary predictions."""
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    return y_pred_train,y_pred_val

def get_prob_predictions(model,X_train:pd.DataFrame,X_val:pd.DataFrame) -> tuple:
    """With given model gets probability predictions."""
    y_pred_train_prob = model.predict_proba(X_train)[:, 1]
    y_pred_val_prob = model.predict_proba(X_val)[:, 1]
    return y_pred_train_prob,y_pred_val_prob

def get_accuracy_scores(y_train:pd.DataFrame, y_pred_train:np.ndarray,y_val:pd.DataFrame, y_pred_val:np.ndarray) -> tuple:
    """Calculates accuracy score for train and validation sets."""
    accuracy_score_train = accuracy_score(y_train, y_pred_train)
    print("Training accuracy is", accuracy_score_train)
    accuracy_score_val = accuracy_score(y_val, y_pred_val)
    print("Validation accuracy is", accuracy_score_val)

    return accuracy_score_train,accuracy_score_val

def get_precision_scores(y_train:pd.DataFrame, y_pred_train:np.ndarray,y_val:pd.DataFrame, y_pred_val:np.ndarray) -> tuple:
    """Calculates precision score for train and validation sets."""
    precision_score_train = precision_score(y_train, y_pred_train)
    print("Training precision is", precision_score_train)
    precision_score_val = precision_score(y_val, y_pred_val)
    print("Validation precision is", precision_score_val)

    return precision_score_train,precision_score_val

def get_recall_scores(y_train:pd.DataFrame, y_pred_train:np.ndarray,y_val:pd.DataFrame, y_pred_val:np.ndarray) -> tuple:
    """Calculates recall score for train and validation sets."""
    recall_score_train = recall_score(y_train, y_pred_train)
    print("Training recall is", recall_score_train)
    recall_score_val = recall_score(y_val, y_pred_val)
    print("Validation recall is", recall_score_val)

    return recall_score_train,recall_score_val

def get_roc_auc_score(y_train:pd.DataFrame, y_pred_train_prob:np.ndarray,y_val:pd.DataFrame, y_pred_val_prob:np.ndarray) -> tuple:
    """Calculates ROC AUC score for train and validation sets."""
    roc_auc_score_train = roc_auc_score(y_train, y_pred_train_prob)
    print("Training ROC AUC is", roc_auc_score_train)
    roc_auc_score_val = roc_auc_score(y_val, y_pred_val_prob)
    print("Validation ROC AUC is", roc_auc_score_val)

    return roc_auc_score_train,roc_auc_score_val

def isolation_forest_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Search for outliers in specified dataframe and feature. Outputs dataframe with isolation forest scores and statistics"""
    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
    iso.fit(df[[feature]])
    pred = iso.predict(df[[feature]])
    scores = iso.decision_function(df[[feature]])
    stats = pd.DataFrame()
    stats["val"] = df[feature]
    stats["score"] = scores
    stats["outlier"] = pred
    stats["min"] = df[feature].min()
    stats["max"] = df[feature].max()
    stats["mean"] = df[feature].mean()
    stats["feature"] = [feature] * len(df)

    return stats

def print_outliers(df: pd.DataFrame, feature: str, n: int) -> None:
     """Prints feature name and specified count of samples statistics"""     
     print(feature)
     print(df[feature].head(n).to_string(), "\n")


def agg_cat(df_cat:pd.DataFrame, df_all:pd.DataFrame,prefix:str) -> pd.DataFrame:
    """
    Takes dataframe with only categorical features and second dataframe with all features.
    Encodes categorical features and aggregates them with mean.
    """        
    df_cat_agg = pd.get_dummies(df_cat, drop_first=True)
    df_cat_agg["SK_ID_CURR"] = df_all["SK_ID_CURR"]
    df_cat_agg = df_cat_agg.groupby("SK_ID_CURR").mean().reset_index()
    df_cat_agg.columns = [
        prefix + column + "_MEAN" if column != "SK_ID_CURR" else column
        for column in df_cat_agg.columns
    ]
    # Remove special characters from categorical features column names ( for LGBM model)
    df_cat_agg = df_cat_agg.rename(
        columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x)
    )
    return df_cat_agg

def agg_num_mean(df_num:pd.DataFrame,prefix:str) -> pd.DataFrame:
    """Takes dataframe with only numerical features and aggregates it with mean statistic"""       
    df_num_agg_mean = df_num.groupby(by=["SK_ID_CURR"]).mean().reset_index()
    df_num_agg_mean.columns = [
        prefix + column + "_MEAN" if column != "SK_ID_CURR" else column
        for column in df_num_agg_mean.columns
    ]
    return df_num_agg_mean

def agg_num_max(df_num:pd.DataFrame,prefix:str) -> pd.DataFrame:
    """Takes dataframe with only numerical features and aggregates it with max statistic"""    
    df_num_agg_max = df_num.groupby(by=["SK_ID_CURR"]).max().reset_index()
    df_num_agg_max.columns = [
        prefix + column + "_MAX" if column != "SK_ID_CURR" else column
        for column in df_num_agg_max.columns
    ]
    return df_num_agg_max

def agg_num_min(df_num:pd.DataFrame,prefix:str) -> pd.DataFrame:
    """Takes dataframe with only numerical features and aggregates it with min statistic""" 
    df_num_agg_min = df_num.groupby(by=["SK_ID_CURR"]).min().reset_index()
    df_num_agg_min.columns = [
        prefix + column + "_MIN" if column != "SK_ID_CURR" else column
        for column in df_num_agg_min.columns
    ]
    return df_num_agg_min

def agg_num_sum(df_num:pd.DataFrame,prefix:str) -> pd.DataFrame:
    """Takes dataframe with only numerical features and aggregates it with sum statistic""" 
    df_num_agg_sum = df_num.groupby(by=["SK_ID_CURR"]).sum().reset_index()
    df_num_agg_sum.columns = [
        prefix + column + "_SUM" if column != "SK_ID_CURR" else column
        for column in df_num_agg_sum.columns
    ]
    return df_num_agg_sum

def aggregate_samples(data: pd.DataFrame, prefix:str) -> pd.DataFrame:
    """
    Takes dataframe, splits it to two dataframes containing numerical and encoded categorical features
    .Numerical dataset groups by SK_ID_CURR feature and aggregates samples by mean, max, min statistics.
    Categorical dataset is processed same way except samples are aggregated by only mean statistics.Finaly
    these dataframes are merged and returned as one dataset.
    """
    df = data
    numerical_col = [
        feature for feature in df.columns if df[feature].dtype in ["int64", "float64"]
    ]
    categorical_col = [
        feature for feature in df.columns if df[feature].dtype == "object"
    ]

    if len(categorical_col) != 0:
        df_cat = df[categorical_col]
        df_cat_agg = agg_cat(df_cat, df, prefix)

    df_num = df[numerical_col]
    df_num_agg_mean = agg_num_mean(df_num,prefix)
    df_num_agg_max = agg_num_max(df_num,prefix)
    df_num_agg_min = agg_num_min(df_num,prefix) 
    df_num_agg_sum = agg_num_sum(df_num,prefix)       

    # Merge aggregateted features
    df_agg = df_num_agg_mean.merge(df_num_agg_max, how="left", on="SK_ID_CURR")
    df_agg = df_agg.merge(df_num_agg_min, how="left", on="SK_ID_CURR")
    df_agg = df_agg.merge(df_num_agg_sum, how="left", on="SK_ID_CURR")

    if len(categorical_col) != 0:
        df_agg = df_agg.merge(df_cat_agg, how="left", on="SK_ID_CURR")

    return df_agg

def get_importance(model, data:pd.DataFrame, num=30) -> pd.DataFrame:
    """Takes sklearn model and finds top 20 most important features"""
    feature_imp = pd.DataFrame(
        {"Feature": data.columns, "Weight": model.feature_importances_}
    )
    feature_imp = feature_imp.sort_values(by="Weight", ascending=False)
    return feature_imp[:num]