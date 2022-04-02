# VERIYI ANLAMAK
# KATEGORIK: value_counts(), countplot (sütun grafik)
# SAYISAL: describe(), boxplot, hist

# DATA PRE-PROCESSING & FEATURE ENGINEERING


# 1. AYKIRI DEGER ANALIZI (boxplot, threshold, sil, baskılamak)
# 2. EKSIK DEGER ANALIZI (silebiliriz, ortalama)
# 3. LABEL ENCODING (sex: k, e, 1,0)
# 4. ONE-HOT ENCODING (getdummies(drop_first))
# 5. SUPER-CATEGORY CATCHING
# 6. RARE ENCODING
# 7. STANDARDIZATION
# 8. FEATURE ENGINEERING
# 9. RECAP
# 10. ODEV


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
import sklearn
import sklearn.metrics

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# os.getcwd()

# os.chdir('/Users/mvahit/Documents/DSMLBC3/4_HAFTA/') md

# AYKIRI GOZLEMLERI YAKALAMAK

df = pd.read_csv("/Users/mvahit/Documents/GitHub/dsmlbc/datasets/train.csv")
df.head()


# Veri setini okutmak lazım olduğunda tekrar tekrar buraya gelmemek için bir fonksiyon yazalım:
def load_titanic():
    data = pd.read_csv("/Users/mvahit/Documents/GitHub/dsmlbc/datasets/train.csv")
    return data


df = load_titanic()


def load_application_train():
    df = pd.read_csv("datasets/application_train.csv")
    return df


df.describe().T

sns.boxplot(x=df["Age"]);

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

up

low

df[(df["Age"] < low) | (df["Age"] > up)][["Age"]].shape[0]

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")

low, up = outlier_thresholds(df, "Fare")

low

up


def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        print(variable, "yes")


has_outliers(df, "Age")

df.head()

num_names = [col for col in df.columns if len(df[col].unique()) > 10
             and df[col].dtypes != 'O'
             and col not in "PassengerId"]

num_names

df = load_application_train()

num_names = [col for col in df.columns if len(df[col].unique()) > 10
             and df[col].dtypes != 'O'
             and col not in "PassengerId"]

num_names

df = load_titanic()

num_names = [col for col in df.columns if len(df[col].unique()) > 10
             and df[col].dtypes != 'O'
             and col not in "PassengerId"]

num_names

for col in num_names:
    has_outliers(df, col)


# 1. Aykırı değer raporlamasını istiyoruz.
# 2. Aykırı değere sahip değişkenlerin box-plot'u oluşturulsun.
# 2. Bu boxplot özelliği kullanıcı tarafından biçimlendirilebilsin
# 3. Aykırı değere sahip olan değişkenlerin isimleri bir liste ile return edilsin.

def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []

    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)

        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]

            print(col, ":", number_of_outliers)
            variable_names.append(col)

            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()

    return variable_names


has_outliers(df, num_names)

has_outliers(df, num_names, plot=True)


# SILME

low, up = outlier_thresholds(df, "Age")

df[~((df["Age"] < low) | (df["Age"] > up))]

df.shape


def remove_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return df_without_outliers


df = remove_outliers(df, "Age")

df.shape

for col in num_names:
    new_df = remove_outliers(df, col)

new_df.shape
df.shape[0] - new_df.shape[0]
df = load_titanic()
df.head()


# BASKILAMA YONTEMI (re-assignment with thresholds)

low, up = outlier_thresholds(df, "Age")
df[((df["Age"] < low) | (df["Age"] > up))]["Age"]
df.loc[(df["Age"] > up), "Age"] = up
df[((df["Age"] < low) | (df["Age"] > up))]["Age"]
df.loc[(df["Age"] < low), "Age"] = low


# ÖDEV: BU İKİ İŞLEMİ TEK FONKSİYONLA YAPINIZ. LAMBDA.

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


low, up = outlier_thresholds(df, "Age")

replace_with_thresholds(df, "Age")

df[((df["Age"] < low) | (df["Age"] > up))]["Age"]

df = load_titanic()

has_outliers(df, num_names)

var_names = has_outliers(df, num_names)

var_names

for col in var_names:
    replace_with_thresholds(df, col)

has_outliers(df, num_names)

# TOPARLAYALIM
df = load_titanic()

num_names = [col for col in df.columns if len(df[col].unique()) > 10
             and df[col].dtypes != 'O'
             and col not in "PassengerId"]

outlier_thresholds(df, "Age")
has_outliers(df, num_names)
remove_outliers(df, "Age")
replace_with_thresholds(df, "Age")
has_outliers(df, num_names)

# DBSCAN VE LOF Yöntemlerini Araştırınız ve 1 Uygulama Yapınız.
# Density - Based Spatial Clustering of Applications with Noise.



# 2. EKSIK DEGER ANALIZI
# Yakala
# Rassallığını İncele (Ort kredi kartı harcama, kredi kartı var mı yok mu)
# Problemi Çöz (Sil, Basit Atama, Kırılımlara Göre Atama, Tahmine Dayalı Atama)) md

# EKSIK DEGERLERIN YAKALANMASI

df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# Oransal olarak görmek icin
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# Peki sadece eksik değere sahip değişkenlerin isimlerini yakalayabilir miyiz?
cols_with_na = [var for var in df.columns if df[var].isnull().sum() > 0]

cols_with_na


def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


cols_with_na = missing_values_table(df)

cols_with_na

# HIZLI ÇÖZÜM 1: DOKUNMA. NE ZAMAN? AĞAÇ KULLANIYORSAN, AGGREGATION COKSA, TEKILLESTIRME COKSA
# HIZLI COZUM 2: SİLME

df.dropna()

# HIZLI COZUM 3: BASIT ATAMA YONTEMLERI (MEAN,MEDIAN,MODE)
df = load_titanic()

df.shape

df["Age"].fillna(0)

df["Age"].fillna(df["Age"].mean(), inplace=True)

missing_values_table(df)

df = load_titanic()

df["Age"].fillna(df["Age"].median())

# df.apply(lambda x: x.fillna(x.mean()), axis=0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

missing_values_table(df)

df["Embarked"].mode()[0]

df = df["Embarked"].fillna(df["Embarked"].mode()[0])

df.head()

# df["Embarked"].fillna("missing")

df = load_titanic()

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# Scikit-learn ile eksik deger atama
# pip install scikit-learn

V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])

df = pd.DataFrame(
    {"V1": V1,
     "V2": V2,
     "V3": V3}
)

df

from sklearn.impute import SimpleImputer

df.columns

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(df)

a = imp_mean.transform(df)

a

pd.DataFrame(a, columns=df.columns)

# HIZLI COZUM 4:
# Kategorik Değişken Kırılımında Değer Atama
V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])
V4 = np.array(["IT", "IT", "IK", "IK", "IK", "IK", "IT", "IT", "IT"])

df = pd.DataFrame(
    {"maas": V1,
     "V2": V2,
     "V3": V3,
     "departman": V4}
)

df

df.isnull().sum()

df["maas"].mean()

df.groupby("departman")["maas"].mean()

df.groupby("departman")["maas"].mean()["IK"]

df.loc[(df["maas"].isnull()) & (df["departman"] == "IK"), "maas"] = df.groupby("departman")["maas"].mean()["IK"]

df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))

# TOPARLAYALIM

df = load_titanic()
missing_values_table(df)
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))


# GELISMIS ANALIZLER

df = load_titanic()

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

# 0.9 bir değişken arttığında diğeride çok şiddetli artar
# -0.9 bir değişken arttığında diğeri azalir. md

# EKSIK DEGERLERIN BAGIMLI DEGISKEN ILE ILISKISININ INCELENMESI

df.head()


def missing_vs_target(dataframe, target, variable_with_na):
    temp_df = dataframe.copy()

    for variable in variable_with_na:
        temp_df[variable + '_NA_FLAG'] = np.where(temp_df[variable].isnull(), 1, 0)

    flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for variable in flags_na:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(variable)[target].mean()}), end="\n\n\n")


cols_with_na

missing_vs_target(df, "Survived", cols_with_na)

# df["Embarked_NA_FLAG"].value_counts()

# ÖDEV: Tahmine Dayalı Atama Yöntemlerini Araştırınız.

# TOPARLAYALIM

df = load_titanic()
cols_with_na = missing_values_table(df)
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))
msno.heatmap(df)
plt.show()
missing_vs_target(df, "Survived", cols_with_na)

# 3. LABEL ENCODING
# - İki sınıflı olan kategorik değişlenlere 1-0 şeklinde label-encoder (binary encoding) uygula. (Cinsiyet)
# - Kategorik değişken nominal ise LABEL ENCODER UYGULANMAZ. (ONE-HOT ENCODING UYGULANIR)
# - Kategorik değişken ordinal ise LABEL ENCODER UYGULANABİLİR. (ONE-HOT ENCODING DE UYGULANABILIR)

df = load_titanic()

df.head()

df = load_titanic()

df.head()
df["Sex"].head()

from sklearn import preprocessing

len(df["Sex"].value_counts())

le = preprocessing.LabelEncoder()

le.fit_transform(df["Sex"])

le.inverse_transform([0, 1])


# kendimiz bir fonksiyon tanımlayalım:

def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder()

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and len(dataframe[col].value_counts()) == 2]

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe


df = label_encoder(df)

df["Sex"].head()

# 4. ONE-HOT ENCODING
#   - İkiden fazla sınıfa sahip olan kategorik değişkenlerin binary olarak encode edilmesi.

df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Sex"])

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()


def one_hot_encoder(dataframe, category_freq=10, nan_as_category=False):
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O']

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    return dataframe


df = load_titanic()

df.head()

df = one_hot_encoder(df)

df.head()

# LABEL ENCODING: 2 SINIFLI KATEGORIK DEGISKENLERE UYGULA
# ONE-HOT ENCODING: KATEGORIK DEGISKENLERE UYGULA(drop_first = True) md

# 5. SUPER CATEGORY CATCHING

df = load_application_train()

df.head()

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

cat_cols

df.groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False)

df.groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False).index[0]

df.groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False).index[
    len(df.groupby("OCCUPATION_TYPE")["TARGET"].mean()) - 1]

df.groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False).first_valid_index()

df.groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False).last_valid_index()

df["OCCUPATION_TYPE_SUPER_MIN_CAT"] = np.where(df["OCCUPATION_TYPE"].str.contains("Low-skill Laborers"), 1, 0)

df["OCCUPATION_TYPE_SUPER_MAX_CAT"] = np.where(df["OCCUPATION_TYPE"].str.contains('Accountants'), 1, 0)

df.head()

df["NAME_INCOME_TYPE"].value_counts()

tmp = df["NAME_INCOME_TYPE"].value_counts() / len(df)

tmp

freq_labels = tmp[tmp > 0.10].index

freq_labels

all_label = df["OCCUPATION_TYPE"].value_counts().index

all_label

labels = [label for label in all_label if label in freq_labels]

labels

df[df["OCCUPATION_TYPE"].isin(labels)]["OCCUPATION_TYPE"].value_counts()

df[df["OCCUPATION_TYPE"].isin(labels)].groupby("OCCUPATION_TYPE")["TARGET"].mean().sort_values(ascending=False)


def super_cat_catch(dataframe, target):
    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']

    for col in categorical_cols:
        tmp_df = dataframe[col].value_counts() / len(dataframe)
        freq_labels = tmp_df[tmp_df > 0.10].index
        all_label = dataframe[col].value_counts().index
        selected_labels = [label for label in all_label if label in freq_labels]

        if len(selected_labels) > 2:
            min_label = dataframe.groupby(col)[target].mean().sort_values(ascending=False).first_valid_index()
            max_label = dataframe.groupby(col)[target].mean().sort_values(ascending=False).last_valid_index()
            dataframe[col + "_SUPER_MIN" + min_label] = np.where(dataframe[col] == min_label, 1, 0)
            dataframe[col + "_SUPER_MAX" + max_label] = np.where(dataframe[col] == max_label, 1, 0)

    return dataframe


df_new = super_cat_catch(df, "TARGET")

df_new.head()


# 6. RARE ENCODING

df = load_application_train()
df.head()

df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
cat_cols


def cat_summary(data, cat_names):
    for var in cat_names:
        print(var, ":", len(data[var].value_counts()))
        print(pd.DataFrame({"COUNT": data[var].value_counts(),
                            "RATIO": data[var].value_counts() / len(data)}), end="\n\n\n")


cat_summary(df, cat_cols)

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# 1. Sınıf Frekansı
# 2. Sınıf Oranı
# 3. Sınıfların target açısından değerlendirilmesi
# 4. rare oranını kendimizin belirleyebilmesi


def rare_analyser(dataframe, target, rare_perc):

    rare_columns = [col for col in df.columns if df[col].dtypes == 'O'
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", 0.001)



temp_df = df.copy()
temp_df["ORGANIZATION_TYPE"].unique()
len(temp_df["ORGANIZATION_TYPE"].unique())
tmp = temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)
rare_labels = tmp[tmp < 0.01].index
len(rare_labels)
temp_df["ORGANIZATION_TYPE"] = np.where(temp_df["ORGANIZATION_TYPE"].isin(rare_labels), 'Rare',
                                        temp_df["ORGANIZATION_TYPE"])

len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)

rare_analyser(temp_df, "TARGET", 0.01)


# Evet şimdi bu işlemi genelleyerek bir fonksiyon yazalım:
def rare_encoder(dataframe, rare_perc):
    tempr_df = dataframe.copy()

    rare_columns = [col for col in tempr_df.columns if tempr_df[col].dtypes == 'O'
                    and (tempr_df[col].value_counts() / len(tempr_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = tempr_df[var].value_counts() / len(tempr_df)
        rare_labels = tmp[tmp < rare_perc].index
        tempr_df[var] = np.where(tempr_df[var].isin(rare_labels), 'Rare', tempr_df[var])

    return tempr_df


new_df = rare_encoder(df, 0.001)

rare_analyser(new_df, "TARGET", 0.001)


# 7. STANDARTLASTIRMA & DEĞİŞKEN DÖNÜŞÜMLERİ


# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
df = load_titanic()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])
df["Age"].describe().T

# RobustScaler: Medyanı çıkar iqr'a böl.
df = load_titanic()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])
df["Age"].describe().T

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
df = load_titanic()
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler((-10, 10)).fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])  # on tanımlı değeri 0 ile 1 arası.
df["Age"].describe().T

# Log: Logaritmik dönüşüm.
# not: - degerler varsa logaritma alınamayacağı için bu durum göz önünde bulundurulmalı.
df = load_titanic()
df["Age"] = np.log(df["Age"])
df["Age"].describe().T


# FEATURE ENGINEER


# FLAG, BOOL
df = load_titanic()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.head()

# LETTER COUNT
df["NEW_NAME_COUNT"] = df["Name"].str.len()

# WORD COUNT
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df["NEW_NAME_DR"].head()
df["NEW_NAME_DR"].mean()

df.groupby("NEW_NAME_DR").agg({"Survived": "mean"})
df["NEW_NAME_DR"].value_counts()

df["Age"].mean()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df.head()

df.groupby("NEW_TITLE").agg({"Age": "mean"})

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})


# NUMERIC TO CATEGORICAL
df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

# INTERACTIONS

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGExPCLASS"] = df["Age"] * df["Pclass"]

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df["Age"] = df["Age"].fillna(df.groupby("NEW_TITLE")["Age"].transform("median"))