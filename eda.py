# 1. GENEL RESIM

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.pandas.set_option('display.max_columns', None)
df = pd.read_csv("/Users/mvahit/Documents/GitHub/dsmlbc/datasets/train.csv")

df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


# 2. KATEGORIK DEGISKEN ANALIZI
df.Survived.unique()
df.Survived.value_counts()

# Kac kategorik değişken var ve isimleri neler?
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Kategorik Değişken Sayısı: ', len(cat_cols))
print(cat_cols)


more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10]
print('Kategorik Değişken Sayısı: ', len(more_cat_cols))
print(more_cat_cols)


# Hangi kategorik değişkenin kaç sınıfı var?
df[cat_cols].nunique()

# Kategorik Değişkenlerin Sütun Grafik İle Gösterilmesi
sns.countplot(x="Sex", data=df)
plt.show()


def cats_summary(data):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and data[col].dtypes == 'O']
    for var in cats_names:
        print(pd.DataFrame({var: data[var].value_counts(),
                            "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
        sns.countplot(x=var, data=data)
        plt.show()


cats_summary(df)


def cats_summary(data, categorical_cols, number_of_classes=10):
    var_count = 0  # Kaç kategorik değişken olduğu raporlanacak
    vars_more_classes = []  # Belirli bir sayıdan daha fazla sayıda sınıfı olan değişkenler saklanacak.
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # sınıf sayısına göre seç
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}),
                      end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cats_summary(df, cat_cols)


# 3. SAYISAL DEGISKEN ANALIZI

# Sayısal değişkenlere genel bakış:
df.describe().T
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

sns.boxplot(x=df["Age"])

# Veri setinde kaç sayısal değişken var?
num_cols = [col for col in df.columns if df[col].dtypes != 'O']
print('Sayısal değişken sayısı: ', len(num_cols))

# Sayısal değişkenlerin isimleri neler?
num_cols

# Veri setindeki id değişkeninden ve survived değişkeninden yukarıdaki kod ile nasıl kurtulabiliriz?
# Normal olarak nasıl kurtuluyorduk ki?

df.drop("PassengerId", axis=1).columns

# Önceki kod ile:
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and
            col not in "PassengerId" and
            col not in "Survived"]
num_cols

# Bir sayısal değişkenin dağılımını inceleyelim:
df["Age"].hist(bins=30)
plt.show()

sns.boxplot(x=df["Age"])
plt.show()


# Sayısal değişkenlerin hepsini otomatik olarak nasıl analiz ederiz?
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)

# Baska bir veri seti ile deneyelim:
df = pd.read_csv("eda/application_train.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and
            col not in "SK_ID_CURR " and
            col not in "TARGET"]

hist_for_nums(df, num_cols)

# İki sınıflı olanlar da geldi. Bunlardan pratik olarak nasıl kurtuluruz?
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and
            len(df[col].unique()) > 20 and
            col not in "SK_ID_CURR " and
            col not in "TARGET"]

hist_for_nums(df, num_cols)

# Bu değişkenlerin dağılımları çok da normal değil gibi.
# Eğer doğrusal bir model kullanacak olsaydık bu durumda bu değişkenlere logaritmik bir dönüşüm uygulamak gerekirdi.


# 4. TARGET ANALIZI
df = pd.read_csv("eda/titanic.csv")

# Survived değişkeninin dağılımını inceleyelim
df["Survived"].value_counts()

# KATEGORIK DEGISKENLERE GORE TARGET ANALIZI
# Nasıl yani? Kategorik değişkenlere göre grup by yapıp survived'ın ortalamasını alarak.
df.groupby("Sex")["Survived"].mean()

more_cat_cols = [col for col in df.columns if len(df[col].unique()) < 10]
print('Kategorik Değişken Sayısı: ', len(more_cat_cols))
print(more_cat_cols)


# Peki bunu tüm değişkenlere otomatik olarak nasıl yapabiliriz?
def target_summary_with_cat(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")


target_summary_with_cat(df, "Survived")

# SAYISAL DEGISKENLERE GORE TARGET ANALIZI
df.groupby("Survived").agg({"Age": np.mean})


def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target
                 and col not in "PassengerId"]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")


target_summary_with_nums(df, "Survived")

from functools import reduce

A = ["Veri", "Bilimi", "Okulu"]

print(reduce(lambda a, b: a + b, list(map(lambda x: x[0], A))))

# 5.SAYISAL DEGISKENLERIN BIRBIRLERINE GORE INCELENMESI

df = sns.load_dataset("tips")
df.head()

sns.scatterplot(x="total_bill", y="tip", data=df)
plt.show()

sns.lmplot(x="total_bill", y="tip", data=df)
plt.show()


df.corr()


