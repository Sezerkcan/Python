
######################################################################
## GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
######################################################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi     (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi   (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi     (Analysis of Target Variables)
# 5. Korelasyon Analizi         (Analysis of Correlation)



#####################
### 1. GENEL RESİM
#####################

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = sns.load_dataset("titanic")
df.head()

def check_df(Dataframe, head = 5):
    print("################ shape #################")
    print(Dataframe.shape)
    print("######## information of dataframe ######")
    print(Dataframe.info())
    print("################ head ##################")
    print(Dataframe.head(head))
    print("################ tail ##################")
    print(Dataframe.tail(head))
    print("################ NA ####################")
    print(Dataframe.isnull().sum())
    print("######### Quantiles ####################")
    print(Dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

df = sns.load_dataset("tips")           #load_dataset üzerine ctrl ile gelerek diğer verisetlerine erişebiliriz.

check_df(df)
check_df()


######################################################################
### 1. Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################################################

df = sns.load_dataset("titanic")
df["embarked"].value_counts()
df["sex"].unique()          # eşsiz değerleri verir
df["sex"].nunique()         # eşsiz değer sayısını verir.


# kategorik değişkenleri fonksiyonel bir şekilde yakalamak istersek

cat_col = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]

# değişken türü içerisinde object ve category varsa bu durumda kategorik değişken olduğunu söylenebilir.
# ancak bazı durumlarda int ve float veri tipinde olan kategorik değişkenler de vardır. Bunları ayıklamak için
# eşsiz değer sorgusu yapılabilir. Böylelikle eşsiz değer sayısı 10' dan küçük olan 'int' ve 'float' değerleri kategorik
# değişken olarak görebiliriz. Örneğin name gibi nesne olarak tanımlanmış ancak çok sayıda eşsiz değer söz konusu olan
# değişkenlerde bir kategori söz konusu olamayacağından bunları da ayıklamak gerekir.

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category"]]
# kategorik ama kardinal

cat_col = cat_col + num_but_cat         # numerik görünen ama kategorik değişkenleri de kategorik değişkenlere ekliyoruz.

cat_col = [col for col in cat_col if col not in cat_but_car]       # eğer kategorik görünen ancak kategorik olmayanlar
                                                                   # listesinde de değişken varsa bu durumda onları da
                                                                   # göz önüne alıp dışındakileri seçtirmeliyiz.


df[cat_col].nunique()           # eşsiz değer sayıları kontrol edildiğinde değişkenlerin kategorik olduğu görülmektedir.

[col for col in df.columns if col not in cat_col]          # numerik değişkenleri listeler.

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)          # kategorileri yüzde olarak verir.

df = sns.load_dataset("titanic")

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################")

cat_summary(df, "sex")


for col in cat_col:                         # cat_summary fonksiyonunu tüm satırlara uyguluyoruz.
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print("########################################")

    if plot:
        sns.countplot(x=ndataframe[col_ame], data=dataframe)
        plt.show(block=True)            # fonksiyon döngüye sokulacağından dolayı blok argümanına true diyoruz.


cat_summary(df, "sex", plot=True)


# tüm kategorik değişkenlere uygulamak için bir for döngüsü kullanabiliriz.

for col in cat_col:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# adult_male bool tipinde olduğundan onu 1 ve 0 şeklinde sayısal bir forma getirmemiz gerekmektedir.


######################################################################
### 2. Sayısal Değişken Analizi (Analysis of Categorical Variables)
######################################################################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")

df.head()

for col in df.columns:
    if str(df[col].dtypes) in ["int64", "float64"]:
        print(col)

num_col = [col for col in df.columns if str(df[col].dtypes) in ["int64", "float64"]]

num_col = [col for col in num_col if col not in cat_col]


def num_summary(dataframe, numerical_col):
    Quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(Quantiles).T)

num_summary(df, "age")

for col in num_col:
    print(num_summary(df, col))

# sayısal değişkenleri görselleştirmek de istersek

def num_summary(dataframe, numerical_col, plot=False):
    Quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(Quantiles).T)

    if plot:
        df[numerical_col].hist()
        plt.title(numerical_col),
        plt.xlabel(numerical_col)
        plt.show(block=True)



for col in num_col:
    print(num_summary(df, col, plot=True))



#############################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# docstring

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)



# BONUS
df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


for col in num_cols:
    num_summary(df, col, plot=True)



#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

#######################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#######################


df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# burada kategorik değişkene("sex") göre hedef değişkenin("survived") ortalamasını alıyoruz ve çıktının
# daha güzel görünmesi ve işlemleri otomatikleştirmek adına bir fonksiyon tanımlaması yapıyoruz. Bu veri setinde kulla-
# cağımız hedef değişkenimiz "survived" değişkenidir.

target_summary_with_cat(df, "survived", "pclass")


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)



#######################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#######################

# bu sefer de hedef değişkenimize göre gruplama işlemi yapıp sayısal değişkenlere göre ortalama alma işlemi yapıyoruz.

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)



#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]               # 1. ve sonuncu sütunlar kullanılmayacağından iloc ile seçim işlemi yapıyoruz.
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

#NOT: Korelasyon, değişkenlerin birbirleriyle ilişkisini ifade eden istatistiksel bir ölçümdür. -1 ile +1 arasında değerler
# alır. +1' e yaklaştıkça ilişkinin şiddeti kuvvetlenir. eğer iki değişkenin arasındaki ilişki pozitif ise buna pozitif
# korelasyon denir ve bir değişkenin değeri arttıkça diğer değişkenin de değeri artar. İki değişken arasındaki ilişki ne-
# gatifse bir değişkenin değeri artarken diğer değişkenin değeri azalır.
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# NOT: her veri seti projesi için korelasyon problemiyle karşılaşmayabiliriz. Korelasyonun fazla olduğu 1' e yakın olduğu
# değişkenler farklılık yaratmadığından ve benzer özellikleri sahip olduğundan veri setimiz için problem oluşturmaması adına
# kurtulmamız gerekmektedir.

#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

cor_matrix = df.corr().abs()


#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN

# burada çoklama ve birbirini tekrar etme durumundan kurtulmak adına aşağıdaki fonksiyonu kullanırız.

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
# burada kolonları gezerken bu kolonlardakı değerlerden herhangibirisi 0.90' dan büyükse seç işlemi uyguluyoruz.


cor_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)
#df' deki korelasyona sahip değişkenler drop edilip daha sonra korelasyon fonksiyonuna sokuluyor.

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)

type(adsa)




