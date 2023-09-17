###########################################
# Veri Görselleştirme: Matplotlib & Seaborn
###########################################

#### Matplotlib

# Düşük seviyeli bir veri görselleştirme aracıdır.
# Kategorik değişkenler sütun grafikleriyle görselleştirilir. Bunu seaborn içerisindeki countplot ya da matplotlib
# içerisindeki barplot ile gerçekleştirebiliriz.
# Sayısal değişkenler histogram ve boxplot(kutu grafiği) ile görselleştirilir.


# NOT: Python veri görselleştirme için en uygun araç değildir. Veri tabanlarına bağlı veri görselleştirme araçları ve
# iş zekası araçları(Power BI, ClickView, tableau vs.) daha kullanışlıdır.


###################################
# Kategorik değişken görselleştirme
###################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df['sex'].value_counts()
df['sex'].value_counts().plot(kind='bar')           # cinsiyet değişkeninin sayısal değerlerine göre bir sütun grafiği
plt.show()                                          # oluşuturur.


###################################
# Sayısal değişken görselleştirme
###################################

plt.hist(df["age"])                     ## hist() metodu istediğimiz değişkene göre bir histogram grafiği oluşturmamızı sağlar.
plt.show()                              ## age değişkeni bazı kişiler için NaN yani eksik değere sahip olduğu için
                                        # histogram çizdirilirken hata verir bunu önlemek için NaN değerleri atmak
                                        # gerekir.

plt.hist(df['age'].dropna().values)
plt.show()

plt.boxplot(df["fare"])                 # Kutu grafiği veri setindeki aykırı değerleri çeyreklik değerler üzerinden yakalar.
plt.show()


##########################
# Matplotlib Özellikleri
##########################

# Matplotlib yapısı itibariyle katmanlı bir şekilde veri görselleştirme imkanı sağlar.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


## plot, veriyi görselleştirmek için kullandığımız fonksiyonlardan bir tanesidir.

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')                 # Noktalara sembol koyar ancak önceki grafiği kapatmazsak grafikler üst üste ça-
plt.show()                          # lışacağından sembolü eğrinin üzerine koyar.


x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])


plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()



## marker, işaret özellikleridir. Kullanabileceğimiz markerlar ["o", "*", ".", ",", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h"]


y = np.array([13, 28, 11, 100])
plt.plot(y, marker="o")
plt.show


## line

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashed")                     # dashed: kesikli çizgi, dotted: noktalı, dashdot: noktalı kesikli
plt.show()

                                                    # renk eklemek için color kullanılabilir.

plt.plot(y, linestyle="dotted", color="r")          # dashed: kesikli çizgi, dotted: noktalı, dashdot: noktalı kesikli
plt.show()



######################
# Multiple Lines
######################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])

plt.plot(x)
plt.plot(y)

plt.show()                              # plt.show() kullanmadan da çizdiriyor ancak jupyternb gibi diğer araçlarda çiz-
                                        # dirmediğinden kullanmak önemlidir.


######################
# Labels
######################

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)


# Grafiğin başlığı

plt.title("Bu ana başlık")

# x ve y ekseni isimlendirmeleri

plt.xlabel("X ekseni isimlendirmesi")

plt.ylabel("Y ekseni isimlendirmesi")

plt.grid()          # grafiğe ızgara ekleme



######################
# Subplots
######################

# yanyana grafikler koymak için kullanılır 1:X şeklinde bir ölçekleme (bölme) yapılabilir.

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)                # 1 ekrana iki tane olacak şekilde yerleştiriyoruz ve ,1 ile ilk tarafının grafiği
plt.title("1")                      # çizdirilir.
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)
plt.show()



# 3 grafiği bir satır 3 sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()



##################################
# Seaborn ile veri görselleştirme
##################################

# Seaborn matplotlib' e göre daha az çabayla daha fazla işlem yapmamızı sağladığından daha yüksek seviyeli bir kütüphanedir.


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)                 # değişken ismini x ile işaretliyoruz. kullanılacak veriyi de data= ile
plt.show()                                                 # yerleştiriyoruz.

# matplotlib kullanarak aynı işlemi yapalım

df["sex"].value_counts().plot(kind='bar')
plt.show()


# sayısal değişken görselleştirme

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

sns.scatterplot(x=df["tip"], y=df["total_bill"], hue=df["smoker"], data=df)
plt.show()
