# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:20:59 2021

@author: yazılım
"""


"""

metin madenciliği v1

"""

import pandas as pd



with open("yeni3.txt", "r") as dosya:
    metinler = dosya.readlines()



vektor= pd.Series(metinler)

mdf= pd.DataFrame(vektor,columns=["twitler"])
yeni_mdf=mdf.copy()

#küçük harf yapma

yeni_mdf=yeni_mdf["twitler"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#noktalama işaretleri siliniyor

type(yeni_mdf)

yeni_mdf=yeni_mdf.str.replace("[^\w\s]","")


#sayıların silinmesi

yeni_mdf=yeni_mdf.str.replace("\d","")

#stopwords mantığı

yeni_mdf=pd.DataFrame(yeni_mdf,columns=["twitler"])
yeni_mdf
!pip install nltk

import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords

sw=stopwords.words("turkish")
sw.append("m")
sw.append("replying to")
sw.append("h")
sw.append("to")
sw.append("replying")
sw.append("and")

yeni_mdf=yeni_mdf["twitler"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


#az geçen kelimelerin silinmesi (opsiyonel) kullanmayacağım şuan

yeni_mdf= pd.DataFrame(yeni_mdf, columns=["twitler"])

az_gecenler=pd.Series(" ".join(yeni_mdf["twitler"]).split()).value_counts()




#son hali ile ilk hali karşılaştırma

#ilk hali

mdf["twitler"][0:5]
#son hali
yeni_mdf["twitler"][0:5]

start = 2

#yeni_mdf["twitler"].str.find("kötü")


# Print the output.  
#print(df)  

with open("mutlukelimeler.txt","r", encoding="utf-8") as dosya:
    mutlu_kelimeler=dosya.readlines()

yeni_mutlu_kelimeler=[]
for i in mutlu_kelimeler:
    
    my_str = i[:-1]
    yeni_mutlu_kelimeler.append(my_str)

yeni_mdf=yeni_mdf["twitler"].dropna()
yeni_mdf.to_csv("adana.csv", encoding='utf-8',index=False)


df = pd.read_csv("adana.csv")
df=df.dropna()

df.loc[0,"status"]=""

durumlar=["poz","neg"]



sayac=1


while 1:
        
    bak_bakalim=df.iloc[sayac]
    
    kelime=bak_bakalim["twitler"]
    
    kelime_boluk=kelime.split()
    
    
    for i in kelime_boluk:
        #print(i)       
        if i in yeni_mutlu_kelimeler:
            print(i)
            df.loc[sayac,"status"] = "poz"
        else:
            df.loc[sayac,"status"] = "neg"
    sayac=sayac+1

df["status"].value_counts()

#kütüphane

from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,Lasso,Ridge,LassoCV,ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR,SVC
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df=df.dropna()

train_x, test_x,train_y,test_y=model_selection.train_test_split(
    df["twitler"],
    df["status"])


encoder= preprocessing.LabelEncoder()

train_y=encoder.fit_transform(train_y)
test_y=encoder.fit_transform(test_y)

#count vectors

vectorizer=CountVectorizer()
vectorizer.fit(train_x)

x_train_count = vectorizer.transform(train_x)

x_test_count = vectorizer.transform(test_x)


#TF-IDF

tf_idf_word_vectorizer=TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf_word= vectorizer.transform(train_x)
x_test_tf_idf_word= vectorizer.transform(test_x)


#ngram level

tf_idf_ngram_vectorizer=TfidfVectorizer(ngram_range=(2,3))
tf_idf_ngram_vectorizer.fit(train_x)


x_train_tf_idf_ngram=tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram=tf_idf_ngram_vectorizer.transform(test_x)

#karakter level

tf_idf_chars_vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(2,3))
tf_idf_chars_vectorizer.fit(train_x)

x_train_tf_idf_chars=tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars=tf_idf_chars_vectorizer.transform(test_x)





#tahmin

from sklearn import linear_model
loj = linear_model.LogisticRegression()
loj_model=loj.fit(x_train_count,train_y)

accuary= model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()


print("model doğruluk oranı %s dir"%accuary)

yorum=pd.Series("bu akşam çok kötü")

v=CountVectorizer()
v.fit(train_x)
yorum= v.transform(yorum)
loj_model.predict(yorum)
#0 çıkarsa kötü 1 çıkarsa iyi





























