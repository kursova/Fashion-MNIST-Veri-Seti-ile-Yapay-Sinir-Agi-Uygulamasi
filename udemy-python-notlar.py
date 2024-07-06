# %% [code]
# %% [code]
# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

liste=[10,20,30]
liste
liste.sort()

?print
#fonksiyon yazma 

def kare_al(x):
    print('girilen sayının karesi: '+ str(x**2))

kare_al(3)

def carpma_yap(x, y):
    print(x*y)
    

sinir=5000
magaza_adi=input("magaza adi nedir")
gelir=input("gelirinizi girin")

if gelir<--if döngüsü yaz

ogrenci=["ali", "veli", "isik", "berk"]

for i in ogrenci:
    print(i)

maaslar=[100,200,800,50,500]

maaslar.sort()
maaslar

for i in maaslar:
    if i==200:
        print ("kesildi")
        break 
    print(i)

    for i in maaslar:
    if i==200:
        print ("devam")
        continue 
    print(i)

sayi=1

while sayi<10:
    sayi+=1
    print(sayi)

class VeriBilimci():
    calisanlar=[]
    def __init__(self):
        self.bildigi_diller=[]
        self.bolum=''
    def dil_ekle(self,yeni_dil):
        self.bildigi_diller.append(yeni_dil)
    
ali=VeriBilimci()

ali.bildigi_diller.clear()

ali.dil_ekle("R")

dir(VeriBilimci)

dir(append)


a=[1,2,3,4]
b=[2,4,8,2]
ab=[]

for i in range(0,len(a)):
    ab.append(a[i]*b[i])
print(ab)   

#bunun yerine aşağıdakini yapıcaz

import numpy as np

a=np.array([1,2,3,4])
b=np([2,4,8,2])
a.b

np.zeros(10, dtype=int)
np.ones(10, dtype=int)
np.ones((3,5), dtype=int)

np.full((3,5),3)

np.linspace(0,1,10)

np.random.normal(10,4,(3,4))
np.random.randint(0,10,(3,3))

np.arange(0,30,3)
a=np.random.randint(10,size=10)
a.ndim
a.shape
a.size
a.dtype

np.arange(1,10)
np.arange(1,10).reshape((3,3))
a=np.arange(1,10) #tek boyutlu array vektör , iki boyutluysa matris
a.reshape(1,9) #burada dönüşümü yaptığımızda 2 tane köşeli parantez görüyoruz.

x=np.array([1,2,3])
y=np.array([4,5,6])

np.concatenate([x,y])

#iki boyut
#not. iki boyutlu array oluştururken 2 tane köşel parantez açarak başka. aşağıdaki gibi 
a=np.array([[1,2,3],
            [4,5,6]])
np.concatenate([a,a])

np.concatenate([a,a], axis=0) # satır bazında birleştirme

np.concatenate([a,a], axis=1) # sütun bazında birleştirme

x=np.array([1,2,3,99,99,3,2,1])
x.ndim
np.split(x,[3,5])

a,b,c=np.split(x,[3,5])
a,b,c

m=np.arange(16).reshape(4,4)
m

ust,alt=np.vsplit(m,[2]) #2.satıra kadar böl
ust
alt

n=np.hsplit(m,[2]) #sütuna göre böldü
n

v=np.array([2,3,4,1,8,10])
# önemli!! np.sort dersek yapısı kalıcı olarak bozulmaz ama atama yaptıktan sonra (v.sort) sort dersek bozulur. 

np.sort(v) 
v #kalıcı olarak değişmedi 

v.sort()
v #kalıcı olarak değişti.

#çok boyutlularda sıralayalım 
m=np.random.normal(20,5,(3,3))
np.sort(m,axis=0)
#indeksler
a=np.random.randint(10, size=(3,5))
a
a[0,1]
a[1,4]=2.2 #var olan bir arraye dışarıdan yeni bir tip eklersek değşiklik yapmaz ama yeni array oluştururken birini float verirsek hepsini float yapar.

a=np.arange(20,30)
a
a[1::2] # a[1]den başka 2şer ikişer yaz

#iki boyutlarda slice 
m=np.random.randint(10,size=(5,5))
m
m[:,0]
m[:,4]
m[0,:]
m[0,0:2]
m[0,2]
m[0:2, 0:3]

#alt küme işlemleri 

a=np.random.randint(10,size=(5,5))
a
alt_a=a[0:3,0:2]
alt_a[0,0]=9999
alt_a
a #alt kümede yapılan değişiklik ana kümeyi de etkiledi 

#bunun için copy kullanırız

alt_a=a[0:3,0:2].copy()
alt_a
alt_a[0,0]=1
alt_a
a

#fancy index
v=np.arange(0,30,3)
v
al_getir=[1,3,5]
v[al_getir] # istediğimiz birden fazla indeksi tek sorugyla getirdik

#iki boyutlu fancy 
m=np.arange(9).reshape((3,3))
m
satir=np.array([0,1])
sutun=np.array([1,2])
m[satir,sutun]
m[0:,[1,2]]

#koşullu eleman işlemleri 
v=np.array([1,2,3,4,5])
v>5
v<3
v[v<3] #burada da fancy kullandık

v[v!=3]

?np

import numpy as np
#numpy ile iki bilinmeyelenli 
5*x0 + x1 = 12
x0 + 3*x1 = 10
a=np.array([[5,1],
            [1,3]]) # xin katsayıları 

b=np.array([12,10])
x=np.linalg.solve(a,b)
x


np.random.rand(5)
np.random.randint(5,size=10)



#pandas 
import pandas as pd
seri=pd.Series([1,2,3,4,5])

type(seri)

seri.axes
seri.dtype
seri.size
seri.ndim 
seri.values
seri.head(3)
seri.tail()
#index isimlendirme 
pd.Series([99,22,332,94,5])
pd.Series([99,22,332,94,5], index=[1,3,5,7,9])
seri=pd.Series([99,22,332,94,5], index=["a","b","c", "d", "e"])
seri["a":"c"]

#iki seriyi birleştirme 
seri=pd.concat([seri, seri])

seri

seri.index
seri.keys
list(seri.items())
seri.values
"a" in seri # eleman sorgulama 

seri[["a", "b"]]

import numpy as np
import pandas as pd
l=[1,2,39,67,90]
pd.DataFrame(l,columns=['degisken_ismi'])

m=np.arange(1,10).reshape((3,3))
m

df=pd.DataFrame(m,columns=['var1','var2', 'var3'] )
pd.columns=['den1', 'den2', 'den3']

df.axes
df.ndim
df.values
type(df.values) #bunu numpya çevirdi


s1=np.random.randint(10,size=5)
s2=np.random.randint(10,size=5)
s3=np.random.randint(10,size=5)

sozluk={'var1': s1, 'var2': s2, 'var3': s3}

sozluk

df=pd.DataFrame(sozluk)
df[0:1]

df.index
df.index=["a", "b", "c", "d", "e"]
df

#silme 
df.drop('a', axis=0) #böyle kullanırsak silme işlemi kalıcı olmaz
df

df.drop('a', axis=0, inplace=True) # inplace true eklersek kalıcı olur 
df

#fancy ile birden fazla satır silebiliriz

'var1' in df

l=['var1', 'var2', 'var4']

for i in l:
    print(i in df)
    
df['var4']=df['var1']/df['var2']

df.drop('var4', axis=1, inplace=True)

df

#gözlem ve değişken seçimi (loc / iloc)

import numpy as np
import pandas as pd

m=np.random.randint(1,30,size=(10,3))
df=pd.DataFrame(m, columns=['var1','var2', 'var3'])
df

#loc

df.loc[0:3] #3 dahil

df.iloc[0:3] #3 hariç
 
df.loc[0:3, 'var3'] #düzgün çalışır
df.iloc[0:3, 'var3'] #hata alır. aynı köşeli parantez içerisinde "var" yazamayız bunun için loc kullanmak gerek ya da aşağıdaki gibi kullanmak lazım 
df.iloc[0:3]['var3']

df[0:2][['var1', 'var2']] # burada ikincisinde çift köşeli parantez kullandık çünkü fancy kullanımı yaptık

df.var1 # bu da kullanılır

df[df.var1>15]['var2']

df[(df.var1>15) & (df.var3<5)]

df.loc[(df.var1>15), ['var2','var1']] # ya da 
df[(df.var1>15)] [['var2','var1']]

m=np.arange(1,10).reshape((3,3))
m

df=pd.DataFrame(m,columns=['var1', 'var2', 'var3'])
df
df.columns=('deg1', 'deg2', 'deg3')

df.ndim
df.axes
df.size


s1=np.random.randint(10,size=5)
s2=np.random.randint(10,size=5)
s3=np.random.randint(10,size=5)

sozluk={'var1': s1, 
        'var2':s2, 
        'var3':s3}
df=pd.DataFrame(sozluk)
df

df.index=["a", "b","c","d","e"]
df.drop("a", axis=0, inplace=True) #axis =0 yazmasak da aynı ancak sütun düşürmek istersek axis=1 yazmalıydık // inplace kalıcı olmasını sağlar 


df2=df+7
df

pd.concat([df,df2], ignore_index=True, join='inner') #birleştirme ignore_index=True yaptığımızda indeks problemi kalmaz

pd.concat([df,df2], join_axes=[df2.columns]) #join axes ile hangi tablonun kolonlarını alsın onu yazdık. left right birleştirme gibi

import pandas as pd
import numpy as np
df1=pd.DataFrame({"calisanlar": ["ali", "veli", "ayse", "fatma"],
    "grup": ["muhasebe", "muhendislik", "muhendislik", "ik"]})

df2=pd.DataFrame({"calisanlar": ["ayse", "ali", "veli",  "fatma"],
     "ilk_giris": [2010,2009,2014,2019]})


df1
df2

pd.merge(df1,df2, on="calisanlar")

import seaborn as sns

df=sns.load_dataset("planets")
df

df.describe().T
df.dropna().describe().T


df=pd.DataFrame({"gruplar": ["A", "B", "C","A", "B", "C"],
                "veri": [10,11,22,35,80,60], 
                "donem": [2020,2021,2022,2023,2024,2025]},
               columns=["gruplar", "veri", "donem"])

df

df.groupby("gruplar").mean() # tek bir değişken olduğu için böyle yapabildik.


df.groupby("method")["orbital_period"].mean() #birden fazla değişken varsa önce neyi gruplamak istediğimizi sonra hangi  sütunun değerini bulmak istediğimizi bu şekilde buluruz.

df.groupby("method")["orbital_period"].describe()

df.groupby("method").aggregate([min, np.mean,max])

df.groupby("method")["orbital_period"].aggregate([min, np.mean,max]) #aggregate bize özel bir işlem yapmamızı sağlar. yani biz burda mean vs değil kendi oluşturduğumuz bir grup işleme göre 
#gruplandırma yapmak istiyoruz. aggregate yazıp liste şeklinde içerisine hangi işlemleri yapmasını istediğimizi yazdık. min ve max pandas içerisinde var zaten ancak mean işlemini
#numpy içinden çağırıyoruz. bu nedenle başına np koyduk. ayrıca min ve maxı pandas içinde olduğu için tırnak içine de alabilirdik.

df.groupby("method").aggregate([min, np.mean,max])
df

df.groupby("method").aggregate([min, np.mean,max])


df.groupby("method").aggregate({"orbital_period": "min",
                               "mass": "min"})


df.groupby("gruplar").aggregate({"veri": min, "donem": max})

df=df[:,1:]
df.transform(lambda x: x-x.mean())