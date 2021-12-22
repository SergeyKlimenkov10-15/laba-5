#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle


# In[1]:


import warnings
warnings.filterwarnings('ignore')


# <hr>
# Загрузим DataFrame, уберем ненужные столбцы (ID и пустой)

# In[3]:


df = pd.read_csv('data.csv')
df = df.drop('Unnamed: 32', axis=1).drop("id", axis=1)

df.head()
# malignant - "bad"
# benign - "good"


# Убедимся, что в датасете нет ячеек с NaN

# In[4]:


df.isnull().values.any()


# Посмотрим типы данных, наименования столбцов

# In[5]:


df.info()


# | №  | Название          | Перевод             | Толкование                                                                                                                                                                                                                                                               |
# | -- | ----------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | 0  | diagnosis         | диагноз             | диагноз                                                                                                                                                                                                                                                                  |
# | 1  | Radius            | Радиус              | Расстояние от центра до периферии                                                                                                                                                                                                                                        |
# | 2  | Texture           | Размер              | Размер раковых клеток.                                                                                                                                                                                                                                                   |
# |    |                   |                     | Однородность: Обозначает локальную изменчивость длины радиуса. Однородность это разность между длиной радиуса и средним значением длины линий вокруг него..                                                                                                              |
# | 3  | Perimeter         | Периметр            |  Величина основной опухоли.                                                                                                                                                                                                                                              |
# |    |                   |                     | Общее расстояние между точками дают периметр.                                                                                                                                                                                                                            |
# | 4  | Area              | Зона                | Область раковых клеток                                                                                                                                                                                                                                                   |
# | 5  | Smoothness        | Однородность        | Обозначает локальную изменчивость длины радиуса. Однородность это разность между длиной радиуса и средним значением длины линий вокруг него.                                                                                                                             |
# | 6  | Compactness       | Компактность        | Это мера сравнения периметра и размера, находится как (периметр^2/размер – 1.0)                                                                                                                                                                                          |
# | 7  | Concavity         | Вогнутость          | Глубина впадин. На меньших струнах чаще образуются меньшие впадины. Это связано с их длиной.                                                                                                                                                                             |
# |    |                   |                     | Количество впадин: Вогнутость измеряет частоту впадин контура, в то время как этот параметр измеряет их количество.                                                                                                                                                      |
# | 8  | Concave points    | Количество впадин   | Вогнутость измеряет частоту впадин контура, в то время как этот параметр измеряет их количество.                                                                                                                                                                         |
# | 9  | Symmetry          | Симметрия           | Самая длинная струна берётся за основную ось. Разница длин основной оси и линии перпендикулярной ей называется симметрией                                                                                                                                                |
# | 10 | fractal dimension | Фрактальные размеры | Показатель нелинейного роста. По мере увеличения масштаба измерения точность снижается и, как следствие, уменьшается значение периметра. Эти данные, представленные в виде графика кривой с отрицательным наклонном, дают приблизительное значение фрактальных размеров. |

# **Mean** – среднее значение
# **SE(standard error)** – обычные ошибки
# **Worst** – худший результат
# 

# # Немного графиков
# ## Отношение статистики по доброкачественным и злокачественым опухалям

# In[6]:


sns.set_style('darkgrid')
plt.figure(figsize=(10,5))
sns.countplot(x="diagnosis", data=df, palette='rocket');
pd.value_counts(df['diagnosis'])


# Отношение количества злокачественных(malignant) к доброкачественным(benign) оп

# ## Построение тепловой карты корреляции значений

# In[7]:


plt.figure(figsize=(25, 20))
plt.title("CORRELATION HEATMAP")
matrix = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, linewidth=.25, mask=matrix, cmap="coolwarm");


# Тепловая карта кореляций всех значений

# Распределения радиуса относительно размера. Так как Однородность: Обозначает локальную изменчивость длины радиуса. Однородность - это разность между длиной радиуса и средним значением длины линий вокруг него.

# In[8]:


sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df,palette='rocket')


# Распределение однородности относительно компактности, так как оба этих параметра связаны с радиусом

# In[9]:


sns.scatterplot(x='smoothness_mean', y='compactness_mean', data=df, hue='diagnosis')


# Распределение периметра относительно области поражения 

# In[10]:


sns.lmplot(x='perimeter_mean', y='area_mean', data=df, hue='diagnosis')


# Распределение однородности относительно компактности 

# In[11]:


sns.lmplot(x='smoothness_mean', y='compactness_mean', data=df, hue='diagnosis')


# ##  Данные можно разделить прямой и получить относительно корректный результат

# In[12]:


fig, table =  plt.subplots(1, 2, figsize=(20,6))
sns.boxplot(ax = table[0], x=df.diagnosis, y=df['area_mean'])
table[0].set_title('Area')

sns.boxplot(ax = table[1], x=df.diagnosis, y=df['perimeter_mean'])
table[1].set_title('Perimeter')

plt.show()


# In[13]:


sns.lmplot(x='area_mean', y='perimeter_mean', data=df, hue='diagnosis')


# In[14]:



bins = 20 #Number of bins is set to 20, bins are specified to divide the range of values into intervals
def histogram(features):
  plt.figure(figsize=(10,15))
  for i, feature in enumerate(features):
      plt.subplot(5, 2, i+1)  #subplot function: the number of rows are given as 5 and number of columns as 2, the value i+1 gives the subplot number, subplot numbers start with 1
      sns.distplot((df[df.diagnosis=='M'].drop("diagnosis",axis=1))[feature], bins=bins, color='red', label='Malignant');
      sns.distplot((df[df.diagnosis=='B'].drop("diagnosis",axis=1))[feature], bins=bins, color='green', label='Benign');
      plt.title(str(' Density Plot of:  ')+str(feature))
      plt.xlabel('X variable')
      plt.ylabel('Density Function')
      plt.legend(loc='upper right')
  plt.tight_layout()
  plt.show()


# In[2]:


df.columns


# <hr>
# <h2>Mean features (Какая-нибудь недокументированная доп. возможность)</h2>

# In[16]:


mean_features = list(filter(lambda x: "mean" in x, df.columns))  #['radius_mean', 'texture_mean']
histogram(mean_features)


# <hr>
# <h2>Error features</h2>

# In[17]:


error_features = list(filter(lambda x: "_se" in x, df.columns))  #['radius_mean', 'texture_mean']
histogram(error_features)


# <hr>
# <h2>Worst features</h2>

# In[18]:


worst_features = list(filter(lambda x: "worst" in x, df.columns))  #['radius_mean', 'texture_mean']
histogram(worst_features)


# ## Нормальное распределение ⇒ "обрезать" данные не требуется
# 

# ##### Преобразуем Malignant и Benign в числа

# In[19]:


# M - 1
# B - 0

print("B:", df.iloc[20]['diagnosis'])
print("M:", df.iloc[0]['diagnosis'])

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

print("B:", df.iloc[20]['diagnosis'])
print("M:", df.iloc[0]['diagnosis'])


# In[20]:


ss = StandardScaler()
cols = list(df)[1:]
df[cols] = ss.fit_transform(df[cols])


# ## Разделяем выборку (датасет) на 3 части
# <img src="hhttps://www.google.com/url?sa=i&url=https%3A%2F%2Fmemepedia.ru%2Freznya%2F&psig=AOvVaw2IK7XhJDZA93am9iEz25D8&ust=1640275168429000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCIjY2afj9_QCFQAAAAAdAAAAABAI" alt="Drawing" style="width: 200px;"/>
# <hr>

# In[21]:


#street magic
train, test, validate = np.split(df.sample(frac=1, random_state=42), 
                       [int(.6*len(df)), int(.8*len(df))])


# ##### Отделяем У от Х

# In[22]:


separate = lambda x: (x['diagnosis'], x.drop("diagnosis",axis=1))

y_train, X_train = separate(train)
y_validate, X_validate = separate(validate)
y_test, X_test = separate(test)

y_full, X_full = separate(df)


# ##### Создаем, модель NN. Подбираем для нее параметры

# In[23]:


rfc = RandomForestClassifier(random_state=0)


parameters = {
    'n_estimators': list(range(100,501,100)),
    'max_depth': list(range(2,10)),
    
}


cvgrid = GridSearchCV(estimator=rfc, param_grid=parameters,n_jobs=-1,verbose=3) 


# In[24]:


cvgrid.fit(X_train,y_train)


# ## Выбираем модель с лучшим показателем

# In[25]:


cvgrid.best_estimator_


# ## Посмотрим процент "точных" предсказаний модели для разного кол-ва эпох

# In[26]:


rfc_model = RandomForestClassifier(max_depth=2, n_estimators=200, random_state=0)
np.random.seed(0)

N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS = 3 # 12 # 4
N_BATCH = 128
N_CLASSES = np.unique(y_train)

scores_train = []
scores_test = []

# EPOCH
epoch = 0
while epoch < N_EPOCHS:
    print('epoch: ', epoch+1)
    # SHUFFLING
    random_perm = np.random.permutation(X_train.shape[0])
    mini_batch_index = 0
    while True:
        # MINI-BATCH
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        rfc_model.fit(X_train.iloc[indices], y_train.iloc[indices])
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    # SCORE TRAIN
    scores_train.append(rfc_model.score(X_train, y_train))

    # SCORE TEST
    scores_test.append(rfc_model.score(X_test, y_test))

    epoch += 1
    pass

print('Epochs:', epoch)
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_train)
ax[0].set_title('Train')
ax[1].plot(scores_test)
ax[1].set_title('Test')
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()

m=max(scores_test)
print('N:',scores_test.index(m)+1, m)


# In[27]:


plt.title("Train")
sns.barplot(x=[i+1 for i in range(len(scores_train))],y=scores_train)


# In[28]:


plt.title("Test")
sns.barplot(x=[i+1 for i in range(len(scores_test))],y=scores_test)


# ## Рассмотрим структурные единицы древа
# 

# In[29]:


from sklearn import tree
plt.figure(figsize=(20,10))
for i in range(2):
    plt.subplot(2, 1, i+1)  #subplot function: the number of rows are given as 5 and number of columns as 2, the value i+1 gives the subplot number, subplot numbers start with 1
    tree.plot_tree(cvgrid.best_estimator_[i]) 
plt.tight_layout()
plt.show()


# <hr>
# <h2>Посмотрим на точность и Среднюю Абсолютную Ошибку</h2>

# In[30]:


predicted = rfc_model.predict(X_validate)
aim=list(y_validate)
c=0
for i in range(len(aim)):
    if aim[i]==predicted[i]:
        c+=1
print(f"Accuracy:\t{round(100*c/len(aim), 2)}%\nMAE:\t\t{round(100*mean_absolute_error(y_validate, predicted), 2)}%")


# <hr>
# <h2>Экспортируем обученную модель с помощью модуля pickle</h2>

# In[3]:


model_fin = rfc_model # final model
with open("model.pckl", 'wb') as file:
    file.write(pickle.dumps(model_fin))
    
validate.to_csv('validate.csv')


# In[ ]:




