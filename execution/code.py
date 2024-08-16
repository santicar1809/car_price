# %% [markdown]
# # Descripción del proyecto
# 
# Rusty Bargain es un servicio de venta de coches de segunda mano que está desarrollando una app para atraer a nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial, especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# 
# A Rusty Bargain le interesa:
# 
# - la calidad de la predicción
# - la velocidad de la predicción
# - el tiempo requerido para el entrenamiento
# 
# ## Instrucciones del proyecto
# 
# 1. Descarga y examina los datos.
# 2. Entrena diferentes modelos con varios hiperparámetros (debes hacer al menos dos modelos diferentes, pero más es mejor. 3. Recuerda, varias implementaciones de potenciación del gradiente no cuentan como modelos diferentes). El punto principal de este paso es comparar métodos de potenciación del gradiente con bosque aleatorio, árbol de decisión y regresión lineal.
# 4. Analiza la velocidad y la calidad de los modelos.
# 
# ## Observaciones:
# 
# - Utiliza la métrica RECM para evaluar los modelos.
# - La regresión lineal no es muy buena para el ajuste de hiperparámetros, pero es perfecta para hacer una prueba de cordura de otros métodos. Si la potenciación del gradiente funciona peor que la regresión lineal, definitivamente algo salió mal.
# - Aprende por tu propia cuenta sobre la librería LightGBM y sus herramientas para crear modelos de potenciación del gradiente (gradient boosting).
# - Idealmente, tu proyecto debe tener regresión lineal para una prueba de cordura, un algoritmo basado en árbol con ajuste de hiperparámetros (preferiblemente, bosque aleatorio), LightGBM con ajuste de hiperparámetros (prueba un par de conjuntos), y CatBoost y XGBoost con ajuste de hiperparámetros (opcional).
# - Toma nota de la codificación de características categóricas para algoritmos simples. LightGBM y CatBoost tienen su implementación, pero XGBoost requiere OHE.
# - Puedes usar un comando especial para encontrar el tiempo de ejecución del código de celda en Jupyter Notebook. Encuentra ese comando.
# - Dado que el entrenamiento de un modelo de potenciación del gradiente puede llevar mucho tiempo, cambia solo algunos parámetros del modelo.
# - Si Jupyter Notebook deja de funcionar, elimina las variables excesivas por medio del operador del:
#   
# ## Descripción de los datos
# 
# ### Características
# 
# - DateCrawled — fecha en la que se descargó el perfil de la base de datos
# - VehicleType — tipo de carrocería del vehículo
# - RegistrationYear — año de matriculación del vehículo
# - Gearbox — tipo de caja de cambios
# - Power — potencia (CV)
# - Model — modelo del vehículo
# - Mileage — kilometraje (medido en km de acuerdo con las especificidades regionales del conjunto de datos)
# - RegistrationMonth — mes de matriculación del vehículo
# - FuelType — tipo de combustible
# - Brand — marca del vehículo
# - NotRepaired — vehículo con o sin reparación
# - DateCreated — fecha de creación del perfil
# - NumberOfPictures — número de fotos del vehículo
# - PostalCode — código postal del propietario del perfil (usuario)
# - LastSeen — fecha de la última vez que el usuario estuvo activo
# 
# ### Objetivo
# 
# Price — precio (en euros)
#  
# ## Evaluación del proyecto
# 
# Hemos definido los criterios de evaluación para el proyecto. Léelos con atención antes de pasar al ejercicio.
# 
# Esto es en lo que se fijarán los revisores al examinar tu proyecto:
# 
# - ¿Seguiste todos los pasos de las instrucciones?
# - ¿Cómo preparaste los datos?
# - ¿Qué modelos e hiperparámetros consideraste?
# - ¿Conseguiste evitar la duplicación del código?
# - ¿Cuáles son tus hallazgos?
# - ¿Mantuviste la estructura del proyecto?
# - ¿Mantuviste el código ordenado?
# - Ya tienes tus hojas informativas y los resúmenes de los capítulos, por lo que todo está listo para continuar con el proyecto

# %% [markdown]
# # Librerias

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn.linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from IPython.display import display
from scipy.spatial import distance
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,mean_squared_error
)
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor
import xgboost as xgb
import joblib

# %% [markdown]
# # Cargue y limpieza de datos

# %%
df=pd.read_csv('./files/datasets/input/car_data.csv')
df.head()

# %%
df.info()

# %%
#df=df.sample(1000,random_state=12345)

# %%
#Funcion para pasar columnas al formato snake_case
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# %%
#Pasamos las columnas al modo snake_case
columns=df.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df.columns=new_cols
print(df.columns)

# %%
#Pasamos datos categoricos a texto
df['postal_code']=df['postal_code'].astype('str')


# %%
df.drop(['date_crawled','date_created','number_of_pictures','last_seen','postal_code'],axis=1,inplace=True)

# %%
df.describe()

# %% [markdown]
# Desde el $ df.info $ podemos ver bastantes diferencias entre columnas lo que puede significar en valores ausentes, lo cual ya veremos más a fondo.

# %% [markdown]
# ## Ausentes

# %%
#Calculamos el número de ausentes
print('Porcentaje de significancia:\n',df.isna().sum())

# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Porcentaje de significancia: \n',df.isna().sum()/df.shape[0])

# %%
def nan_values(data):
    # Tratamiento de ausentes
    null_cols=['vehicle_type','gearbox','model','fuel_type','not_repaired']
    for column in null_cols:   
        if data[column].isna().sum()/data.shape[0] < 0.15:
            mode=data[column].mode()[0]
            data[column].fillna(value=mode,inplace=True)
        elif data[column].isna().sum()/data.shape[0] > 0.15:
            data.dropna(inplace=True)
        return data

# %%
df=nan_values(df)

# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Porcentaje de significancia: \n',100*df.isna().sum()/df.shape[0])

# %% [markdown]
# Vemos una gran cantidad de valores ausentes para la columna 'not_repaired' por lo tanto los vamos a eliminar, y el resto de columnas los imputaremos con el valor de la moda.

# %%
null_cols=['vehicle_type','gearbox','model','fuel_type']
for col in null_cols:
    mode=df[col].mode()[0]
    df[col].fillna(value=mode,inplace=True)
    print(df[col].isna().sum())


# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Porcentaje de significancia: \n',100*df.isna().sum()/df.shape[0])

# %%
df.dropna(inplace=True)

# %% [markdown]
# Aplicamos nuestras técnicas para eliminar los valores ausentes.

# %%
numeric=['registration_year','registration_month','price']
for column in numeric:
    plt.boxplot(df[column])
    plt.title(column)   
    plt.show()

# %% [markdown]
# Podemos ver varios datos atipicos en el año de registro, debido a que hay años mayores a la fecha actual, debemos reemplazarlos. De la misma manera debemos tener en cuenta los precios de los carros que son iguales a cero, debido a que no tiene sentido, los reemplazaremos con la media.

# %%

df['registration_year'][df['registration_year']==0]

# %%
df['registration_year'][df['registration_year']>2024]=0

# %%
df['registration_year']=df['registration_year'].replace(0,df['registration_year'].mean())
df['price']=df['price'].replace(0,df['price'].mean())

# %%
df['registration_year']=df['registration_year'].astype('str')
df['registration_month']=df['registration_month'].astype('str')
df['price']=df['price'].astype('int')


# %%
numeric=['registration_year','registration_month','price']
for column in numeric:
   print(df[column].value_counts())

# %% [markdown]
# ## Duplicados

# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Total duplicados: \n',df.duplicated().sum())

# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Total duplicados: \n',100*df.duplicated().sum()/df.shape[0])

# %% [markdown]
# Al tener un porcentaje de datos duplicados tan bajo, optamos por eliminarlos

# %%
df.drop_duplicates(inplace=True)

# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Total duplicados: \n',100*df.duplicated().sum()/df.shape[0])

# %% [markdown]
# # Entrenamiento

# %% [markdown]
# ## Separamos los datos de entrenamiento y validación

# %%
categorical=['vehicle_type', 'registration_year', 'gearbox',
       'model', 'registration_month', 'fuel_type', 'brand',
       'not_repaired']
seed=12345
features=df.drop(['price'],axis=1)
target=df['price']
features_oh=pd.get_dummies(features[categorical],drop_first=True)
features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.25,random_state=seed)
features_train_oh,features_valid_oh,target_train_oh,target_valid_oh=train_test_split(features_oh,target,test_size=0.25,random_state=seed)

# %% [markdown]
# ## Modelos con potenciación del gradiente

# %% [markdown]
# ### Catboost Regressor

# %%
features_train.info()
features_train.columns

# %%
grid = {'iterations': [50],
    'learning_rate': [0.03, 0.1],
    'depth':[4, 6, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# %%


#Entrenamos el modelo
features_total=['vehicle_type', 'registration_year', 'gearbox', 'power', 'model',
       'mileage', 'registration_month', 'fuel_type', 'brand', 'not_repaired']
features_cat=['vehicle_type', 'registration_year', 'gearbox', 'model', 'registration_month', 'fuel_type', 'brand', 'not_repaired']
train_pool = Pool(features_train[features_total], label=target_train, cat_features=features_cat)
valid_pool = Pool(features_valid[features_total], label=target_valid, cat_features=features_cat)
model_cat=CatBoostRegressor(loss_function='RMSE',random_seed=seed)
grid_search_cat=model_cat.grid_search(grid, train_pool, shuffle=False,verbose=3)
best_params=grid_search_cat['params']
print(best_params)

# %%
#Evaluamos el modelo

best_model = CatBoostRegressor(**best_params)
best_model.fit(train_pool, eval_set=valid_pool, verbose=10)
predictions = best_model.predict(valid_pool)
r2_rmse = r2_score(target_valid, predictions)
rmse_score_rmse_model = np.sqrt(mean_squared_error(target_valid, predictions))
print('R2 score: {:.3f}\nRMSE score: {:.2f}'.format(r2_rmse, rmse_score_rmse_model))

# %%
joblib.dump(best_model,'models/model_cat.joblib')

# %%
categorical_features = ['vehicle_type', 'registration_year', 'gearbox', 'model', 'registration_month', 'fuel_type', 'brand', 'not_repaired']
for i in categorical_features:
    features_train[categorical_features]=features_train[categorical_features].astype('category')
    features_valid[categorical_features]=features_valid[categorical_features].astype('category')

# %%
features_train.info()

# %% [markdown]
# ## Light gbm sin optimizar hiperparametros

# %%
params = {
    'objective': 'regression',
    'metric': 'rmse'
    }

# %%


categorical_features = ['vehicle_type', 'registration_year', 'gearbox', 'model', 'registration_month', 'fuel_type', 'brand', 'not_repaired']
lgb_train_features=lgb.Dataset(features_train,label=target_train,categorical_feature=categorical_features)
lgb_valid_features=lgb.Dataset(features_valid,label=target_valid,reference=lgb_train_features)
#Entrenamos el modelo
model_lgbm=lgb.train(params,lgb_train_features,valid_sets=[lgb_train_features, lgb_valid_features])

prediction = model_lgbm.predict(features_valid)
# Evaluamos el modelo
r2_rmse = r2_score(target_valid, prediction)
rmse_score_rmse_model = np.sqrt(mean_squared_error(target_valid, prediction))
print('R2 score: {:.3f}\nRMSE score: {:.2f}'.format(r2_rmse, rmse_score_rmse_model))

# %%
joblib.dump(model_lgbm,'models/model_lgbm.joblib')

# %% [markdown]
# ## Light gbm optimizando hiperparametros

# %%


#Entrenamos el modelo
model_lg = lgb.LGBMRegressor(random_state=seed,force_col_wise=True)


param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [20, 30, 50]
}


grid = GridSearchCV(model_lg, param_grid, cv=3, scoring='r2')

grid.fit(features_train_oh, target_train_oh)

print("Mejores parametros:", grid.best_params_)
print("Mejores parametros: {:.2f}".format(grid.best_score_))
#Evaluamos el modelo
best_model = grid.best_estimator_
y_pred = best_model.predict(features_valid_oh)
rmse = mean_squared_error(target_valid, y_pred)**0.5
r2=r2_score(target_valid, y_pred)
print("RMSE:",rmse)
print('r2: ',r2)

# %%
joblib.dump(best_model,'models/model_lg.joblib')

# %% [markdown]
# ## XGBoost

# %%

#Entrenamos los datos
dtrain=xgb.DMatrix(features_train_oh,label=target_train_oh)
dtest=xgb.DMatrix(features_valid_oh,label=target_valid_oh)
param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
param['nthread'] = 4
param['eval_metric'] = 'rmse'
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

#Evaluamos el modelo
y_pred = bst.predict(dtest)
xg_rmse=mean_squared_error(target_valid_oh,y_pred)**0.5
xg_r2=r2_score(target_valid_oh,y_pred)
print("RMSE:",xg_rmse)
print('r2: ',xg_r2)

# %%
joblib.dump(bst,'models/bst.joblib')

# %% [markdown]
# ## Creamos pipelines de los modelos que vamos a evaluar

# %%
#Escalamos nuestros datos
numeric=['power','mileage']
scaler=StandardScaler()
scaler.fit(features_train[numeric])
features_train_oh[numeric]=scaler.transform(features_train[numeric])
features_valid_oh[numeric]=scaler.transform(features_valid[numeric])


# %%
# Comenzamos por los modelos sin potenciación del gradiente
pipe_rf=Pipeline([('rf',RandomForestRegressor())])
pipe_dt=Pipeline([('dt',DecisionTreeRegressor())])

params=[{'rf__max_depth':[3,4,2,1],
         'rf__min_samples_split':[5,10,11,12],
         'rf__n_estimators':[10,20,30,40],
         'rf__min_samples_leaf':[1, 2, 4],
         'rf__bootstrap':[True, False],
         'rf__max_features':[np.random.randint(1, 11)]},
         {'dt__max_depth': [3,4,2,1],
         'dt__max_features':[np.random.randint(1, 9)],
         'dt__min_samples_leaf': [1, 2, 4]},
         ]

# %%

#Entrenamos nuestros modelos

pipes=[pipe_rf,pipe_dt]
for pipe,grid in zip(pipes,params):
    rs=RandomizedSearchCV(estimator=pipe,param_distributions=grid,scoring='r2',cv=2,random_state=seed)
    rs.fit(features_train_oh,target_train_oh)
    print(rs.best_params_)
    print(rs.best_score_)
    print(rs.best_estimator_)
    random_prediction = rs.best_estimator_.predict(features_valid_oh)
    random_rmse=mean_squared_error(target_valid_oh,random_prediction)**0.5
    random_r2=r2_score(target_valid,random_prediction)
    print("RMSE:",random_rmse)
    print('r2: ',random_r2)
    

# %% [markdown]
# # Modelamieto sin pipelines

# %% [markdown]
# ## Random Forest

# %%
params_rf={'max_depth':[3,4,2,1],
         'min_samples_split':[5,10,11,12],
         'n_estimators':[10,20,30,40],
         'min_samples_leaf':[1, 2, 4],
         'bootstrap':[True, False],
         'max_features':[np.random.randint(1, 11)]}

# %%


#Entrenamos el modelo
model_rf=RandomForestRegressor()
rs=RandomizedSearchCV(estimator=model_rf,param_distributions=params_rf,scoring='r2',cv=2,random_state=seed)
rs.fit(features_train_oh,target_train_oh)
print(rs.best_params_)
print(rs.best_score_)
print(rs.best_estimator_)


# %%
#Evaluamos el modelo
best_random = rs.best_estimator_
random_prediction = best_random.predict(features_valid_oh)
random_rmse=mean_squared_error(target_valid_oh,random_prediction)**0.5
random_r2=r2_score(target_valid,random_prediction)
print("RMSE:",random_rmse)
print('r2: ',random_r2)

# %%
joblib.dump(best_random,'models/rs.joblib')

# %% [markdown]
# ## Desicion Tree

# %%
params_dt={'max_depth': [3,4,2,1],
         'max_features':[np.random.randint(1, 9)],
         'min_samples_leaf': [1, 2, 4]}

# %%


#Entrenamos el modelo
model_dt=DecisionTreeRegressor()
dt=RandomizedSearchCV(estimator=model_dt,param_distributions=params_dt,scoring='r2',cv=2,random_state=seed)
dt.fit(features_train_oh,target_train_oh)
print(dt.best_params_)
print(dt.best_score_)
print(dt.best_estimator_)


# %%
#Evaluamos el modelo
best_random_dt = dt.best_estimator_
random_prediction_dt = best_random_dt.predict(features_valid_oh)
random_rmse=mean_squared_error(target_valid_oh,random_prediction_dt)**0.5
random_r2=r2_score(target_valid_oh,random_prediction_dt)
print("RMSE:",random_rmse)
print('r2: ',random_r2)

# %%
joblib.dump(best_random_dt,'models/dt.joblib')

# %% [markdown]
# ## Linear Regression (Dummie Test)

# %%


#Entrenamos el modelo
model_lr=LinearRegression()
model_lr.fit(features_train_oh,target_train_oh)
#Evaluamos el modelo
linear_prediction=model_lr.predict(features_valid_oh)
random_rmse=mean_squared_error(target_valid_oh,linear_prediction)**0.5
random_r2=r2_score(target_valid_oh,linear_prediction)
print("RMSE:",random_rmse)
print('r2: ',random_r2)


# %%
joblib.dump(model_lr,'models/model_lf.joblib')

# %% [markdown]
# ## Tabla de resultados

# %%
datos={'Modelo':['Cat_boost','Light_gbm','Light_gbm_hyperparameters','XGBoost','RandomForestRegressor','DecisionTreeRegressor','LinearRegression'],
       'RMSE':['1825.17','1718.21','2270.28','2854.74','4230.52','4569.04','2329187786074.899'],
       'R2':['0.842','0.860','0.755','0.613','0.151','0.010','-2.571e+17'],
       'Tiempo':['2min 25s','861 ms','21.9 s','2.62 s','9.04 s','7.22 s','7.99 s']}
results=pd.DataFrame(datos)
results.head(7)

# %% [markdown]
# En la anterior tabla tenemos la muestra de los resultados de calidad de los modelos y de tiempo. Podemos ver que el modelo que tuvo la mejor calidad fue el Light_gbm con 1718.21 de error en el precio, seguido por el Cat_boost con 1825.17 de error en el precio y el Light_gbm con hyperparametros que obtuvo un error de 2270.28. Los mismos modelos obtuvieron un r2 de 0.860, 0.842 y 0.755. El modelo que tuvo un tiempo de ejecución menor fue el Light_gbm con tiempo de 918 microsegundos, seguido del XGBoost con 2.76 segundos y la regresion lineal con 7.75 segundos.
# 
# En cuanto a la valoración general, podemos clasificar a Light_gbm, como el mejor modelo debido a que tiene el menor error y el menor tiempo de ejecución, superando al CatBoost en calidad y tiempo, debido a que este fue el modelo con mayor tiempo de ejecución.
# 
# Respecto a la comparación de los modelos con potenciación del gradiente y los que no tienen potenciación, claramente vemos una diferencia grande, debido a que el modelo que menor error tuvo que no presenta potenciación del gradiente es el RandomForest que tuvo un error de 4230.53 y un r2 de 0.151.
# 
# Por último comparando nuestros modelos con la regresión lineal que tuvo el peor resultado y era nuestra prueba de cordura, podemos decir que nuestros modelos fueron superiores a lo esperado.
#  

# %% [markdown]
# # Conclusiones
# 
# 1. El mejor modelo de predicción del precio de los carros fue el **Light_gbm con 1718.21 de error, 0.86 de r2 y tiempo de 918 ms**.
# 
# 2. Los modelos con potenciación del gradiente son superiores a los que no lo tienen y mejoran la calidad del modelo en gran medida.
# 
# 3. El tiempo de ejecución de los modelos con potenciación del gradiente fueron menores a los que no la tienen a excepción del **CatBoost** que es uno de los algoritmos más poderosos en cuanto a calidad, sin embargo, tuvo el tiempo de ejecución mayor con **2 minutos y 24 segundos**.
# 
# 4. Podemos tener modelos muy buenos en calidad, pero si este demora mucho tiempo, deja de ser un modelo valioso.


