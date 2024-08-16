import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

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

def duplicated_values(data):
    # Tratamiento de duplicados
    if data.duplicated().sum() > 0:
            data.drop_duplicates()
    return data

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID''' 
        
    
    
    # Pasamos columnas a formato snake_case
    data = columns_transformer(data)
    data['postal_code']=data['postal_code'].astype('str')
    # Eliminamos columnas no relevantes
    data.drop(['date_crawled','date_created','number_of_pictures','last_seen','postal_code'],axis=1,inplace=True)
    
    # Ausentes
    
    data=nan_values(data)
    
    # Boxplots
    
    numeric=['registration_year','registration_month','price']
    for column in numeric:
        fig,ax=plt.subplots()
        ax.boxplot(data[column])
        ax.set_title(column)   
        fig.savefig(f'./files/modeling_output/figures/box_{column}')
    
    # Reemplazamos los precios y el año de registro con valor 0 por la media
    data['registration_year']=data['registration_year'].replace(0,data['registration_year'].mean())
    data['price']=data['price'].replace(0,data['price'].mean())    
    
    # Cambiamos los tipos de variable
    
    data['registration_year']=data['registration_year'].astype('str')
    data['registration_month']=data['registration_month'].astype('str')
    data['price']=data['price'].astype('int')
    
    data=duplicated_values(data)
    
    path = './files/datasets/intermediate/'

    data.to_csv(path+'preprocessed_data.csv', index=False)
    return data