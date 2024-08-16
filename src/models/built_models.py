import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from src.models.hyper_parameters import all_models,category_models
import joblib

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = all_models() 
    
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    results = []

    # Iterating the models
    models_name = ['lr','xg','rf','dt']
    for model,i in zip(models,models_name):
        best_estimator, best_score, rmse_val,r2_val= model_structure(data, model[1], model[2])
        results.append([model[0],best_estimator,best_score, rmse_val,r2_val])      
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_{i}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','rmse_score','r2_score'])
    
    results_category = []
    cat_features=['vehicle_type', 'registration_year', 'gearbox', 'model', 'registration_month', 'fuel_type', 'brand', 'not_repaired']
    models_category=category_models(cat_features)
    # Iterating the models
    models_name = ['cat','lgbm']
    for model,i in zip(models_category,models_name):
        best_estimator, best_score, rmse_val,r2_val= model_category(data, model[1], model[2])
        results_category.append([model[0],best_estimator,best_score, rmse_val,r2_val])      
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_{i}.joblib')
    results_category = pd.DataFrame(results_category, columns=['model','best_estimator','best_train_score','rmse_score','r2_score'])
    
    final_rev = pd.concat([results_df,results_category])
    final_rev.to_csv('./files/modeling_output/reports/model_report.csv',index=False)

    return final_rev[['model','rmse_score','r2_score']]


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    features=data[1]
    target=data[2]
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.25,random_state=seed)   
    # Training the model
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    rmse_val,r2_val = eval_model(best_estimator,features_valid,target_valid)
    print(f'RMSE: {rmse_val}')
    
    results = best_estimator, best_score,rmse_val,r2_val
    return results
    
def eval_model(best,features_valid,target_valid):
    random_prediction = best.predict(features_valid)
    random_rmse=mean_squared_error(target_valid,random_prediction)**0.5
    r2=r2_score(target_valid,random_prediction)
    print("RMSE:",random_rmse)
    print("R2:",r2)
    return random_rmse,r2

def model_category(data, pipeline, param_grid):
    
    seed=12345
    
    features=data[0]
    target=data[2]
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.25,random_state=seed)
    
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    rmse_val,r2_val = eval_model(best_estimator,features_valid,target_valid)
    
    print(f'RMSE: {rmse_val}')
    
    results = best_estimator, best_score,rmse_val,r2_val
    return results
    