from sklearn.model_selection import train_test_split
import pandas as pd

def feature_engineer(data):
   
   categorical=['vehicle_type', 'registration_year', 'gearbox',
      'model', 'registration_month', 'fuel_type', 'brand',
      'not_repaired']
   seed=12345
   features=data.drop(['price'],axis=1)
   target=data['price']
   features_oh=pd.get_dummies(features[categorical],drop_first=True)
   for i in categorical:
      features[i]=features[i].astype('category')
    
   return features,features_oh,target