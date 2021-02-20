import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def process_df(df, model_path, mode):
    # Columna ID
    id_col = ["customerID"]

    # Variable target
    target_col = ["Churn"]

    # Variables categóricas
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod']

    # Variables numéricas
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Reemplazo de espacios por valores nulos en el campo TotalCharges
    df['TotalCharges'] = df["TotalCharges"].replace(" ", np.nan)

    # Eliminación de registros con valores nulos en el campo TotalCharges 
    df = df[df["TotalCharges"].notnull()]
    df = df.reset_index()[df.columns]

    # Conversión a tipo float
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Reemplazo de 'No internet service' por 'No' en las siguientes columnas
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']
    for i in replace_cols : 
        df[i]  = df[i].replace({'No internet service' : 'No'})
        
    # Reemplazo de valores en SeniorCitizen
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})

    if mode == "train":
        # Objeto scaler de las variables numéricas
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        with open(os.path.join(model_path, 'scaler.pkl'), 'wb') as out:
            pickle.dump(scaler, out)

        # Transformamos los valores 
        df_scaled = scaler.transform(df[num_cols])
        df_scaled = pd.DataFrame(df_scaled, columns = num_cols)

        # Variables dummy de las variables multiclase
        dummies_frame = pd.get_dummies(df[cat_cols], drop_first = True)
        dummies_frame.iloc[0:2, :].to_csv(os.path.join(model_path, "dummies.csv"), index = False)

        # Transformamos los valores
        df_dummies = df.reindex(columns = dummies_frame.columns, fill_value = 0)

        # Transformamos la variable target
        df[target_col] = df[target_col].replace({"Yes": 1, "No": 0})
       
        # Output
        X = df_dummies.merge(df_scaled, left_index = True, right_index = True)
        y = df[target_col]
                
    elif mode == "inference":
            # Cargamos el scaler para utilizarlos con nuevos datos
            with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as inp:
                scaler = pickle.load(inp)
            
            # Transformamos los valores 
            df_scaled = scaler.transform(df[num_cols])
            df_scaled = pd.DataFrame(df_scaled, columns = num_cols)

            # Cargamos las variables dummy
            dummies_frame = pd.read_csv(os.path.join(model_path, "dummies.csv"))

            # Transformamos los valores
            df_dummies = df.reindex(columns = dummies_frame.columns, fill_value = 0)

            # Output
            X = df_dummies.merge(df_scaled, left_index = True, right_index = True)
            y = None

    return X, y