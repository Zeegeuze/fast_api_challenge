import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler

from immo_eliza.ml_logic.encoders import * # transform_time_features, transform_lonlat_features, compute_geohash

from colorama import Fore, Style


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    X = compress(X)
    X = convert_types(X)
    X = X.drop_duplicates()
    X = drop_cols_and_fill_nas(X)

    X = encoding_df(X)

    # return X
    def create_sklearn_preprocessor(X) -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
        """

    #     # PASSENGER PIPE
    #     p_min = 1
    #     p_max = 8
    #     passenger_pipe = FunctionTransformer(lambda p: (p - p_min) / (p_max - p_min))

    #     # DISTANCE PIPE
    #     dist_min = 0
    #     dist_max = 100

    #     distance_pipe = make_pipeline(
    #         FunctionTransformer(transform_lonlat_features),
    #         FunctionTransformer(lambda dist: (dist - dist_min) / (dist_max - dist_min))
    #     )

    #     # TIME PIPE
    #     timedelta_min = 0
    #     timedelta_max = 2090

    #     time_categories = [
    #         np.arange(0, 7, 1),  # days of the week
    #         np.arange(1, 13, 1)  # months of the year
    #     ]

    #     time_pipe = make_pipeline(
    #         FunctionTransformer(transform_time_features),
    #         make_column_transformer(
    #             (OneHotEncoder(
    #                 categories=time_categories,
    #                 sparse_output=False,
    #                 handle_unknown="ignore"
    #             ), [2,3]), # corresponds to columns ["day of week", "month"], not the other columns

    #             (FunctionTransformer(lambda year: (year - timedelta_min) / (timedelta_max - timedelta_min)), [4]), # min-max scale the columns 4 ["timedelta"]
    #             remainder="passthrough" # keep hour_sin and hour_cos
    #         )
    #     )

    #     # GEOHASH PIPE
    #     lonlat_features = [
    #         "pickup_latitude", "pickup_longitude", "dropoff_latitude",
    #         "dropoff_longitude"
    #     ]

    #     # Below are the 20 most frequent district geohashes of precision 5,
    #     # covering about 99% of all dropoff/pickup locations,
    #     # according to prior analysis in a separate notebook
    #     most_important_geohash_districts = [
    #         "dr5ru", "dr5rs", "dr5rv", "dr72h", "dr72j", "dr5re", "dr5rk",
    #         "dr5rz", "dr5ry", "dr5rt", "dr5rg", "dr5x1", "dr5x0", "dr72m",
    #         "dr5rm", "dr5rx", "dr5x2", "dr5rw", "dr5rh", "dr5x8"
    #     ]

    #     geohash_categories = [
    #         most_important_geohash_districts,  # pickup district list
    #         most_important_geohash_districts  # dropoff district list
    #     ]

    #     geohash_pipe = make_pipeline(
    #         FunctionTransformer(compute_geohash),
    #         OneHotEncoder(
    #             categories=geohash_categories,
    #             handle_unknown="ignore",
    #             sparse_output=False
    #         )
    #     )

    #     # COMBINED PREPROCESSOR
    #     final_preprocessor = ColumnTransformer(
    #         [
    #             ("passenger_scaler", passenger_pipe, ["passenger_count"]),
    #             ("time_preproc", time_pipe, ["pickup_datetime"]),
    #             ("dist_preproc", distance_pipe, lonlat_features),
    #             ("geohash", geohash_pipe, lonlat_features),
    #         ],
    #         n_jobs=-1,
    #     )

    #     return final_preprocessor

    # print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    # preprocessor = create_sklearn_preprocessor(X)
    # X_processed = preprocessor.fit_transform(X)

    # print("✅ X_processed, with shape", X_processed.shape)
    X = X.drop(0, axis='columns')

    print(X.dtypes)
    return X

def compress(df, **kwargs):
    """
    Reduces the size of the DataFrame by downcasting numerical columns
    """
    input_size = df.memory_usage(index=True).sum()/ 1024**2
    print("old dataframe size: ", round(input_size,2), 'MB')

    in_size = df.memory_usage(index=True).sum()

    for t in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=t))

        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=t)

    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio,2)))
    print("new DataFrame size: ", round(out_size / 1024**2,2), " MB")

    return df

def convert_types(df):
    '''
    Takes a df, convert all int8 columns into int16 type and returns that df
    '''
    df.fl_furnished = df.fl_furnished.astype("int16")
    df.fl_open_fire = df.fl_open_fire.astype("int16")
    df.fl_terrace  = df.fl_terrace .astype("int16")
    df.fl_garden = df.fl_garden.astype("int16")
    df.fl_swimming_pool = df.fl_swimming_pool.astype("int16")
    df.fl_floodzone = df.fl_floodzone.astype("int16")
    df.fl_double_glazing  = df.fl_double_glazing .astype("int16")

    return df

def drop_cols_and_fill_nas(df):
    '''
    Takes a df, will fill N/As or delete the column and return a df
    '''

    df = df.drop('id', axis='columns')
    df = df.drop('epc', axis='columns')
    df = df.drop('cadastral_income', axis='columns')
    df['surface_land_sqm'].fillna(df['surface_land_sqm'].mean(), inplace=True)
    df['construction_year'].fillna(2002, inplace=True)
    df = df.drop('primary_energy_consumption_sqm', axis='columns')
    df = df.drop('nbr_frontages', axis='columns')
    df = df.drop('latitude', axis='columns')
    df = df.drop('longitude', axis='columns')
    df['terrace_sqm'].fillna(df['terrace_sqm'].mean(), inplace=True)
    df['total_area_sqm'].fillna(df['total_area_sqm'].mean(), inplace=True)
    df['garden_sqm'].fillna(df['garden_sqm'].mean(), inplace=True)

    return df

def encoding_df(X):
    '''
    Takes and returns a df (X), will do all necessary encoding
    '''
    print('Start encoding')
    # One hot encoding
    cols = ['property_type', 'subproperty_type', 'region', 'province', 'heating_type']
    X = one_hot(X, cols)

    print("one hot done")
    print(X.columns)

    # # Add missing columns for training
    # if "region_MISSING" not in X.columns:
    #     X.insert(44, 'region_MISSING', 0)

    #     if "province_MISSING" not in X.columns:
    #         X.insert(54, 'province_MISSING', 0)
    # else:
    #     if "province_MISSING" not in X.columns:
    #         X.insert(54, 'province_MISSING', 0)

    # Ordinal encoding: State building
    categories = ["AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP", "TO_RENOVATE", "TO_RESTORE", "MISSING"]
    X = ordinal(X, categories, "state_building")

    print("categories done")
    print(X.columns)

    # Ordinal encoding: kitchen
    categories = ["USA_HYPER_EQUIPPED", "HYPER_EQUIPPED", "USA_SEMI_EQUIPPED", "SEMI_EQUIPPED", "USA_INSTALLED", "INSTALLED", "USA_UNINSTALLED", "NOT_INSTALLED", "MISSING"]
    X = ordinal(X, categories, "equipped_kitchen")

    # Label encoding zip_code and locality
    label_encoder = LabelEncoder()

    # Encode labels in column 'species'.
    X['zip_code']= label_encoder.fit_transform(X['zip_code'])
    X['locality']= label_encoder.fit_transform(X['locality'])

    return X

def one_hot(X, cols):
    '''
    Takes and returns X after one_hot encoding. Also takes a list with features to encode.
    '''
    print(X.columns)
    print("----------------")
    print(cols)
    OH_encoder = OneHotEncoder()
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[cols]))
    print(OH_encoder.get_feature_names_out())
    print(OH_cols.columns)
    print(X.index)
    OH_cols.index = X.index
    # OH_cols.columns = OH_encoder.get_feature_names_out()
    X = X.drop(cols, axis=1)
    X = pd.concat([X, OH_cols], axis=1)

    return X

def ordinal(X, categories, feature):
    '''
    Takes X, a feature and a list of categories of that feature and ordinal encodes them. Returns X
    '''
    # Instantiate the Ordinal Encoder
    ordinal_encoder = OrdinalEncoder(categories = [categories])

    # Fit it
    ordinal_encoder.fit(X[[feature]])

    # Transforming categories into ordered numbers
    X[feature] = ordinal_encoder.transform(X[[feature]])

    return X
