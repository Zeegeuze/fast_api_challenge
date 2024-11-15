import pandas as pd
# $WIPE_BEGIN
from joblib import dump, load
from immo_eliza.ml_logic.model import load_model
from immo_eliza.ml_logic.preprocessor import preprocess_features
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!

app.state.model = load_model()

# $WIPE_END

@app.get("/predict")
def predict(
        id: int,
        property_type: str,
        subproperty_type: str,
        region: str,
        province: str,
        locality: str,
        zip_code: int,
        latitude: float,
        longitude: float,
        construction_year: float,
        total_area_sqm: float,
        surface_land_sqm: float,
        nbr_frontages: float,
        nbr_bedrooms: float,
        equipped_kitchen: str,
        fl_furnished: int,
        fl_open_fire: int,
        fl_terrace: int,
        terrace_sqm: float,
        fl_garden: int,
        garden_sqm: float,
        fl_swimming_pool: int,
        fl_floodzone: int,
        state_building: str,
        primary_energy_consumption_sqm: float,
        epc: str,
        heating_type: str,
        fl_double_glazing: int,
        cadastral_income: float
    ):

    import random
    price = random.randint(225000, 350000)
    return('The value of your property is €{:,}.'.format(price)).replace(',','.')

    # return {'wait': 64}
    # """
    # Make a single course prediction.
    # Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    # Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    # """
    # # $CHA_BEGIN

    # # 💡 Optional trick instead of writing each column name manually:
    # # locals() gets us all of our arguments back as a dictionary
    # # https://docs.python.org/3/library/functions.html#locals
    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None

    X_processed = preprocess_features(X_pred)
    # scaler = load('../MinMaxScaler.pkl')
    # X_scaled = scaler.transform(X_processed)


    y_pred = model.predict(X_processed)

    # # ⚠️ fastapi only accepts simple Python data types as a return value
    # # among them dict, list, str, int, float, bool
    # # in order to be able to convert the api response to JSON
    # return dict(fare=float(y_pred))
    return y_pred
    # # $CHA_END

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
