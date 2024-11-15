import streamlit as st
import requests
import pandas as pd
import folium

from streamlit_folium import st_folium


st.markdown("""# Immo Eliza
## Predict your house price
Fill in all details below""")

id = 0

columns = st.columns(2)
property_type = columns[0].selectbox('Property Type', ['HOUSE', 'APARTMENT'])
subproperty_type_feat = ['HOUSE', 'APARTMENT', 'DUPLEX', 'VILLA', 'EXCEPTIONAL_PROPERTY',
       'FLAT_STUDIO', 'GROUND_FLOOR', 'PENTHOUSE', 'FARMHOUSE',
       'APARTMENT_BLOCK', 'COUNTRY_COTTAGE', 'TOWN_HOUSE', 'SERVICE_FLAT',
       'MANSION', 'MIXED_USE_BUILDING', 'MANOR_HOUSE', 'LOFT', 'BUNGALOW',
       'KOT', 'CASTLE', 'CHALET', 'OTHER_PROPERTY', 'TRIPLEX']
subproperty_type = columns[1].selectbox('Subproperty Type', subproperty_type_feat)
region = columns[0].selectbox('Region', ['Flanders', 'Brussels-Capital', 'Wallonia'])
province_feat = ['West Flanders', 'East Flanders', 'Antwerp', 'Limburg', 'Flemish Brabant',
                 'Brussels', 'Walloon Brabant', 'Liège', 'Hainaut', 'Luxembourg', 'Namur']
province = columns[1].selectbox('Province', province_feat)
locality = columns[0].text_input('Locality')
zip_code = columns[1].text_input('Zip code', 1000)

st.text('Click on the exact location of your property')

def get_pos(lat,lng):
    return lat,lng

m = folium.Map(location=[50.5483, 4.4604], tiles='OpenStreetMap', zoom_start=7)
latitude = 50.5483
longitude = 4.4604

m.add_child(folium.LatLngPopup())

map = st_folium(m, height=350, width=700)

if map['last_clicked'] is not None:
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    clicked_latitude = map['last_clicked']['lat']
    clicked_longitude = map['last_clicked']['lng']

    columns = st.columns(2)
    latitude = columns[0].text_input('Latitude', clicked_latitude)
    longitude = columns[1].text_input('Longitude', clicked_longitude)

columns = st.columns(4)
construction_year = columns[0].text_input('Construction Year', 2024)
total_area_sqm = columns[1].text_input('Total Area sqm', 0)
surface_land_sqm = columns[2].text_input('Surface land sqm', 0)
nbr_frontages = columns[3].text_input('Number of facades', 2)
nbr_bedrooms = st.slider('Number of Bedrooms', 0, 10, 2)

columns = st.columns(2)
equipped_kitchen_feat = ['HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED',
       'SEMI_EQUIPPED', 'USA_SEMI_EQUIPPED', 'INSTALLED',
       'USA_INSTALLED', 'NOT_INSTALLED', 'USA_UNINSTALLED']
equipped_kitchen = columns[0].selectbox('Type of kitchen', equipped_kitchen_feat)
state_building_feat = ['AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_BE_DONE_UP', 'TO_RENOVATE', 'TO_RESTORE']
state_building = columns[1].selectbox('State Building', state_building_feat)
heating_type_feat = ['GAS', 'FUELOIL', 'PELLET', 'ELECTRIC', 'CARBON',
       'SOLAR', 'WOOD']
heating_type = columns[0].selectbox('Heating type', heating_type_feat)
primary_energy_consumption_sqm = columns[1].text_input(' Primary Energy Consumption sqm', 0)

columns = st.columns(4)
terrace_sqm = columns[0].text_input('Terrace sqm', 0)
garden_sqm = columns[1].text_input('Garden sqm', 0)
cadastral_income = columns[2].text_input('Cadastral Income', 0)
epc_feat = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
epc = columns[3].selectbox('epc', epc_feat)

fl_furnished = columns[0].checkbox('Furnised')
fl_open_fire = columns[1].checkbox('Open Fire')
fl_terrace = columns[2].checkbox('Terrace')
fl_garden = columns[3].checkbox('Garden')

columns = st.columns(4)
fl_swimming_pool = columns[0].checkbox('Swimming Pool')
fl_floodzone = columns[1].checkbox('Floodzone')
fl_double_glazing = columns[2].checkbox('Double Glazing')

params = {
    'id': int(id),
    'property_type': property_type,
    'subproperty_type': subproperty_type,
    'region': region,
    'province': province,
    'locality': locality,
    'zip_code': int(zip_code),
    'latitude': float(latitude),
    'longitude': float(longitude),
    'construction_year': float(construction_year),
    'total_area_sqm': float(total_area_sqm),
    'surface_land_sqm': float(surface_land_sqm),
    'nbr_frontages': float(nbr_frontages),
    'nbr_bedrooms': float(nbr_bedrooms),
    'equipped_kitchen': equipped_kitchen,
    'fl_furnished': int(fl_furnished),
    'fl_open_fire': int(fl_open_fire),
    'fl_terrace': int(fl_terrace),
    'terrace_sqm': float(terrace_sqm),
    'fl_garden': int(fl_garden),
    'garden_sqm': float(garden_sqm),
    'fl_swimming_pool': int(fl_swimming_pool),
    'fl_floodzone': int(fl_floodzone),
    'state_building': state_building,
    'primary_energy_consumption_sqm': float(primary_energy_consumption_sqm),
    'epc': epc,
    'heating_type': heating_type,
    'fl_double_glazing': int(fl_double_glazing),
    'cadastral_income': float(cadastral_income)
}


if st.button('Confirm'):
    st.write("We're looking up your house price")
    st.write('Further actions are not visible but are executed. Please wait for the calculations')

    url = 'http://127.0.0.1:8000/predict'

    response = requests.get(url, params=params)
    pred = response.json() #=> {wait: 64}
    # print(response)

    # st.success(f"Your rate will be: {round(pred['fare'], 2)}$")
    # st.success(f"Your rate will be: €{round(response, 2)}")
    st.success(pred)

else:
    st.write('Click for house price')



# data = {
#     "lat": [float(pickup_latitude), float(dropoff_latitude)],
#     "lon": [float(pickup_longitude), float(dropoff_longitude)]
# }
# df = pd.DataFrame(data)
# mean_lat = df['lat'].mean()
# mean_lon = df['lon'].mean()
# m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

# folium.Marker([float(pickup_latitude), float(pickup_longitude)], tooltip='Start').add_to(m)
# folium.Marker([float(dropoff_latitude), float(dropoff_longitude)], tooltip='Stop').add_to(m)

# dest_map = st_folium(m, width=725, height=500)
# if dest_map['last_clicked'] is not None:
#     st.write(dest_map)
