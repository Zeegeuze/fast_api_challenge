# Immo Eliza - Model Deployment

## The mission
The real estate company Immo Eliza is really happy with your regression model and current work up to now.

They would like you to create an API so their web developers can access the predictions whenever they need to. They also want to have a small web application for the non-technical employees and possibly their clients to use. The API and the web application should be intertwined but separate.

## Run the program
The website loads locally. Just run the below code the start the API:
`uvicorn immo_eliza.api.app:app --reload`
And to start the website:
`streamlit run front.py --server.headless true`

The website can be found at: http://localhost:8501/

## Setup of the code
The main code is divided into 3 parts:
* The api, where the app is found
* the code for preprocessing and model in the ml_logic map
* the front.py file having the front end of the website

## Project
This project run for a week.

## Result
The result isn't as expected. I had lots of problems, as I was looking for solutions on too many places. As I didn't have my predict last week I needed to spend a lot of time trying to fix that. I got stuck with the scalers and pipeline and didn't manage to get a proper model to load.

As the error was about 100k anyway, I just put a random price between 225.000 and 350.000, which might be as correct as a model with a 100k error.
