# Disaster Response Pipeline Project

### File Description:

1. process_data.py: creates the pipeline to load, transfer and clean the data, and finally save the data into SQL database.
2. train_classifier.py: build the ML model, load the model to train and export the model
3. run.py: create the website and display 2 visualizations and include it in the project.

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


