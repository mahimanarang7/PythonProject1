Briefing about the Python Project:
(MBD-S2 : Group 7)

DATA:
The dataset taken is a timeseries dataset on the electricity prices from Kaggle : https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather?resource=download
The dataset contains hourly data of 4 years from 2015-2018.
The dataset from csv was stored in a dataframe and saved into an SQL database (energy.db) using SQLLite(SQLAlchemy) in pycharm(IDE).

DEPENDENCIES:
To install packages and their related dependencies, poetry was added (with .toml file) and going furhter all packages were added using poetry.

PRE-PROCESSING:
The data has been cleaned by removing the features which did not have significant information.
For imputing the features having missing values, interpolate method has been used as we cannot remove rows directly considerign time-series.
Some new features we created based on the time- such as week of the year and month of the year in cyclical way.

GIT:
With the basic code in beginning, git was created for version controls at IDE and remote repository on the Github to sync across.

MODELLING:
To train the dataset, timeSeriesSplit was used based on the date instead of train_test_split.
RandomForestRegressor with cross-validation on 5 splits was performed and the resultign model was saved in model.pkl file.

Using the saved model, predictions were performed for the prices in test dataset and saved into a new table in the database.

Then the predictions for the price were plotted corresponding to the actual prices against the time column.
For clear view, the prices were averaged by month due to huge dataset.

TYPER COMMANDS:
The above 3 fucntions were performed through Typer (app) command for CLI interaction:
python model_and_evaluation.py <typer_command>
  train-model-and-save
  make-predictions
  plot-predictions

STREAMLIT:
New file streamlit.py contains the functions to show the information including plots and graphs.
Command: streamlit run streamlit.py
The website contains two pages:
First about the exploratory data analysis
Second is about the training, making predictions and to see the plots of the predictions.

--Thank You--





