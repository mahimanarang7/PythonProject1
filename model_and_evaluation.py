from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from matplotlib import pyplot as plt
import joblib
import typer
from typer import Option
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


# Initialize Typer app
app = typer.Typer()

# Function to load data from a database
def loading_file() -> pd.DataFrame:
    # DB connection details
    DB_URI = "sqlite:///energy.db"
    engine = create_engine(DB_URI, pool_pre_ping=True)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    # Load data from DB
    with SessionLocal() as session:
        df = pd.read_sql_table("energy_set_prep", con=engine)
    return df


df = loading_file()

def plot_time_vs_price_initial() :
    # plot the chart for time and price actual averaged by per month
    daily_meanx = df.groupby(df['time'].dt.to_period('M')).mean()
    daily_meanx.index = daily_meanx.index.to_timestamp()
    plt.plot(daily_meanx.index, daily_meanx["price actual"])
    plt.show()

#split the data into train and test by the date as this is timeseries data
train_df = df[df['time'] < '2018-07-01 00:00:00']
test_df = df[df['time'] >= '2018-07-01 00:00:00']

print(test_df.head())

# Function to train the ML model
def train_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    # train the timeseries dataframe df with random forest regressor where price actual is the target variable
    X = train_df.drop(columns=['time', 'price actual'])
    y = train_df['price actual']

    # Initialize time series split
    tscv = TimeSeriesSplit(n_splits=5)

    # Train the model using cross-validation
    model = RandomForestRegressor()
    mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=tscv)

    # Print mean squared error scores
    typer.echo("Mean Squared Error Scores:")
    for i, mse_score in enumerate(mse_scores):
        typer.echo(f"Fold {i + 1}: {mse_score}")

    # Train the model on the full dataset
    model.fit(X, y)
    return model

# Function to save the trained model to a file
def save_model(model: RandomForestRegressor, file_path: str):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def save_predictions(predictions: pd.DataFrame, database_path: str, table_name: str):

    conn = sqlite3.connect(database_path)
    predictions.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# Function to make predictions
def make_predictions(initial_date: str, n_time_units: int, model: RandomForestRegressor,
                     test_df: pd.DataFrame) -> pd.DataFrame:

    # Initialize the predictions DataFrame
    # Convert initial date to datetime object
    initial_date = datetime.strptime(initial_date, '%Y-%m-%d %H:%M:%S')

    # Generate datetime index for prediction dates
    prediction_dates = [initial_date + timedelta(hours=i) for i in range(n_time_units)]

    # Prepare DataFrame for predictions
    prediction_data = pd.DataFrame({'time': prediction_dates})

    # Merge prediction_data with test_df on 'date' column to get all features for prediction
    prediction_data = pd.merge(prediction_data, test_df, on='time', how='left')

    # Extract features for prediction
    X_pred = prediction_data.drop(columns=['time', 'price actual'])

    # Make predictions using the model
    predictions = model.predict(X_pred)

    # Add predictions to prediction_data DataFrame
    prediction_data['prediction'] = predictions
    return prediction_data


# Function to plot predictions vs. real data
def plot_predictions_vs_real(predictions_df: pd.DataFrame, actual_df: pd.DataFrame):
    # Plot the results
    # Group by date and calculate the mean price for predicted and actual prices
    predicted_daily_mean = predictions_df.groupby(predictions_df['time'].dt.to_period('M'))['prediction'].mean()
    actual_daily_mean = actual_df.groupby(actual_df['time'].dt.to_period('M'))['price actual'].mean()
    # Convert PeriodIndex to datetime
    predicted_daily_mean.index = predicted_daily_mean.index.to_timestamp()
    actual_daily_mean.index = actual_daily_mean.index.to_timestamp()
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_daily_mean.index, predicted_daily_mean, label='Predicted Price', color='blue')
    plt.plot(actual_daily_mean.index, actual_daily_mean, label='Actual Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Average Actual vs. Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.show()
    return plt


# CLI commands
@app.command("train-model-and-save")
def train_model_and_save() -> None:
    model = train_model(train_df)
    save_model(model, "model.pkl")


@app.command("make-predictions")
def predict_and_save():
    model = joblib.load("model.pkl")
    initial_date = "2018-07-01 00:00:00"
    n_time_units = 24*31*5
    predictions = make_predictions(initial_date, n_time_units, model, test_df)
    save_predictions(predictions, "energy.db", "predicted_data")

@app.command("plot-predictions")
def plot_predictions():
    #get predictions through table in database predicted_prices
    conn = sqlite3.connect('energy.db')
    predictions = pd.read_sql_query("SELECT * FROM predicted_data", conn)
    actual = pd.read_sql_query("SELECT * FROM energy_set_prep", conn)
    # predictions_df = predictions[['time', 'prediction']]
    # actual_df = actual[['time', 'price actual']]
    print(predictions.dtypes)
    print(actual.dtypes)
    predictions['time'] = pd.to_datetime(predictions['time'])
    actual['time'] = pd.to_datetime(actual['time'])
    print(predictions.head())
    print(actual.head())
    plot1 = plot_predictions_vs_real(predictions, actual)


if __name__ == "__main__":
    app()