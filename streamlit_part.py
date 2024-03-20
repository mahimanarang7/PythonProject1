import streamlit as st
import pandas as pd
from model_and_evaluation import loading_file, train_model, save_model, make_predictions, plot_predictions_vs_real
from model_and_evaluation import plot_time_vs_price_initial
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = loading_file()


import matplotlib.pyplot as plt

def exploratory_analysis():
    st.title("Exploratory Data Analysis")
    st.subheader("Data Overview")
    st.write(df.head())

    # Add visualizations
    st.subheader("Visualizations")

    # Example: Plotting a line chart
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['price actual'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Price over Time')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    daily_meanx = df.groupby(df['time'].dt.to_period('M')).mean()
    daily_meanx.index = daily_meanx.index.to_timestamp()
    ax.plot(daily_meanx.index, daily_meanx["price actual"])
    ax.set_xlabel('Time(Monthly)')
    ax.set_ylabel('Price')
    ax.set_title('Price over Months')
    st.pyplot(fig)

    # Example: Plotting a histogram
    fig, ax = plt.subplots()
    ax.hist(df['price actual'], bins=20)
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Price Distribution')
    st.pyplot(fig)

    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, annot=True)
    st.pyplot(plt)

    # Add more visualizations, descriptive statistics, etc.

def prediction_page():
    st.title("Make Predictions")

    # Train the model
    with st.form("train_model_form"):
        st.subheader("Train the Model")
        train_model_btn = st.form_submit_button("Train Model and Save")
        if train_model_btn:
            model = train_model(df)
            save_model(model, "model.pkl")
            st.success("Model trained and saved successfully!")

    # Make predictions
    with st.form("make_predictions_form"):
        st.subheader("Make Predictions")
        initial_date = st.date_input("Select the initial date for predictions", value=pd.to_datetime("2018-07-01"))
        n_time_units = st.number_input("Enter the number of time units (e.g., hours, days) for predictions", value=24 * 31 *5, step=1)
        make_predictions_btn = st.form_submit_button("Make Predictions")
        if make_predictions_btn:
            model = joblib.load("model.pkl")
            predictions = make_predictions(initial_date.strftime('%Y-%m-%d %H:%M:%S'), n_time_units, model, df)
            st.success("Predictions made successfully!")

            # Plot predictions
            plot_predictions_vs_real(predictions, df)

    with st.form("plot_predictions_form"):
        st.subheader("Plot Predictions")
        plot_predictions_btn = st.form_submit_button("Plot Predictions")
        if plot_predictions_btn:
            conn = sqlite3.connect('energy.db')
            predictions = pd.read_sql_query("SELECT * FROM predicted_data", conn)
            actual = pd.read_sql_query("SELECT * FROM energy_set_prep", conn)
            predictions_df = predictions[['time', 'prediction']]
            actual_df = actual[['time', 'price actual']]
            predictions_df['time'] = pd.to_datetime(predictions_df['time'])
            actual_df['time'] = pd.to_datetime(actual_df['time'])
            plotx = plot_predictions_vs_real(predictions_df, actual_df)
            st.pyplot(plotx)
            st.success("Predictions plotted successfully!")

pages = {
    "Exploratory Data Analysis": exploratory_analysis,
    "Make Predictions": prediction_page}
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()



