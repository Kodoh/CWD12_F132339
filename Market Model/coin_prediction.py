from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers.legacy import Adam
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import GPyOpt
import matplotlib.pyplot as plt

# Function to create a bidirectional LSTM model
def create_bidirectional_model(params):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=params['num_units'], activation=params['activation_func'], return_sequences=True), input_shape=(params['length'], 1)))
    model.add(Dropout(params['dropout_rate']))

    for _ in range(params['num_lstm_layers'] - 1):
        model.add(Bidirectional(LSTM(units=params['num_units'], activation=params['activation_func'], return_sequences=True)))
        model.add(Dropout(params['dropout_rate']))

    model.add(Bidirectional(LSTM(units=params['num_units'], activation=params['activation_func'])))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to train the model
def train_model(params, ts_generator_train, ts_generator_val):
    model = create_bidirectional_model(params)
    history = model.fit(
        ts_generator_train,
        epochs=params['epochs'],
        validation_data=ts_generator_val,
        verbose=0
    )

    # Make predictions
    predictions = model.predict(ts_generator_val)
    predictions_original_scale = scaler.inverse_transform(predictions[:, :1])[:, 0]

    # Extract actual values from the generator
    actual_values = ts_generator_val[0][1][:, 0]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_values, predictions_original_scale))

    # Calculate sMAPE
    smape = np.mean(200 * np.abs(predictions_original_scale - actual_values) / (np.abs(predictions_original_scale) + np.abs(actual_values)))

    return rmse, smape

# Objective function to minimize
def objective(params):
    params_dict = {
        'num_lstm_layers': int(params[0][0]),
        'num_units': int(params[0][1]),
        'activation_func': 'tanh' if int(params[0][2]) == 0 else 'tanh',
        'length': int(params[0][3]),
        'dropout_rate': float(params[0][4]),
        'epochs': int(params[0][5]),
    }

    rmse, smape = train_model(params_dict, ts_generator_train, ts_generator_val)

    # Combine RMSE and sMAPE as a single objective to minimize
    combined_objective = rmse + smape

    return -combined_objective  # Negative since GPyOpt minimizes

# Function to perform Bayesian optimization using GPyOpt
def bayesian_optimization(ts_generator_train, ts_generator_val):
    space = [
        {'name': 'num_lstm_layers', 'type': 'discrete', 'domain': (2, 3)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (50, 100, 150)},
        {'name': 'activation_func', 'type': 'discrete', 'domain': (0, 1)},  # Mapping 'relu' to 0, 'tanh' to 1
        {'name': 'length', 'type': 'discrete', 'domain': (10, 20, 30)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'epochs', 'type': 'discrete', 'domain': (50, 100)}
    ]

    opt = GPyOpt.methods.BayesianOptimization(
        f=objective,
        domain=space,
        acquisition_type='EI',
        model_type='GP',
        maximize=False
    )

    # Running the optimization
    opt.run_optimization(max_iter=10, verbosity=True)

    # Extracting the best parameters
    best_params = {
        'num_lstm_layers': int(opt.x_opt[0]),
        'num_units': int(opt.x_opt[1]),
        'activation_func': 'tanh' if int(opt.x_opt[2]) == 0 else 'tanh',
        'length': int(opt.x_opt[3]),
        'dropout_rate': float(opt.x_opt[4]),
        'epochs': int(opt.x_opt[5]),
    }

    return best_params



def k_fold_cross_validation(k, data_features, data_target, params):
    tscv = TimeSeriesSplit(n_splits=k)

    rmse_list = []
    smape_list = []

    for train_index, test_index in tscv.split(data_features):
        train_data_features, test_data_features = data_features[train_index], data_features[test_index]
        train_data_target, test_data_target = data_target[train_index], data_target[test_index]

        ts_generator_train = TimeseriesGenerator(train_data_features, train_data_target, length=params['length'], sampling_rate=1, batch_size=16)
        ts_generator_val = TimeseriesGenerator(test_data_features, test_data_target, length=params['length'], sampling_rate=1, batch_size=16)

        rmse, smape = train_model(params, ts_generator_train, ts_generator_val)

        rmse_list.append(rmse)
        smape_list.append(smape)

    # Calculate and return the average RMSE and sMAPE across all folds
    avg_rmse = np.mean(rmse_list)
    avg_smape = np.mean(smape_list)
    return avg_rmse, avg_smape



# Main loop
stocks = ['BTC','ETH','XRP']
fig, axs = plt.subplots(len(stocks), 1, figsize=(12, 6 * len(stocks)))


k_folds = 5

manual_params = {'num_lstm_layers': 3, 'num_units': 150, 'activation_func': 'tanh', 'length': 30, 'dropout_rate': 0.436, 'epochs': 100}

for i, stock in enumerate(stocks):
    scaler = MinMaxScaler()

    # Load and preprocess training data
    csv_path = f'./Daily_Stock_Report/Stocks{stock}-GBP.csv'
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date'))

    plot_features = np.array(df['Open']).reshape(-1, 1)
    target = np.array(df['Close']).reshape(-1, 1)
    axs.set_xlabel('Date')  
    axs.set_ylabel('Close Prices (GBP)')  

    train_size = 0.7
    test_size = 0.15

    train_samples = int(len(plot_features) * train_size)
    test_samples = int(len(plot_features) * test_size)

    all_data = np.concatenate([plot_features, target], axis=1)
    all_data_normalized = scaler.fit_transform(all_data)

    train_data_normalized = all_data_normalized[:train_samples]
    test_data_normalized = all_data_normalized[train_samples:train_samples + test_samples]

    train_data_features = train_data_normalized[:, :-1]
    train_data_target = train_data_normalized[:, -1].reshape(-1, 1)

    test_data_features = test_data_normalized[:, :-1]
    test_data_target = test_data_normalized[:, -1].reshape(-1, 1)

    test_data_target_original = target[train_samples:train_samples + test_samples]

    validation_data_features = plot_features[train_samples + test_samples - 1:]
    validation_data_target = target[train_samples + test_samples - 1:]

    validation_data_normalized = scaler.transform(
        np.concatenate([validation_data_features, validation_data_target], axis=1))
    validation_data_features_normalized = validation_data_normalized[:, :-1]
    validation_data_target_normalized = validation_data_normalized[:, -1].reshape(-1, 1)

    # Create TimeseriesGenerators
    ts_generator_train = TimeseriesGenerator(train_data_features, train_data_target, length=20, sampling_rate=1,
                                                batch_size=16)
    ts_generator_val = TimeseriesGenerator(validation_data_features_normalized, validation_data_target_normalized,
                                            length=20, sampling_rate=1, batch_size=16)

    best_params = manual_params

    # Train the best model
    best_model = create_bidirectional_model(best_params)
    best_model.fit(ts_generator_train, epochs=best_params['epochs'], validation_data=ts_generator_val, verbose=0)

    # Make predictions
    ts_generator_test = TimeseriesGenerator(test_data_features, test_data_target, length=best_params['length'],
                                            sampling_rate=1, batch_size=16)
    predictions = best_model.predict(ts_generator_test)

    test_data_target_original_scale = test_data_target_original
    test_data_target_original_scale = test_data_target_original_scale[best_params['length']:].reshape(-1, 1)

    predictions_with_zeros = np.concatenate([predictions, np.zeros_like(predictions)], axis=1)

    # Inverse transform
    predictions_original_scale = scaler.inverse_transform(predictions_with_zeros)[:, 0]

    # Calculate and print RMSE
    rmse = np.sqrt(mean_squared_error(test_data_target_original_scale, predictions_original_scale))
    print(f"RMSE: {rmse:.4f}")

    # Calculate and print sMAPE
    smape = np.mean(200 * np.abs(predictions_original_scale - test_data_target_original_scale) / (np.abs(predictions_original_scale) + np.abs(test_data_target_original_scale)))
    print(f"sMAPE: {smape:.4f}")

    # Linear regression
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(train_data_features, train_data_target)
    intercept = lin_reg_model.intercept_
    slope = lin_reg_model.coef_

    x_values_numeric = date_time[train_samples + best_params['length']:train_samples + test_samples].astype(np.int64) // 10**9
    linear_regression_line = slope * x_values_numeric + intercept

    # Display the results
    axs.set_title(f"Stock: {stock}, Config: {best_params}, RMSE: {rmse:.4f}, sMAPE: {smape:.4f}")
    axs.plot(date_time[train_samples + best_params['length']:train_samples + test_samples], test_data_target_original_scale,
            label="Actual Close Prices", color='blue')
    axs.plot(date_time[train_samples + best_params['length']:train_samples + test_samples], predictions_original_scale,
            label="LSTM Predicted Close Prices", color='red')
    axs.plot(date_time[train_samples + best_params['length']:train_samples + test_samples], linear_regression_line,
         label="Linear Regression Line", color='green')

    axs.legend()

plt.tight_layout()
plt.show()
