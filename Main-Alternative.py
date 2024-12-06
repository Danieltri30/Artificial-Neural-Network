# Import all packages
# General packages for analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import random
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Modeling tools
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import keras_tuner as kt
import tensorflow as tf

# Setting Seeds now for Neural Networks Later
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Get the current working directory
# Slight change that now gets the relative path to the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the dataset
dataset_path = os.path.join(current_directory, 'apartments_for_rent_classified_100K.csv')

# Load the dataset
df = pd.read_csv(dataset_path, delimiter=';', encoding='ISO-8859-1')

# Check the first few rows
print(df.head())

# check for duplicate data
duplicates_all = df[df.duplicated()]
print("Exact Duplicate Rows:\n", duplicates_all)

#drop duplicates
df = df.drop_duplicates()

# Get basic information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check summary statistics
print(df.describe())

print(df['has_photo'])

# pets allowed is missing a large number of values and it appears that body does not have enough relevant information that we could hope to impute this value.
# We can either fill the empty values with a place holder or remove it all together. it likely will not be a helpful column...
    # print(df[df['body'].str.contains('Dogs')])
    # print(df[df['body'].str.contains('Cats')])

# For now lets fill it anyways
df.fillna({'pets_allowed':'Unknown'}, inplace=True)
print(df.isnull().sum())

# Alternatively we could impute data. In order to know what would be best; mean, median, or mode we should figure out if data is skewed.
# All of the values are skewed, so intuition suggests that median imuptation is the best here.
plt.figure(1)
plt.hist(df['bathrooms'], bins = 10)
plt.title('Distribution of Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Frequency')
plt.figure(2)
plt.hist(df['bedrooms'], bins = 10)
plt.title('Distribution of Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.figure(3)
plt.hist(df['price'], bins = 100)
plt.title('Distribution of Rent Prices')
plt.xlabel('Rent Price')
plt.ylabel('Frequency')
plt.show()

# They can't be seen very well on the graph above, but there are indeed outliers. Otherwise it would not stretch all the way to 50000
print(df['price'].describe())

# We will set this for later on...
original_max_price = df['price'].max()
original_min_price = df['price'].min()

# Calculate the median price
median_rent_price = df['price'].median()
# Imput values
df.fillna({'price':median_rent_price}, inplace=True)

# Calculate the median bathroom count
median_bathroom = df['bathrooms'].median()
# Imput values
df.fillna({'bathrooms':median_bathroom}, inplace=True)

# Calculate the median bedroom count
median_bedroom = df['bedrooms'].median()
# Imput values
df.fillna({'bedrooms':median_bedroom}, inplace=True)

# Imputing latitude and longitude does not make sense, so these still have to be removed like before.
# We are also removing null cityname and state rows, since there is no guarantee we can fill them with information from body and title and the population of missing data is so small it is not worth the effort.
df = df.dropna(subset = ['latitude', 'longitude', 'cityname', 'state'])
# Check for missing values
print(df.isnull().sum())
print(df.shape)
# The helpfulness of some of these variables is questionable, but we wont know for sure until we move into week 2

# lets take a look at where all of these rentals are

# Create a plot with Cartopy
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())  # Limit to U.S. boundaries

# Add features for land, coastlines, and borders
ax.stock_img()
ax.coastlines()
ax.add_feature(cfeature.BORDERS)

# Plot each point on the map
plt.scatter(df['longitude'], df['latitude'], color='green', s=10, transform=ccrs.PlateCarree(), label='Locations')

plt.title("Map of Rental Locations")
plt.legend()
plt.show()

#Lets convert has photo and pets allowed to numerical variables
# One-hot encode both has photo and pets allowed
# First we will break up pets allowed
# Split the 'pets_allowed' column by comma and expand into separate columns
split_df = df['pets_allowed'].str.split(',', expand=True)

# Create binary columns for 'cats', 'dogs', 'unknown', 'Yes', 'Thumbnail'
df['cats_allowed'] = split_df.apply(lambda x: 1 if 'Cats' in x.values else 0, axis=1)
df['dogs_allowed'] = split_df.apply(lambda x: 1 if 'Dogs' in x.values else 0, axis=1)
df['unknown_allowed'] = split_df.apply(lambda x: 1 if 'Unknown' in x.values else 0, axis=1)
df['has_photo_Thumbnail'] = df.apply(lambda x: 1 if 'Thumbnail' in x.values else 0, axis=1)
df['has_photo_Yes'] = df.apply(lambda x: 1 if 'Yes' in x.values else 0, axis=1)

# Use pandas to create one-hot encoded columns
df = pd.get_dummies(df, columns=['category'], prefix='category')
# Convert only boolean columns to integers
boolean_columns = df.select_dtypes(include='bool').columns
df[boolean_columns] = df[boolean_columns].astype(int)
print(df.dtypes)

# Check Correlations
print(df.select_dtypes(include='number').corr())

# Triple checking there are no duplicates, since I made a mistake earlier that made duplicate columns. It appears to be corrected now...
# Check for duplicate columns by comparing column names
duplicate_columns = df.columns[df.columns.duplicated()]

# Print out the duplicate columns, if any
if len(duplicate_columns) > 0:
    print(f"Duplicate columns found: {duplicate_columns}")
else:
    print("No duplicate columns found.")

# It might be worthwhile to convert the unix timestamp to actual dates, but since we are using it to predict a continuous feature and the correlation isnt really there based on prior observations. 
# Convert Unix timestamp to datetime
df['datetime'] = pd.to_datetime(df['time'], unit='s')

# Extract time-based features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['day_of_week'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour

print(df.describe())

print(df.select_dtypes(include='number').columns)

# There isn't a great way to make use of the amenities column and trying to dig out a few thousand addresses from the body column to put in address does not seem worthwhile when a majority of it is empty, So we will drop these columns.
# We will drop body and title, since we have already extracted what we can from them. id and currency dont provide any value to what we are trying to accomplish.
df = df.drop(columns = ['amenities', 'address', 'body', 'title', 'id', 'currency', 'price_display', 'source', 'has_photo', 'time'])

# The data is highly skewed so lets apply a log transform to correct that. 
# We cant scale longitude due to negative values, so we will leave out coordinates.

def log_transform_and_min_max_normalize(df, column):
    # Log transformation with a small constant to avoid log(0)
    df[column] = np.log1p(df[column])
    
    # Min-max normalization to [0, 1] after log transformation
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)

# Apply log transformation and min-max normalization to each numeric column
numeric_columns = [
    'price', 'bathrooms', 'bedrooms', 'square_feet', 'cats_allowed', 'dogs_allowed', 'category_housing/rent',
    'category_housing/rent/apartment', 'category_housing/rent/commercial/retail',
    'category_housing/rent/condo', 'category_housing/rent/home',
    'category_housing/rent/other', 'category_housing/rent/short_term',
    'unknown_allowed', 'has_photo_Thumbnail', 'has_photo_Yes', 'year',
    'month', 'day', 'day_of_week', 'hour'
]

# Apply transformations to each numeric column
for col in numeric_columns:
    log_transform_and_min_max_normalize(df, col)

print(df.describe())

# Set up data
X = df[['bathrooms', 'bedrooms', 'square_feet', 'latitude',
       'longitude', 'cats_allowed', 'dogs_allowed', 'unknown_allowed', 
       'category_housing/rent', 'category_housing/rent/apartment', 
       'category_housing/rent/commercial/retail',
       'category_housing/rent/condo', 'category_housing/rent/home',
       'category_housing/rent/other', 'category_housing/rent/short_term',
       'has_photo_Thumbnail', 'has_photo_Yes', 'year',
       'month', 'day', 'day_of_week', 'hour']]

y = df['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# Save the test data and train data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Are the datasets balanced?
plt.figure(figsize=(10, 6))
sns.histplot(y_train, bins=100, kde=True)
plt.title('Distribution of Price (Train Set)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# They appear to be balanced
plt.figure(figsize=(10, 6))
sns.histplot(y_test, bins=100, kde=True)
plt.title('Distribution of Price (Test Set)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Set up models and initialize dictionaries
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),  
    "Lasso Regression": Lasso()   
}

results = {}
predictions_dict = {}

# mean squared error is deprecated and the new version of rmse doesn't seem to work on my version, so we are just calculating it manually.
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, predictions)

    # Manually calculate RMSE
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

    # Calculate R²
    r2 = r2_score(y_test, predictions)

    # Store the results
    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    predictions_dict[name] = predictions

print(results)

# Set up subplots for each model
num_models = len(predictions_dict)
fig, axes = plt.subplots(1, num_models, figsize=(16, 6), sharey=True)

for i, (name, predictions) in enumerate(predictions_dict.items()):
    axes[i].scatter(y_test, predictions, alpha=0.5)
    # Plot a reference line for perfect predictions
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    axes[i].set_xlabel('Actual Rent')
    axes[i].set_title(f'{name}: Predicted vs Actual Rent')

# Set the common y-label and main title
axes[0].set_ylabel('Predicted Rent')
fig.suptitle('Predicted Rent vs Actual Rent for Different Models')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Some of the prediction values seemed a little funky so we are double checking everything is unique
# Check if y_train and y_test have the same values
if np.array_equal(y_train.values, y_test.values):
    print("y_train and y_test have the same values.")
else:
    print("y_train and y_test are different.")

# Check unique values in y_train and y_test
print("Unique values in y_train:", y_train.unique())
print("Unique values in y_test:", y_test.unique())

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso()
}

# Initialize K-Folds
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Initialize results storage
results = {}

for name, model in models.items():
    print(f"\nEvaluating {name} with K-Fold Cross-Validation")
    
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit the model and make predictions
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        
        # Calculate RMSE manually
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        # Calculate R²
        r2 = r2_score(y_test, predictions)

        # Append scores to lists
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        print(f"Model: {name}, Fold: {fold}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
    # Store mean and standard deviation of each metric for each model
    results[name] = {
        "Mean MAE": np.mean(mae_scores),
        "Std MAE": np.std(mae_scores),
        "Mean RMSE": np.mean(rmse_scores),
        "Std RMSE": np.std(rmse_scores),
        "Mean R²": np.mean(r2_scores),
        "Std R²": np.std(r2_scores)
    }

print(results)

# Extract feature names
feature_names = X.columns

# Dictionary to store feature importances for each model
feature_importances = {}

for model_name, model in models.items():
    # Extract coefficients
    coefs = model.coef_
    
    # Create a DataFrame for easy viewing
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefs
    })
    
    # Sort by absolute value of coefficients to identify top features
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False)
    
    # Store sorted DataFrame for each model
    feature_importances[model_name] = importance_df

    # Display feature importance
    print(f"Top features for {model_name}:\n")
    print(importance_df[['Feature', 'Importance']], "\n")

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='b')
    plt.xlabel("Coefficient Value")
    plt.title(f"Top 10 Features for {model_name}")
    plt.gca().invert_yaxis()
    plt.show()

# results indicate that day, month, hour are our most likely candidate to be removed from future modeling. removing these does not change model performance negatively or positively for regression purposes.

# We began with establishing a basic neural network model
# We used FeedForward Architecture, this implies that data flows from input to output without cycles
# One hidden layer , Output layer is one neuron wide
# Relu for activiation function, Linear is used in output layer for regression tasks
# We have a optimized Loss Function
# Slight hyperparameter tuning, made slight adjustments to fed data size and epochs.

# Custom R² metric
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + K.epsilon()))
    return r2


def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2_metric]) 
    return model


class NeuralNetwork:

    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = Sequential()

    def b_model(self, hp=None):
        self.model = Sequential()
        units = hp.Int('units', min_value=32, max_value=128, step=32) if hp else 64  # Default to 64 units if hp is None
        self.model.add(Dense(units=units, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2_metric]) 

    def train(self, epochs=50, batch_size=32):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size)

    def evaluate(self): 
        loss, mae, r2 = self.model.evaluate(self.X_test, self.y_test)
        print(f"MAE : {mae}")
        print(f"R² : {r2}")


def main():
    tuner = kt.Hyperband(
        build_model,
        objective='val_mae',
        max_epochs=50,
        directory='tuner_results',
        project_name='apartment_model'
    )

    tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    nn = NeuralNetwork(X_train, X_test, y_train, y_test)
    nn.b_model(hp=best_hps)
    nn.train()
    nn.evaluate()


if __name__ == "__main__":
    main()
                
# This is one of many other models we tried
# Loss: 0.0013324442552402616, MAE: 0.027234883978962898, R²: 0.6954784989356995

def build_model(hp):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(365, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae', r2_metric]
    )
    return model

# NeuralNetwork Class
class NeuralNetwork:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None  # Initialize as None

    def build_model(self):
        self.model = build_model(None)  # Use the standalone build_model function

    def train(self, epochs=45, batch_size=32):
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
        )
        self.model.save("trained_rnn_model.h5")
        print("Model saved as 'trained_rnn_model.h5'")

    def evaluate(self):
        result = self.model.evaluate(self.X_test, self.y_test)
        if len(result) == 3:  
            loss, mae, r2 = result
            print(f"Loss: {loss}, MAE: {mae}, R²: {r2}")
        else:
            loss = result
            print(f"Loss: {loss}")

# Main function
def main():
    # Perform hyperparameter tuning with Hyperband
    tuner = kt.Hyperband(
        build_model,  # Reference the standalone function
        objective='val_mae',
        max_epochs=45,
        directory='tuner_results',
        project_name='apartment_model',
    )

    tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=32)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters: {best_hps.values}")

    # Use the NeuralNetwork class for training and evaluation
    nn = NeuralNetwork(X_train, X_test, y_train, y_test)
    nn.build_model()
    nn.train()
    nn.evaluate()

if __name__ == "__main__":
    main()

# Final graph

class NeuralNetwork:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = Sequential()
    def predict_future(self, steps=365):
        """
        Generate future predictions starting from the last sequence in the test set.
        """
        last_sequence = self.X_test.iloc[-1].values.copy()  # Ensure this is a NumPy array
        predictions = []

        for _ in range(steps):
            # Prepare the input sequence
            input_sequence = last_sequence.reshape(1, -1, 1)
            next_prediction = self.model.predict(input_sequence)
            predictions.append(next_prediction[0, 0])
            
            # Update the sequence
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_prediction[0, 0]
        
        return predictions

    def plot_predictions(self, predictions):
        """
        Plot the predictions and convert y-axis ticks to display prices in dollars.
        """
        # Plot the normalized predictions
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(predictions) + 1), predictions, label="Predicted Prices")

        # Access the current axes
        ax = plt.gca()

        # Get the original y-ticks from the plot
        original_y_ticks = ax.get_yticks()
        print("Original y-ticks (normalized):", original_y_ticks)  # Debugging info

        # Map the normalized y-ticks directly to dollar values
        y_tick_labels = []
        for tick in original_y_ticks:
            # Apply reverse normalization to get the actual dollar value
            dollar_value = tick * (original_max_price - original_min_price) + original_min_price
            
            # If dollar_value is out of reasonable bounds, handle it appropriately
            if dollar_value > 0 and dollar_value < float('inf'):
                y_tick_labels.append(f"${dollar_value:,.2f}")
                print(f"Tick {tick} converted to ${dollar_value:,.2f}")
            else:
                y_tick_labels.append("$∞")
                print(f"Tick {tick} resulted in an invalid value; set to $∞")

        # Update the y-axis with new tick labels
        ax.set_yticks(original_y_ticks)
        ax.set_yticklabels(y_tick_labels)

        # Add plot details
        plt.xlabel("Days")
        plt.ylabel("Predicted Rental Price ($)")
        plt.title("Predicted Rental Prices for the Next Year based on all types of housing")
        plt.legend()
        plt.grid()
        plt.show()  

# Main function
def main():

    print("Loading trained model...")

    # Initialize the NeuralNetwork class with globally defined data
    nn = NeuralNetwork(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Load the trained model
    nn.model = load_model("trained_rnn_model.h5", custom_objects={"r2_metric": r2_metric})

    # Predict future values
    future_predictions = nn.predict_future(steps=365)
    
    # Debugging info
    print("Min Price:", original_min_price)
    print("Max Price:", original_max_price)
    print("Predictions (first 5):", future_predictions[:5])

    # Plot the predicted values
    nn.plot_predictions(future_predictions)


if __name__ == "__main__":
    main()

print("Script Finished.")
