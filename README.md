# Smoke Detection IoT Project using Support Vector Machine (SVM)
This repository contains a Python script for building a smoke detection system using Support Vector Machine (SVM) classifier. The system analyzes various environmental factors obtained from IoT sensors to determine the presence of smoke or a fire alarm condition.

# Dataset
The dataset used in this project is loaded from a CSV file named "smoke_detection_iot.csv". It contains the following columns:

Temperature[C]: Temperature in Celsius.
TVOC[ppb]: Total Volatile Organic Compounds in parts per billion.
CNT: Carbon Nanotubes concentration.
Fire Alarm: Binary label indicating the presence (1) or absence (0) of a fire alarm condition.
# Data Preprocessing
Checked for and handled missing values.
Checked for and removed outliers using the Interquartile Range (IQR) method.
Removed irrelevant columns: 'eCO2[ppm]', 'Raw H2', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'UTC', 'Humidity[%]', 'Raw Ethanol', 'Pressure[hPa]', 'Unnamed: 0'.
# Model Building
Split the data into training and testing sets (80% training, 20% testing).
Implemented a base SVM model and evaluated its performance using Receiver Operating Characteristic (ROC) curve and Gini Score.
# Hyperparameter Tuning
Conducted a Randomized Search to find the best hyperparameters for the SVM model.
Used the best hyperparameters obtained from the Randomized Search to build an optimized SVM model.
Evaluated the performance of the optimized model using ROC curve and Gini Score.
Results
The base SVM model achieved a Gini Score of 97% on the test data.
The optimized SVM model, obtained through hyperparameter tuning, achieved a Gini Score of 57% on the test data, as it just used one iteration due to the limited computer performance.
