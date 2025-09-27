## CO₂ Emission Prediction using Scatter Plots and Linear Regression 
## Developed By: 
 # Name: SUDHARSAN S 
# Register Number: 212224040334 
## Algorithm 
1. Start 
2. Import the required Python libraries (pandas, matplotlib, sklearn). 
3. Load the dataset (CSV file). 
4. Rename the column "Cylinders" → "CYLINDERS". 
5. Plot scatter plots: 
o Q1: CYLINDERS vs CO2EMISSIONS 
o Q2: CYLINDERS vs CO2EMISSIONS and ENGINESIZE vs CO2EMISSIONS 
o Q3: CYLINDERS, ENGINESIZE, FUELCONSUMPTION_COMB vs 
   CO2EMISSIONS 
6. Train regression models: 
o Q4: CYLINDERS vs CO2EMISSIONS 
o Q5: FUELCONSUMPTION_COMB vs CO2EMISSIONS 
7. Evaluate models with different train-test ratios (Q6). 
8. Display accuracy results. 
9. End 
## Program 
``` 
# Program developed by : SUDHARSAN S 
# Register Number      :212224040334 
# Q0: Import Libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
# Load dataset (Upload your CSV in Colab) 
# Example file name: FuelConsumption.csv 
df = pd.read_csv("FuelConsumption.csv") 
# Convert 'Cylinders' to uppercase column name 
df.rename(columns={'Cylinders': 'CYLINDERS'}, inplace=True) 
# Display first rows 
print("\n--- DATASET PREVIEW ---") 
print(df.head()) 
# Q1: Scatter plot CYLINDERS vs CO2EMISSIONS 
print("\nQ1: Scatter plot - CYLINDERS vs CO2EMISSIONS") 
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green') 
plt.xlabel("CYLINDERS") 
plt.ylabel("CO2 EMISSIONS") 
plt.title("CYLINDERS vs CO2 EMISSIONS") 
plt.show() 
# Q2: Compare CYLINDERS vs CO2EMISSIONS and ENGINESIZE vs CO2EMISSIONS 
print("\nQ2: Scatter plot comparison - CYLINDERS vs CO2EMISSIONS and 
ENGINESIZE vs CO2EMISSIONS") 
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='CYLINDERS 
vs CO2') 
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='ENGINESIZE 
vs CO2') 
plt.xlabel("INDEPENDENT VARIABLE") 
plt.ylabel("CO2 EMISSIONS") 
plt.title("Comparison Scatter Plot") 
plt.legend() 
plt.show() 
# Q3: Compare CYLINDERS, ENGINESIZE, FUELCONSUMPTION_COMB vs 
CO2EMISSIONS 
print("\nQ3: Scatter plot comparison - CYLINDERS, ENGINESIZE, 
FUELCONSUMPTION_COMB vs CO2EMISSIONS") 
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='CYLINDERS 
vs CO2') 
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='ENGINESIZE 
vs CO2') 
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='green', 
label='FUELCONSUMPTION_COMB vs CO2') 
plt.xlabel("INDEPENDENT VARIABLE") 
plt.ylabel("CO2 EMISSIONS") 
plt.title("Multiple Comparison Scatter Plot") 
plt.legend() 
plt.show() 
# Q4: Train model (CYLINDERS vs CO2 Emission) 
print("\nQ4: Linear Regression Model - CYLINDERS vs CO2EMISSIONS") 
X_cyl = df[['CYLINDERS']] 
y = df['CO2EMISSIONS'] 
X_train, X_test, y_train, y_test = train_test_split(X_cyl, y, test_size=0.2, 
random_state=42) 
model_cyl = LinearRegression() 
model_cyl.fit(X_train, y_train) 
y_pred_cyl = model_cyl.predict(X_test) 
print("Accuracy (R2 Score):", r2_score(y_test, y_pred_cyl)) 
# Q5: Train model (FUELCONSUMPTION_COMB vs CO2 Emission) 
print("\nQ5: Linear Regression Model - FUELCONSUMPTION_COMB vs 
CO2EMISSIONS") 
X_fuel = df[['FUELCONSUMPTION_COMB']] 
X_train, X_test, y_train, y_test = train_test_split(X_fuel, y, test_size=0.2, 
random_state=42) 
model_fuel = LinearRegression() 
model_fuel.fit(X_train, y_train) 
y_pred_fuel = model_fuel.predict(X_test) 
print("Accuracy (R2 Score):", r2_score(y_test, y_pred_fuel)) 
# Q6: Train with different train-test ratios 
print("\nQ6: Model Accuracy with Different Train-Test Ratios 
(FUELCONSUMPTION_COMB vs CO2EMISSIONS)") 
ratios = [0.2, 0.3, 0.4] 
for ratio in ratios: 
X_train, X_test, y_train, y_test = 
train_test_split(df[['FUELCONSUMPTION_COMB']], y, test_size=ratio, 
random_state=42) 
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
print(f"Train-Test Ratio {1-ratio:.1f}:{ratio:.1f} → Accuracy (R2 Score): 
{r2_score(y_test, y_pred)}") 
``` 
## Output 
<img width="1033" height="494" alt="image" src="https://github.com/user-attachments/assets/17cc00e8-b7c2-414f-827a-44d339d555bd" />

Q1: Scatter plot - CYLINDERS vs CO2EMISSIONS 
 
 <img width="833" height="484" alt="image" src="https://github.com/user-attachments/assets/677a2723-3648-4a6f-a251-f3ecc4162f1b" />

Q2: Scatter plot comparison - CYLINDERS vs CO2EMISSIONS and ENGINESIZE vs 
CO2EMISSIONS 
 
 <img width="780" height="510" alt="image" src="https://github.com/user-attachments/assets/e690f91f-8d86-43e2-8cd7-0875fb20b813" />

Q3: Scatter plot comparison - CYLINDERS, ENGINESIZE, 
FUELCONSUMPTION_COMB vs CO2EMISSIONS 

<img width="798" height="500" alt="image" src="https://github.com/user-attachments/assets/505d9c21-00ae-4517-a230-59a6af98c845" />

Q4: Linear Regression Model - CYLINDERS vs CO2EMISSIONS 

Q5: Linear Regression Model - FUELCONSUMPTION_COMB vs CO2EMISSIONS 

Q6: Model Accuracy with Different Train-Test Ratios 

Output (for q4,q5,q6): 

<img width="783" height="262" alt="image" src="https://github.com/user-attachments/assets/cf09f2da-cb0f-4d3a-8865-a09ac1ffab53" />
