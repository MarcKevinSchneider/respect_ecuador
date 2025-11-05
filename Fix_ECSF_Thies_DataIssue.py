# Purpose: Model ECSF Thies gRad2 values from ECSF Campbell radiation data
# Author: Marc Kevin Schneider
# Date: November 2025

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# path to folder
path = '~/CSV_Files'

# read the data
# ecsf thies station
ecsf = pd.read_csv(f'{path}/ECSFThies_2022_2024.csv')
# ecsf campbell station
ecsf_campbell = pd.read_csv(f'{path}/ECSF_Campbell_2022_2024.csv')

ecsf['datetime'] = pd.to_datetime(ecsf['datetime'])
ecsf_campbell['datetime'] = pd.to_datetime(ecsf_campbell['datetime'])

########################################################################

# overview plot over grad2 

# sort by datetime just in case the data isn’t in order
ecsf = ecsf.sort_values('datetime')

# create figure
plt.figure(figsize=(12, 6))
# plot as lineplot
plt.plot(ecsf['datetime'], ecsf['gRad2'], color='orange', linewidth=1)

# labels and title
plt.title('Incoming global radiation (gRad2)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Global Radiation (W/m²)', fontsize=12)

# formatting
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(pd.to_datetime('2022-01-01'), pd.to_datetime('2024-12-31'))
plt.tight_layout()

# show
plt.show()


########################################################################

# same for campbell data

# sort by datetime just in case the data isn’t in order
ecsf_campbell = ecsf_campbell.sort_values('datetime')

# create figure
plt.figure(figsize=(12, 6))
# plot as lineplot
plt.plot(ecsf_campbell['datetime'], ecsf_campbell['SlrkW_Avg'], color='orange', linewidth=1)

# labels and title
plt.title('Incoming global radiation (SlrkW_Avg)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Global Radiation (kW/m²)', fontsize=12)

# formatting
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(pd.to_datetime('2022-01-01'), pd.to_datetime('2024-12-31'))
plt.tight_layout()

# show
plt.show()

################################################################################
################################################################################

# Modeling the missing data at the ECSF thies station using a random forest model

# uses the other parameters measured at the station to approximate gRad2
# (had to use this since the Campbell station has a lot of missing data)

#################################################################################
#################################################################################

# ecsf thies station
ecsf = pd.read_csv(f'{path}/ECSFThies_2022_2024.csv')
# ecsf campbell station
ecsf_campbell = pd.read_csv(f'{path}/ECSF_Campbell_2022_2024.csv')

ecsf['datetime'] = pd.to_datetime(ecsf['datetime'])
ecsf_campbell['datetime'] = pd.to_datetime(ecsf_campbell['datetime'])

ecsf = ecsf.set_index('datetime')

# period with faulty data
faulty_start = '2023-06-29'
faulty_end   = '2024-02-04'

# period with good data
good_data = ecsf[(ecsf.index < faulty_start) | (ecsf.index > faulty_end)].copy()
# period with faulty data
faulty_data = ecsf[(ecsf.index >= faulty_start) & (ecsf.index <= faulty_end)].copy()

# add time parameters 
for df in [good_data, faulty_data]:
    df['hour'] = df.index.hour
    df['doy']  = df.index.dayofyear

# predictors
features = ['Ta', 'Huma', 'gRaddown', 'PCP', 'hour', 'doy']
# dropna for grad2
good_data = good_data.dropna(subset=['gRad2'])

# interpolating small missing data steps
faulty_data[features] = faulty_data[features].interpolate(limit_direction='both')

# drop all NaN
train_data_clean = good_data[features + ['gRad2']].dropna()
# predictors
X = train_data_clean[features]
# predicted variable
y = train_data_clean['gRad2']

# 80% training 20% testing split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train RF model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# validate on the testing data
y_val_pred = model.predict(X_val)
r2_val = r2_score(y_val, y_val_pred)
rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

print(f"Validation R²: {r2_val:.3f}") # R2 = 0.926
print(f"Validation RMSE: {rmse_val:.2f} W/m²") # RMSE = 72.47W/m2

# scatter plot: predicted vs. observed values
plt.figure(figsize=(12,10))
plt.scatter(y_val, y_val_pred, alpha=0.5, label="Observed data")
plt.plot([0, y_val.max()], [0, y_val.max()], 'r--', label='1:1 line')

# annotate R2 and RMSE
plt.text(
    0.005 * y_val.max(), 0.995 * y_val.max(),  
    f"R² = {r2_val:.3f}\nRMSE = {rmse_val:.1f} W/m²",
    fontsize=14
)

# labels etc.
plt.xlabel('Observed Radiation (W/m²)')
plt.ylabel('Predicted Radiation (W/m²)')
plt.title('Random Forest Validation: Predicted vs. observed incoming global radiation')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("~/ECSF_Thies/RF_Model_Validation.png", dpi=300)
plt.show()

# predicting for faulty period
X_faulty = faulty_data[features]
faulty_data['gRad2_corrected'] = model.predict(X_faulty)

# combine back into ecsf dataframe
ecsf['gRad2_filled'] = ecsf['gRad2']
ecsf.loc[faulty_data.index, 'gRad2_filled'] = round(faulty_data['gRad2_corrected'], 0)

# flag values that were modeled or measured
ecsf['gRad2_source'] = np.where(
    ecsf.index.isin(faulty_data.index), 'modeled', 'measured'
)


# save dataset
ecsf_reset = ecsf.reset_index()
ecsf_reset.to_csv(f'{path}/ECSFThies_2022_2024_filled.csv', index=False)


# plot over time
# create figure
plt.figure(figsize=(12,6))
# filter for measured or modeled
measured = ecsf[ecsf['gRad2_source'] == 'measured']
modeled  = ecsf[ecsf['gRad2_source'] == 'modeled']

# plot both
plt.plot(measured.index, measured['gRad2_filled'], color='orange', linewidth=1.0, label='Measured')
plt.plot(modeled.index, modeled['gRad2_filled'], color='deepskyblue', linewidth=1.0, label='Modeled (filled)')

# labels etc.
plt.title('Incoming Global Radiation (gRad2)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Incoming Global Radiation (W/m²)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(pd.to_datetime('2022-01-01'), pd.to_datetime('2024-12-31'))
plt.legend()
plt.tight_layout()
plt.show()


################################################################################
################################################################################

# Filling the remaining large gap between 2023-02-08 and 2023-02-24 with the
# campbell data since it was online at the time

#################################################################################
#################################################################################



# ecsf thies station
ecsf = pd.read_csv(f'{path}/ECSFThies_2022_2024_filled.csv')
# ecsf campbell station
ecsf_campbell = pd.read_csv(f'{path}/ECSF_Campbell_2022_2024.csv')

# datetime
ecsf['datetime'] = pd.to_datetime(ecsf['datetime'])
ecsf_campbell['datetime'] = pd.to_datetime(ecsf_campbell['datetime'])

# setting index for resampling and merging
ecsf_campbell = ecsf_campbell.set_index('datetime')
ecsf = ecsf.set_index('datetime')

# resample campbell to 30min to match ecsf thies format
ecsf_campbell_30min = ecsf_campbell.resample('30min').mean()
# Convert SlrkW_Avg to W/m2 (simply multiplying by 1000)
ecsf_campbell_30min['SlrkW_Avg_Wm2'] = ecsf_campbell_30min['SlrkW_Avg'] * 1000

# merging both
merged = ecsf[['gRad2']].merge(
    ecsf_campbell_30min[['SlrkW_Avg_Wm2']], 
    left_index=True, right_index=True, how='inner'
)

# define training period (outside the offline gap)
train_data = merged[
    (merged.index < '2023-02-08') | (merged.index > '2024-02-05')
].dropna()

# traing linear regression
X_train = train_data[['SlrkW_Avg_Wm2']]
y_train = train_data['gRad2']

model = LinearRegression()
model.fit(X_train, y_train)

# evaluation
y_pred_train = model.predict(X_train)
r2 = r2_score(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train, squared=False)

print(f'Model R²: {r2:.3f}, RMSE: {rmse:.2f} W/m²') # R2 = 0.912
print(f'Model equation: gRad2 = {model.coef_[0]:.3f} * SlrkW_Avg_Wm2 + {model.intercept_:.3f}') # RMSE = 80.33W/m2
# Model equation: gRad2 = 1.036 * SlrkW_Avg_Wm2 + 7.533

# period when ecsf thies was offline
offline_start = '2023-02-08 18:30'
offline_end   = '2023-02-24 11:30'

# get campbell data for offline period
offline_campbell = ecsf_campbell_30min.loc[offline_start:offline_end, 
                                           ['Ta', 'Huma', 'winddirection', 'Windspeed', 'SlrkW_Avg_Wm2']].copy()

# rename column
offline_campbell = offline_campbell.rename(columns={"winddirection": "Wind_direction"})

# predict grad2
X_pred = offline_campbell[['SlrkW_Avg_Wm2']]
offline_campbell['gRad2_predicted'] = model.predict(X_pred)
# round to no decimal space
offline_campbell['gRad2_predicted'] = round(offline_campbell['gRad2_predicted'], 0)

# ensuring ecsf thies has all timestamps
full_index = pd.date_range(start=offline_start, end=offline_end, freq='30min')
ecsf = ecsf.reindex(ecsf.index.union(full_index))

# use campbell data for the columns where it is possible
ecsf.loc[offline_campbell.index, 'Ta'] = round(offline_campbell['Ta'], 1)
ecsf.loc[offline_campbell.index, 'Huma'] = round(offline_campbell['Huma'], 1)
ecsf.loc[offline_campbell.index, 'Wind_direction'] = round(offline_campbell['Wind_direction'], 0)
ecsf.loc[offline_campbell.index, 'Windspeed'] = round(offline_campbell['Windspeed'], 1)

# ecsf-only columns set to NaN
ecsf.loc[offline_campbell.index, ['gRaddown', 'PCP', 'Ts_10cm', 'Ts_60cm']] = np.nan

# fill gRad2_filled with predicted values
ecsf.loc[offline_campbell.index, 'gRad2_filled'] = round(offline_campbell['gRad2_predicted'], 0)

# flag this period as modeled
ecsf.loc[offline_campbell.index, 'gRad2_source'] = 'modeled'

# set nighttime 8 W/m2 to 0 (apparently the baseline value for night with this model)
ecsf['gRad2_filled'] = ecsf['gRad2_filled'].mask(ecsf['gRad2_filled'] == 8, 0)


# update source flag for measured values outside offline period
ecsf['gRad2_source'] = np.where(
    ecsf.index.isin(offline_campbell.index),
    'modeled',
    ecsf.get('gRad2_source', 'measured')
)

# resetting the index
ecsf_reset = ecsf.reset_index()
# renaming back to datetime
ecsf_reset = ecsf_reset.rename(columns={"index":"datetime"})

# remove winddirection from df (annoying column which wont go away)
if 'winddirection' in ecsf_reset.columns:
    ecsf_reset = ecsf_reset.drop(columns=['winddirection'])

# save as csv
ecsf_reset.to_csv(f'{path}/ECSFThies_2022_2024_Filled_WithCampbell.csv', index=False)


#######################################################################################
######################################################################################

# plots

# read data 
ecsf_reset = pd.read_csv(f'{path}/ECSFThies_2022_2024_Filled_WithCampbell.csv')
ecsf_reset["datetime"] = pd.to_datetime(ecsf_reset["datetime"])

# scatter plot of training data
plt.figure(figsize=(12,10))
plt.scatter(X_train, y_train, alpha=0.4, label='Observed data')
plt.plot([0, X_train.max()], [0, X_train.max()], 'r--', label='1:1 line')
plt.plot(X_train, y_pred_train, color='orange', alpha=0.6, label='Linear fit')
plt.xlabel('Radiation at the ECSF Campbell station (W/m²)', fontsize=12)
plt.ylabel('Radiation at the ECSF Thies station (W/m²)', fontsize=12)
plt.title('Comparison of incoming global radiation from both stations', fontsize=13)
plt.text(0.05 * X_train.max(), 0.9 * y_train.max(),
         f'R² = {r2:.3f}\nRMSE = {rmse:.1f} W/m²', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("~/ECSF_Thies/LinearRegression_Model.png", dpi=300)
plt.show()

# modeled and observed over time
plt.figure(figsize=(12,6))
measured = ecsf_reset[ecsf_reset['gRad2_source'] == 'measured']
modeled  = ecsf_reset[ecsf_reset['gRad2_source'] == 'modeled']

plt.bar(measured["datetime"], measured['gRad2_filled'], color='orange', width=2, label='Measured')
plt.bar(modeled["datetime"], modeled['gRad2_filled'], color='deepskyblue', width=2, label='Modeled (filled)')

plt.title('Incoming Global Radiation (gRad2)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Incoming Global Radiation (W/m²)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(pd.to_datetime('2022-01-01'), pd.to_datetime('2024-12-31'))
plt.legend()
plt.tight_layout()
plt.show()
