import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

hou_dat_file_path = 'C:/Users/pc/Desktop/AI and ML/ML/Datasets/Iowa Houses/train.csv'
hou_dat = pd.read_csv(hou_dat_file_path)

a = hou_dat.columns
# print(a)

y = hou_dat.SalePrice
# print(y)

hou_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = hou_dat[hou_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

hou_model = RandomForestRegressor(random_state=1)
hou_model.fit(train_X, train_y)

preds = hou_model.predict(val_X)
mod_mae = mean_absolute_error(val_y, preds)
print(mod_mae)

hou_rf_model = RandomForestRegressor()
hou_rf_model.fit(X,  y)

test_data = 'C:/Users/pc/Desktop/AI and ML/ML/Datasets/Iowa Houses/test.csv'

rf_test_mod_feat = test_data[hou_features]
test_pred = hou_rf_model.predict(train_X)