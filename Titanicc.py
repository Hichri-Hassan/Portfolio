# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
def find_best_max_depth(train_X, val_X, train_y, val_y, max_depth_values):
    best_mae = float('inf')
    best_depth = None
    
    # Optional: log-transform target for better performance
    
    for depth in max_depth_values:
        model = RandomForestClassifier(n_estimators=500, max_depth=depth, random_state=1)
        model.fit(train_X, train_y)
        
        val_preds_log = model.predict(val_X)
        val_preds = np.expm1(val_preds_log)  # convert back from log scale
        
        mae = mean_absolute_error(val_y, val_preds)
        print(f"max_depth = {depth} \t MAE = {mae:.0f}")
        
        if mae < best_mae:
            best_mae = mae
            best_depth = depth
            print(f"➡️ New best MAE found! max_depth = {depth}")
    
    print(f"\n✅ Best max_depth = {best_depth} with MAE = {best_mae:.0f}")
    return best_depth

titanic_data = pd.read_csv("C:/Users/hichr/OneDrive/Bureau/Kaagle/Titanic/train.csv")
titanic_test = pd.read_csv("C:/Users/hichr/OneDrive/Bureau/Kaagle/Titanic/test.csv")
# Create target object and call it y
y = titanic_data.Survived

train_data  = pd.get_dummies(titanic_data.drop("Survived", axis=1))
test_data =pd.get_dummies(titanic_test)
train_data , test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)
train_X, val_X, train_y, val_y = train_test_split(train_data, y, random_state=1)
max_depth_candidates = [5, 10, 20, 30,40,50,60, None]
best_depth= find_best_max_depth(train_X, val_X, train_y, val_y, max_depth_candidates)
#max_depth_candidates = [5, 10, 20, 30,40,50,60, None]
#best_depth= find_best_max_depth(train_X, val_X, train_y, val_y, max_depth_candidates)

titanic_model = RandomForestClassifier(
    n_estimators=1000,       # more trees = better performance (to a point)
    max_depth=best_depth,            # limits tree depth to reduce overfittin
    random_state=1          # for reproducibility
)

titanic_model.fit(train_data, y)
test_preds = titanic_model.predict(test_data)
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_preds
})

# 8. Save submission file
submission.to_csv("C:/Users/hichr/OneDrive/Bureau/Kaagle/Titanic/gender_submission.csv", index=False)
print("✅ Submission saved as 'gender_submission.csv'")

print("h")