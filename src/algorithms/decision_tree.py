from sklearn.tree import DecisionTreeRegressor

def get_model():
    return DecisionTreeRegressor(random_state=42)