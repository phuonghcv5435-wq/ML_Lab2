from sklearn.ensemble import GradientBoostingRegressor

def get_model():
    return GradientBoostingRegressor(n_estimators=100, random_state=42)