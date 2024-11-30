from sklearn.preprocessing import StandardScaler

def scale_features(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.iloc[:, :-1])  # Scale features except the target
    data.iloc[:, :-1] = data_scaled
    return data
