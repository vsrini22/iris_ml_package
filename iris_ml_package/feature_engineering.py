def scale_features(dataframe):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    feature_columns = dataframe.columns.drop("target")
    dataframe[feature_columns] = scaler.fit_transform(dataframe[feature_columns])
    return dataframe
