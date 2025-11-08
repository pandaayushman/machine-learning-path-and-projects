from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test):
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Evaluation Results → MAE: {mae:.4f}, R²: {r2:.4f}")
    return preds, mae, r2
