import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_results(df, preds, y_test, results_dir="results"):
    """
    Saves correlation heatmap and Predicted vs Actual comparison.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Matrix")
    plt.savefig(f"{results_dir}/correlation_matrix.png", dpi=300)
    plt.close()
    
    # Predictions vs Actual
    comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=comparison, x="Actual", y="Predicted", alpha=0.6)
    plt.title("Predicted vs Actual")
    plt.savefig(f"{results_dir}/predictions_comparison.png", dpi=300)
    plt.close()
    
    comparison.to_csv(f"{results_dir}/model_comparison_results.csv", index=False)
    print("Results saved successfully to 'results/'")
