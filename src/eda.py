import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.show()

def plot_target_distribution(df, target):
    sns.histplot(df[target], kde=True)
    plt.show()