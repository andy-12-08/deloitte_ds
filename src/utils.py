import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix



def remove_correlated_features(df, threshold=0.9):
    """
    Removes correlated features from the DataFrame based on the given correlation threshold.

    Parameters:
    - df: pandas DataFrame containing the features.
    - threshold: float, the correlation threshold above which features are considered correlated.

    Returns:
    - pandas DataFrame with correlated features removed.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()
    
    upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than the threshold
    cols_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Drop the correlated features
    reduced_df = df.drop(columns=cols_to_drop)

    return reduced_df

def plot_correlation_heatmap(df, annot=True):
    """
    Plots a heatmap of the correlation matrix for the given DataFrame.

    Parameters:
    - df: pandas DataFrame containing the features.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Add title
    plt.title('Correlation Heatmap', pad=40)
    plt.show()

def remove_low_variance_features(df, threshold=0.01):
    """
    Removes low-variance features from the DataFrame based on the given variance threshold.

    Parameters:
    - df: pandas DataFrame containing the features.
    - threshold: float, the variance threshold below which features are considered low-variance.

    Returns:
    - pandas DataFrame with low-variance features removed.
    """
    # Scale the DataFrame
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Initialize VarianceThreshold with the given threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(scaled_df)

    # Get columns retained after removing low-variance features
    retained_columns = scaled_df.columns[selector.get_support()]
    reduced_df = df[retained_columns]  # Use unscaled DataFrame for final output

    return reduced_df

def plot_pca(df):
    """
    Plots a PCA plot for the given features DataFrame.

    Parameters:
    - features_df: pandas DataFrame containing the features.
    """

    # scale the data
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA plot')
    plt.show()

def confusion_matrix_heatmap(y_test, y_pred):
    """
    Plots a heatmap of the confusion matrix for the given true labels and predicted labels.

    Parameters:
    - y_test: array-like, true labels.
    - y_pred: array-like, predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def plot_feature_importances(importances, features):
    """
    Plots a bar plot of feature importances.

    Parameters:
    - importances: array-like, feature importances.
    - features: array-like, feature names.
    """
    # Create a DataFrame with feature importances
    feature_importances = pd.DataFrame({'feature': features, 'importance': importances})
    
    # Sort features based on importance
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Plot the feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importances['feature'], feature_importances['importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

