import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from music21 import converter, note, chord, key, tempo
from collections import Counter
import glob




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


def extract_features_from_mdi(folder_path, composer):    
    processed_data = pd.DataFrame()
    mid_files = glob.glob(f'{folder_path}*.mid')
    for mid_file in mid_files:
        try:
            # Load mid file
            mid_data  = converter.parse(mid_file)

            # ----------------------------
            # 1. Key Features
            # ----------------------------
            try:
                key_estimation = mid_data.analyze('key')
                key_name = key_estimation.tonic.name  # Tonic (e.g., "C", "D")
                key_mode = key_estimation.mode        # Major or minor mode
                key_strength = key_estimation.correlationCoefficient  # Confidence in key estimation
            except Exception as e:
                print(f"Error analyzing key for {mid_file}: {e}")
                key_name, key_mode, key_strength = None, None, 0

            # Count key signature changes and summarize the timeline
            key_signatures = [ks for ks in mid_data.recurse() if isinstance(ks, key.KeySignature)]
            num_key_signature_changes = len(key_signatures)
            most_frequent_key_signature = (
                Counter([ks.sharps for ks in key_signatures]).most_common(1)[0][0] if key_signatures else None
            )

            # ----------------------------
            # 2. Pitch Features
            # ----------------------------
            pitches = []
            for elem in mid_data.flat.notes:
                if isinstance(elem, note.Note):
                    pitches.append(elem.pitch.ps)
                elif isinstance(elem, chord.Chord):
                    pitches.extend(p.ps for p in elem.pitches)

            average_pitch = np.mean(pitches) if pitches else 0
            median_pitch = np.median(pitches) if pitches else 0
            std_dev_pitch = np.std(pitches) if pitches else 0
            pitch_range = np.ptp(pitches) if pitches else 0
            unique_pitch_classes = len(set(int(p) % 12 for p in pitches))

            # Replace pitch class distribution with entropy (a scalar representation)
            pitch_class_distribution = Counter(int(p) % 12 for p in pitches)
            pitch_entropy = -sum(
                (freq / len(pitches)) * np.log2(freq / len(pitches))
                for freq in pitch_class_distribution.values()
            ) if pitches else 0

            # Extract melodic intervals and summarize
            melodic_intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
            average_melodic_interval = np.mean(melodic_intervals) if melodic_intervals else 0

            # ----------------------------
            # 3. Rhythmic Features
            # ----------------------------
            durations = [elem.quarterLength for elem in mid_data.flat.notes if isinstance(elem, (note.Note, chord.Chord))]
            note_density = len(durations) / (mid_data.quarterLength if mid_data.quarterLength else 1)
            rhythmic_variance = np.var(durations) if durations else 0
            rest_proportion = len([elem for elem in mid_data.flat.notesAndRests if elem.isRest]) / len(durations) if durations else 0

            # ----------------------------
            # 4. Harmonic Features
            # ----------------------------
            chords = [elem for elem in mid_data.flat.notes if isinstance(elem, chord.Chord)]
            chord_progressions = [chord.Chord(elem.pitches).commonName for elem in chords]
            chord_diversity = len(set(chord_progressions)) if chord_progressions else 0
            most_common_chord = Counter(chord_progressions).most_common(1)[0][0] if chord_progressions else None

            # Compute consonance ratio
            consonant_chords = {'major triad', 'minor triad', 'perfect fifth', 'major seventh'}
            num_consonant_chords = sum(1 for chord in chord_progressions if chord in consonant_chords)
            consonance_ratio = num_consonant_chords / len(chord_progressions) if chord_progressions else 0

            # ----------------------------
            # 5. Instrument Features
            # ----------------------------
            instruments = [part.partName for part in mid_data.parts if part.partName]
            num_instruments = len(set(instruments))
            instrument_diversity = len(set(instruments)) if instruments else 0
            most_common_instrument = max(set(instruments), key=instruments.count) if instruments else None

            # ----------------------------
            # 6. Tempo and Structural Features
            # ----------------------------
            tempo_changes = mid_data.flat.getElementsByClass(tempo.MetronomeMark)
            tempo_values = [t.number for t in tempo_changes] if tempo_changes else []
            average_tempo = np.mean(tempo_values) if tempo_values else 0
            min_tempo = min(tempo_values) if tempo_values else 0
            max_tempo = max(tempo_values) if tempo_values else 0
            tempo_variability = np.std(tempo_values) if tempo_values else 0

            # Count time signature changes and summarize
            time_signatures = [ts.ratioString for ts in mid_data.flat.getTimeSignatures()]
            time_signature_changes = len(time_signatures)
            most_frequent_time_signature = (
                Counter(time_signatures).most_common(1)[0][0] if time_signatures else None
            )

            # Structural details
            measure_count = len(mid_data.getElementsByClass('Measure'))
            total_duration = mid_data.duration.quarterLength

            # ----------------------------
            # 7. Compile All Features
            # ----------------------------
            mid_features = {
                "key_name": key_name,
                "key_mode": key_mode,
                "key_strength": key_strength,
                "num_key_signature_changes": num_key_signature_changes,
                "most_frequent_key_signature": most_frequent_key_signature,
                "average_pitch": average_pitch,
                "median_pitch": median_pitch,
                "std_dev_pitch": std_dev_pitch,
                "pitch_range": pitch_range,
                "unique_pitch_classes": unique_pitch_classes,
                "pitch_entropy": pitch_entropy,
                "average_melodic_interval": average_melodic_interval,
                "note_density": note_density,
                "rhythmic_variance": rhythmic_variance,
                "rest_proportion": rest_proportion,
                "chord_diversity": chord_diversity,
                "most_common_chord": most_common_chord,
                "consonance_ratio": consonance_ratio,
                "num_instruments": num_instruments,
                "instrument_diversity": instrument_diversity,
                "most_common_instrument": most_common_instrument,
                "average_tempo": average_tempo,
                "min_tempo": min_tempo,
                "max_tempo": max_tempo,
                "tempo_variability": tempo_variability,
                "time_signature_changes": time_signature_changes,
                "most_frequent_time_signature": most_frequent_time_signature,
                "measure_count": measure_count,
                "total_duration": total_duration,
                "composer": composer
            }

            # Store the features in a dataframe and save to a CSV file
            mid_features_df = pd.DataFrame(mid_features, index=[0])
            processed_data = pd.concat([processed_data, mid_features_df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {mid_file}: {e}")
    return processed_data

