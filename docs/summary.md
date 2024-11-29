





## Model Evaluation

The classification model achieved an overall accuracy of 86.8%, which indicates strong performance given the data constraints. The confusion matrix shows that the model performs particularly well on the majority class, Beethoven (class 1), correctly classifying 24 out of 25 samples. However, the model struggles with the minority classes, such as Brahms (class 2) and Schubert (class 3), where it exhibits lower recall and F1-scores. For instance, Brahms has a recall of 100%, meaning all actual Brahms samples were correctly identified, but its precision is only 60%, indicating that some predictions were incorrect.

 ![Alt text](../images/confusion_matrix.png "confusion matrix")


The classification report provides deeper insights:

- Precision represents the proportion of correct predictions among all predictions made for a class. For Beethoven, the model achieves a precision of 92%, demonstrating high confidence in its predictions for this class.
- Recall measures the ability of the model to correctly identify all instances of a class. For Bach (class 0), the model achieves a perfect recall of 100%, but for Schubert (class 3), recall drops to 43%, indicating challenges in identifying all Schubert samples.
- F1-score, which balances precision and recall, is high for Beethoven (0.94) but lower for Schubert (0.60), reflecting the difficulty in consistently handling minority classes.

 ![Alt text](../images/classification_report.png "Classification Report")


### Challenges and Observations

An analysis of the data distribution reveals class imbalance, with Beethoven (class 1) having 127 samples compared to only 17–25 samples for other composers. This imbalance likely influences the model's bias toward the majority class, as evidenced by the higher precision and recall for Beethoven. The lower scores for minority classes highlight the impact of limited training data, as well as possible overlaps in the feature space for the less-represented composers.

The model's ability to handle the minority classes could be improved by addressing the imbalance in the training data. Potential solutions include techniques such as oversampling the minority classes, data augmentation, or collecting additional data for the underrepresented composers. Additionally, incorporating more advanced algorithms or domain-specific feature engineering could further improve performance.



##  Discussion on Feature Importance

- The analysis of feature importance reveals that the pitch-related features play a crucial role in predicting the composer of a sound. Specifically, derived features such as pitch_range^2, pitch_range, and interactions like pitch_range num_instruments and consonance_ratio pitch_range are among the top contributors to the model’s predictive performance. These features highlight how variations in pitch and the range of pitch values in a composition are strongly indicative of the composer’s unique style.

    ![Alt text](../images/feature_importance.png "Feature Importance")

- This importance is consistent with musical theory, as composers often have distinct preferences for pitch ranges and patterns. For instance, composers like Beethoven might exhibit wider pitch ranges to emphasize dramatic contrasts, while others like Bach may favor intricate and constrained harmonic structures. Other relevant features, such as time_signature_changes and consonance_ratio, further reflect the rhythmic and harmonic diversity that is characteristic of different composers. Together, these features suggest that the model effectively captures both melodic (pitch-related) and structural (instrumentation and rhythm) elements, leading to its ability to differentiate between composers accurately.