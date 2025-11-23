# utils/evaluation.py

# TODO: Task 4: Evaluation of Detection Quality (15%)
'''
For a subset of frames, compute detection metrics (when bounding-box ground truth is available):
• Precision, recall, F1 score using a suitable IoU threshold (e.g., 0.5).
Discuss common failure cases: missed people (false negatives) and non-person detections (false positives).
'''
def evaluate_detection_quality():
    pass

# TODO: Task 5: Robustness to Degradation (10%)
'''
# To assess robustness, deliberately degrade your input frames using at least two different types of degradation, 
# such as Gaussian blur, heavy downsampling followed by upsampling, additive noise, or strong JPEG
# compression artifacts. For each degradation type and level, re-run your detection and counting pipeline
# and evaluate how the performance changes, again using counting and/or detection metrics. In your re-
# port, compare results across conditions and explain why certain degradations harm performance more
# than others, linking these observations to properties of your chosen model and preprocessing pipeline.
'''
def evaluate_robustness_to_degradation():
    pass