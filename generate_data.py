import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 500 samples
n_samples = 500

hours_studied = np.random.uniform(1, 10, n_samples)
sleep_hours = np.random.uniform(4, 10, n_samples)
attendance_percent = np.random.uniform(50, 100, n_samples)
previous_scores = np.random.uniform(40, 100, n_samples)

# exam_score depends on the features plus some noise
exam_score = (
    10 + 
    3.5 * hours_studied + 
    1.5 * sleep_hours + 
    0.3 * attendance_percent + 
    0.4 * previous_scores + 
    np.random.normal(0, 5, n_samples)
)

# Cap the exam score at 100
exam_score = np.clip(exam_score, 0, 100)

data = pd.DataFrame({
    'student_id': range(1, n_samples + 1),
    'hours_studied': hours_studied,
    'sleep_hours': sleep_hours,
    'attendance_percent': attendance_percent,
    'previous_scores': previous_scores,
    'exam_score': exam_score
})

data.to_csv('student_exam_scores.csv', index=False)
print("Generated student_exam_scores.csv successfully!")
