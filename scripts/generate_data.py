import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_students=1000):
    """Generates synthetic student data for training."""
    
    np.random.seed(42)
    random.seed(42)

    data = []
    
    for student_id in range(1, num_students + 1):
        # Simulate student behavior
        # 3 Courses available
        course_id = random.choice(['C101', 'C102', 'C103'])
        
        # Number of chapters varies by course
        num_chapters = {'C101': 10, 'C102': 12, 'C103': 8}[course_id]
        
        # Randomly decide if student is a "completer" or "dropout" based on some latent factors
        # This helps create realistic correlations
        student_capability = np.random.normal(0.6, 0.2) # 0 to 1 scale roughly
        student_capability = max(0.1, min(0.95, student_capability))
        
        for chapter in range(1, num_chapters + 1):
            # Time spent (minutes) - correlated with difficulty and capability
            # Let's say chapters 3, 7 are hard
            difficulty = 1.0
            if chapter in [3, 7]:
                difficulty = 1.5
            
            base_time = 30 * difficulty
            time_spent = np.random.normal(base_time, 10) * (1.5 - student_capability)
            time_spent = max(5, time_spent)
            
            # Score (0-100)
            base_score = 70 * student_capability
            score = np.random.normal(base_score, 15) / difficulty
            score = max(0, min(100, score))
            
            data.append({
                'student_id': student_id,
                'course_id': course_id,
                'chapter_id': chapter,
                'time_spent': round(time_spent, 2),
                'score': round(score, 2),
                # Latent factor for training target generation, not a feature
                '_capability': student_capability 
            })

    df = pd.DataFrame(data)
    
    # Generate Target Variable: Completion Status
    # Logic: If average score < 50 or total time < expected * 0.5 -> Dropout
    # This is a simplification for the ground truth
    
    student_stats = df.groupby('student_id').agg({
        'score': 'mean',
        'time_spent': 'sum',
        '_capability': 'first'
    }).reset_index()
    
    # Define completion logic (Ground Truth)
    # A student completes if they have decent scores and engagement
    student_stats['completed'] = (student_stats['score'] > 55) & (student_stats['_capability'] > 0.4)
    student_stats['completed'] = student_stats['completed'].astype(int)
    
    # Merge back to main dataframe
    df = df.merge(student_stats[['student_id', 'completed']], on='student_id', how='left')
    
    # Drop latent factor
    df = df.drop(columns=['_capability'])
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    output_path = os.path.join("data", "student_data.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())
