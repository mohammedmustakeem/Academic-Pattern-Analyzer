import os
import pandas as pd
import kagglehub 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Download dataset
path = kagglehub.dataset_download(
    "spscientist/students-performance-in-exams"
)

# CSV path
p = os.path.join(path, "StudentsPerformance.csv")


df = pd.read_csv(p)

# Create new feature
df['total_score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
)
df['gender']=df['gender'].replace({'male':'engineer','female':'doctor'}) inplace=True)
# Features and target
X = df[['math score', 'reading score',
        'writing score', 'total_score']]



# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, Y_train)
