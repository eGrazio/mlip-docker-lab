from sklearn.svm import SVC
from sklearn import datasets
import joblib

# Load the Iris dataset
iris = datasets.load_iris()

# Create an SVM classifier
clf = SVC()

# Train the model using the iris dataset
model = clf.fit(iris.data, iris.target_names[iris.target])

# Save the trained model to the shared volume (make sure to use the correct path)
fout = 'iris_model.pkl'
joblib.dump(model, f'/app/models/{fout}')

print(f"Model training complete and saved as {fout}")

