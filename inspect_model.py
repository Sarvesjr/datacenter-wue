
import joblib

model = joblib.load("models/knn_wue_model.pkl")

print("MODEL TYPE")
print(type(model))

print("\nMODEL PARAMETERS")
if hasattr(model, "get_params"):
    print(model.get_params())

print("\nFEATURE COUNT")
print(getattr(model, "n_features_in_", "Not available"))

print("\nFEATURE NAMES")
print(getattr(model, "feature_names_in_", "Not stored"))

print("\nTRAINING SAMPLES")
print(getattr(model, "n_samples_fit_", "Not available"))

print("\nINTERNAL TRAINING DATA SHAPE")
if hasattr(model, "_fit_X"):
    print(model._fit_X.shape)

print("\nINTERNAL TARGET VALUES SHAPE")
if hasattr(model, "_y"):
    print(model._y.shape)

knn = model.named_steps["model"]
scaler = model.named_steps["scaler"]

print("Scaler mean:", scaler.mean_)
print("Scaler standard deviation:", scaler.scale_)

print("Training samples:", knn.n_samples_fit_)
print("Training feature matrix shape:", knn._fit_X.shape)
print("Training target shape:", knn._y.shape)

print("First training example:")
print(knn._fit_X[0])

print("First stored target:")
print(knn._y[0])