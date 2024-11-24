import numpy as np
from keras.models import load_model # type: ignore

model = load_model('my_model.keras')

print("Enter Apartment Details to Predict Rent")
a = int(input("Number of Bedrooms: "))
b = int(input("Size of the Apartment: "))
c = int(input("Zip Code of the borough: "))
d = int(input("Furnishing Status (Unfurnished = 0, Furnished = 1): "))
features = np.array([[a, b, c, d]])

predicted_rent = model.predict(features)
print("Predicted Rent = ", model.predict(features))

