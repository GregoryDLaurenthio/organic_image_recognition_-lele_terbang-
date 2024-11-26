# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc*100:.2f}%')

# Prediction example
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

prediction = model.predict(img_array)
if prediction < 0.5:
    print("Biodegradable")
else:
    print("Non-Biodegradable")