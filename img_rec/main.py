from data_preprocessing import get_data_generators
from model_definition import create_model

# Paths to dataset
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

# Get data generators
train_generator, test_generator = get_data_generators(TRAIN_DIR, TEST_DIR)

# Create the model
model = create_model()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc*100:.2f}%')

# Save the model
model.save('biodegradable_classifier.h5')

# Prediction example
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    return "Biodegradable" if prediction < 0.5 else "Non-Biodegradable"

# Example usage
print(predict_image(model, 'water.jpg'))
