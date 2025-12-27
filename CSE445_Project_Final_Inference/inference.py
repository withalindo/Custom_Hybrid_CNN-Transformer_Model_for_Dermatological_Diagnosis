import os
import random
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms
from keras.models import load_model as load_keras_model
from PIL import Image

# --- Configuration ---
KERAS_MODEL_PATH = "best_hybrid_skin_model.h5"
PYTORCH_MODEL_PATH = "best_model.pth" 
DATA_DIR = "image_Input" 
NUM_CLASSES = 5       
IMAGES_PER_CLASS = 2    
IMG_SIZE = 224          

# --- PyTorch Model Definition (Keep this section as it is working) ---

class PyTorchModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(PyTorchModel, self).__init__()
        self.resnet = models.resnet101(weights=None) 
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# --- Loading Functions (Keep this section as it is working) ---

def load_keras_model_safe(path):
    print(f"🔬 Loading Keras model from: {path}...")
    try:
        # TensorFlow setup messages (omitted for brevity, keep them if you need them)
        model = load_keras_model(path)
        print(f"   ✅ Keras Model loaded successfully from: {path}")
        return model
    except Exception as e:
        print(f"❌ Error loading Keras model: {e}")
        return None

def load_pytorch_model_safe(path, num_classes=NUM_CLASSES):
    print(f"🔬 Loading PyTorch model from: {path}...")
    try:
        model = PyTorchModel(num_classes=num_classes)
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_state_dict['resnet.' + key] = value
            
        model.load_state_dict(new_state_dict, strict=False) 
        model.eval() 
        print(f"   ✅ PyTorch Model loaded successfully from: {path}")
        return model
    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}")
        print("   - Ensure PyTorchModel uses the correct depth and the path is correct.")
        return None

# --- Data and Preprocessing Functions (Updated Class Discovery) ---

def get_images_for_ensemble(data_dir, images_per_class=2):
    """
    Automatically finds class directories and randomly selects N images per class.
    
    Returns: A tuple (selected_images_list, class_names_list)
    """
    selected_images = []
    
    # Get all subdirectories (assumed to be class folders)
    all_class_folders = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))]
    
    class_names = sorted(all_class_folders) # Sort for consistent indexing
    
    if not class_names:
        print(f"❌ Error: No class folders found in {data_dir}. Check the directory name and structure.")
        return [], []
    
    # Optional: Check if the number of discovered classes matches NUM_CLASSES
    if len(class_names) != NUM_CLASSES:
        print(f"⚠️ Warning: Expected {NUM_CLASSES} classes, but found {len(class_names)}: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        all_images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_images:
            print(f"⚠️ No images found in class folder: {class_name}")
            continue

        # Randomly sample the required number of images
        sample_size = min(images_per_class, len(all_images))
        sample = random.sample(all_images, sample_size)
        
        for img_path in sample:
            selected_images.append({
                'path': img_path,
                'class': class_name
            })

    return selected_images, class_names

def preprocess_image(img_path):
    """Loads and preprocesses an image for both Keras and PyTorch models."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # 1. Preprocessing for Keras (TensorFlow) model (NHWC, normalized to [0, 1])
        # **Note: Update normalization if your Keras model used a different scheme!**
        img_np = np.array(img, dtype=np.float32) / 255.0 
        img_keras = np.expand_dims(img_np, axis=0)  # Add batch dimension (1, H, W, C)

        # 2. Preprocessing for PyTorch model (NCHW, standard normalization)
        preprocess_torch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_torch = preprocess_torch(img).unsqueeze(0) # Add batch dimension (1, C, H, W)

        return img_keras, img_torch
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None, None

# --- Ensemble Function ---

def ensemble_predict(keras_model, pytorch_model, img_keras, img_torch):
    """
    Makes predictions using both models and combines them using simple averaging.
    """
    
    # Keras/TensorFlow Prediction
    try:
        keras_preds = keras_model.predict(img_keras, verbose=0)
        keras_probs = keras_preds[0] # Take the batch of 1 prediction
    except Exception as e:
        print(f"Keras prediction failed: {e}")
        keras_probs = np.zeros(NUM_CLASSES)

    # PyTorch Prediction
    try:
        with torch.no_grad():
            pytorch_logits = pytorch_model(img_torch)
            pytorch_probs = torch.softmax(pytorch_logits, dim=1).cpu().numpy()[0]
    except Exception as e:
        print(f"PyTorch prediction failed: {e}")
        pytorch_probs = np.zeros(NUM_CLASSES)
    
    # Simple Averaging Ensemble 
    ensemble_probs = (keras_probs + pytorch_probs) / 2
    
    # Get the final ensemble prediction (index of the max probability)
    ensemble_prediction = np.argmax(ensemble_probs)
    
    return ensemble_prediction, ensemble_probs

# --- Main Inference Script ---

def main():
    # 1. Load Models
    keras_model = load_keras_model_safe(KERAS_MODEL_PATH)
    pytorch_model = load_pytorch_model_safe(PYTORCH_MODEL_PATH)
    
    if not (keras_model and pytorch_model):
        print("\n❌ Ensemble Prediction Aborted: One or more models failed to load.")
        return

    # 2. Define Classes and Select Images
    print(f"\nScanning for {IMAGES_PER_CLASS} images per class from {DATA_DIR}...")
    selected_images, class_names = get_images_for_ensemble(DATA_DIR, IMAGES_PER_CLASS)

    if not selected_images:
        print(f"\n❌ Could not find any images for testing. Check the {DATA_DIR} path and folder structure.")
        return

    # 3. Perform Ensemble Inference
    print(f"\n--- Starting Ensemble Inference on {len(selected_images)} images ---")
    print(f"Classes Found: {class_names}")
    
    correct_predictions = 0
    
    for i, img_data in enumerate(selected_images):
        img_path = img_data['path']
        true_class = img_data['class']
        
        img_keras, img_torch = preprocess_image(img_path)
        
        if img_keras is None:
            continue
        
        # Perform ensemble prediction
        ensemble_pred_index, ensemble_probs = ensemble_predict(
            keras_model, pytorch_model, img_keras, img_torch
        )
        
        predicted_class = class_names[ensemble_pred_index]
        is_correct = predicted_class == true_class
        
        if is_correct:
            correct_predictions += 1
            
        print(f"\n[{i+1}/{len(selected_images)}] Image: {os.path.basename(img_path)}")
        print(f"  -> True Label:   {true_class}")
        
        # --- Confidence score line removed as requested ---
        print(f"  -> Prediction:   {predicted_class} {'✅' if is_correct else '❌'}") 


    # 4. Final Summary
    accuracy = correct_predictions / len(selected_images)
    print("\n" + "="*50)
    print(f"✨ ENSEMBLE TEST COMPLETE ✨")
    print(f"Total Images Tested: {len(selected_images)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()