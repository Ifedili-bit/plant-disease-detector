# Create the complete Streamlit app Python file
app_code = '''
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #2E8B57;
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .healthy { color: #2E8B57; font-weight: bold; }
    .powdery { color: #FF6B35; font-weight: bold; }
    .rust { color: #8B4513; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('plant_disease_model.keras')
        return model
    except:
        try:
            model = tf.keras.models.load_model('plant_disease_model.h5')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image)
    # Convert RGBA to RGB if needed
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Class names
class_names = ['Healthy', 'Powdery', 'Rust']

# Main app
def main():
    st.markdown('<h1 class="main-header">🌿 Plant Disease Detection</h1>', unsafe_allow_html=True)
    
    st.write("""
    Upload an image of a plant leaf to detect if it's **Healthy**, has **Powdery Mildew**, or **Rust**.
    The AI model will analyze the image and provide a diagnosis with confidence scores.
    """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a leaf image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded leaf image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis")
            
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing your image..."):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image, verbose=0)[0]
                    
                    # Simulate some processing time for better UX
                    time.sleep(1)
                
                # Get results
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                # Display main result
                st.markdown("---")
                if predicted_class == "Healthy":
                    st.success(f"## ✅ HEALTHY PLANT")
                    st.write("The leaf appears to be in good condition with no signs of disease.")
                else:
                    st.error(f"## ⚠️ {predicted_class.upper()} DETECTED")
                    st.write(f"The model detected signs of **{predicted_class}** disease.")
                
                st.info(f"**Confidence: {confidence:.2f}%**")
                
                # Show confidence scores for all classes
                st.subheader("📊 Confidence Scores")
                for i, (class_name, score) in enumerate(zip(class_names, predictions)):
                    confidence_percent = score * 100
                    color_class = class_name.lower()
                    
                    st.markdown(f"<span class='{color_class}'>{class_name}:</span>", unsafe_allow_html=True)
                    st.progress(float(score))
                    st.write(f"{confidence_percent:.2f}%")
                
                # Recommendations based on diagnosis
                st.subheader("💡 Recommendations")
                if predicted_class == "Healthy":
                    st.write("""
                    - Continue with your current plant care routine
                    - Monitor plants regularly for early detection
                    - Maintain proper watering and sunlight
                    """)
                elif predicted_class == "Powdery":
                    st.write("""
                    - Remove affected leaves if possible
                    - Apply fungicide specifically for powdery mildew
                    - Improve air circulation around plants
                    - Avoid overhead watering
                    """)
                else:  # Rust
                    st.write("""
                    - Remove and destroy infected leaves
                    - Apply copper-based fungicide
                    - Ensure proper plant spacing for air flow
                    - Avoid wetting foliage when watering
                    """)

if __name__ == "__main__":
    main()
'''

# Write the app code to a file
with open('app.py', 'w') as f:
    f.write(app_code)

print("✅ Streamlit app file 'app.py' created successfully!")
print("📁 Files in current directory:")
for file in os.listdir('/kaggle/working'):
    if file.endswith('.py') or 'plant_disease' in file:
        print(f"✓ {file}")