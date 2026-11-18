import tempfile  # Import tempfile so uploaded videos can be stored as temporary files for inference.
from pathlib import Path  # Import Path to handle uploaded file names and temporary file names cleanly.

import streamlit as st  # Import Streamlit for the interactive upload and results dashboard.

from predict import CLASSES, load_model, predict_video  # Import the reusable model-loading and prediction functions from the prediction module.

# st.set_page_config(page_title="Robbery Detection", page_icon=":material/video_camera:", layout="centered")  # Configure the Streamlit page metadata and layout for a clean upload experience.
st.set_page_config(
    page_title="Robbery Detection",
    page_icon="📹",
    layout="centered"
)

@st.cache_resource  # Cache the model object so it is loaded once and reused across reruns instead of on every upload.
def get_model():  # Return a cached shared model instance for the app.
    return load_model()  # Load the trained model from disk once and reuse it.


st.title("Videos robbery detection")  # Show the main page title for the app.
st.caption("Upload a video, run the robbery detector, and view the prediction confidence score.")  # Add a short description so the user knows what the uploader does.

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "wmv", "webm"], accept_multiple_files=False)  # Let the user upload a local video clip with supported extensions.

if uploaded_file is not None:  # Only run inference when the user actually selects a file.
    suffix = Path(uploaded_file.name).suffix or ".mp4"  # Preserve the file extension so OpenCV can read the uploaded video correctly.
    temp_path = None  # Initialize a variable to hold the temporary file path.

    try:  # Use a try block so the temporary uploaded file is cleaned up after prediction.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:  # Write the uploaded bytes to a temporary file for inference.
            temp_file.write(uploaded_file.getvalue())  # Save the uploaded video bytes to disk.
            temp_path = Path(temp_file.name)  # Store the file path for preprocessing and display.

        st.video(str(temp_path))  # Render the uploaded video in the app so the user can confirm the clip they selected.

        with st.spinner("Processing video..."):  # Show a loading indicator while the model analyzes the clip.
            model = get_model()  # Retrieve the cached model instance.
            result = predict_video(model, temp_path, verbose=False)  # Run inference without printing terminal output, returning structured results.

        predicted_label = result["predicted_label"]  # Extract the model's predicted label from the result dictionary.
        confidence = float(result["confidence"])  # Extract the confidence probability for the returned class.

        st.success(f"Prediction: {predicted_label.upper()}")  # Display the final predicted class as a success message.
        st.metric("Accuracy score / confidence", f"{confidence * 100:.2f}%")  # Show the model confidence as the score value the user wants.

        st.subheader("Class probabilities")  # Add a heading for the probability breakdown.
        for name in CLASSES:  # Loop through every class to show the score in a readable format.
            score = result["scores"][name]  # Read the associated probability for each class.
            st.markdown(f"**{name.title()}:** {score * 100:.2f}%")  # Show the percentage in a simple, scalable format.
    finally:  # Always delete the temporary file after the prediction is completed.
        if temp_path is not None and temp_path.exists():  # Ensure the temporary file exists before removing it.
            temp_path.unlink()  # Clean up the uploaded video file to avoid leaving temp artifacts behind.
