import os  # Import os so environment variables (like HF_TOKEN) can be read.
import cv2  # Import OpenCV for reading and decoding video frames from uploaded or local clips.
import torch  # Import PyTorch for tensor operations and model inference.
import torch.nn as nn  # Import the neural network module helpers used to rebuild the output classifier.
import numpy as np  # Import NumPy for frame normalization, array sampling, and probability handling.

import sys  # Import sys so the script can read CLI arguments when run from a terminal.
from pathlib import Path  # Import Path for robust file and directory path handling.
from torchvision.models.video import r3d_18  # Import the 3D ResNet-18 model used for video classification.
from huggingface_hub import hf_hub_download  # Import the helper that downloads the model weights from Hugging Face.

BASE_DIR = Path(__file__).resolve().parent  # Resolve the directory containing this script so model and test paths stay stable.
MODEL_FILENAME = "robbery_r3d18_test.pth"  # Name of the weight file, same as the file uploaded to Hugging Face.
MODEL_PATH = BASE_DIR / MODEL_FILENAME  # Set the path to the trained checkpoint used for inference.
HF_REPO_ID = "Rafay1337/robbery-r3d18"  # Hugging Face repo that stores the model weights.
CLASSES = ["normal", "robbery"]  # Define the expected output labels for the binary classification task.
NUM_FRAMES = 16  # Sample a fixed number of frames from each video to create a compact, consistent input tensor.
IMAGE_SIZE = 112  # Resize each frame to 112x112 to match the model's expected input size.
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)  # Store the RGB mean used during training for normalization.
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)  # Store the RGB standard deviation used during training for normalization.
device = torch.device("cpu")  # Force inference onto CPU for maximum compatibility across machines without CUDA.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}  # Define the file types supported when scanning a directory for test videos.


def ensure_model():  # Download the model weights from Hugging Face if they are not already on disk.
    if MODEL_PATH.is_file():  # If the file already exists (e.g. on your own PC), skip the download.
        return  # Nothing to do.

    print("Model file not found locally. Downloading from Hugging Face...")  # Tell the logs that a download is starting.
    hf_hub_download(  # Download the file from the Hugging Face repo.
        repo_id=HF_REPO_ID,  # The repo that holds the weights.
        filename=MODEL_FILENAME,  # The exact file name inside the repo.
        local_dir=str(BASE_DIR),  # Save it next to this script so MODEL_PATH finds it.
        token=os.getenv("HF_TOKEN"),  # Read token from the HF_TOKEN environment variable (needed for private repos).
    )
    print("Model downloaded.")  # Confirm the download finished.


def load_model():  # Load the saved model weights and return the ready-to-use network.
    ensure_model()  # Make sure the weight file exists, downloading it first if needed.

    if not MODEL_PATH.is_file():  # Check whether the model checkpoint exists before trying to load it.
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")  # Raise a clear error if the weight file is missing.

    model = r3d_18(weights=None)  # Build the R3D-18 architecture from scratch without pretrained ImageNet weights.
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # Replace the final layer with two output neurons for normal vs robbery.

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)  # Load the checkpoint into CPU memory to avoid VRAM issues.

    if isinstance(checkpoint, dict):  # If the checkpoint is a dictionary, unwrap any wrapper keys used during saving.
        if "model_state_dict" in checkpoint:  # If the checkpoint uses a model_state_dict key, prefer it.
            checkpoint = checkpoint["model_state_dict"]  # Extract the model parameters from the saved dictionary.
        elif "state_dict" in checkpoint:  # If the checkpoint stores a generic state_dict, extract that instead.
            checkpoint = checkpoint["state_dict"]  # Extract the state dictionary.

    model.load_state_dict(checkpoint, strict=True)  # Load the saved weights into the model with strict key matching.
    model.eval()  # Put the model in evaluation mode so it behaves like inference and not training.
    model.to(device)  # Move the model to the CPU device selected at the top of the script.

    return model  # Return the loaded model ready for inference.


def preprocess_video(video_path):  # Convert a video file into a normalized 4D tensor for the model.
    cap = cv2.VideoCapture(str(video_path))  # Open the video file using OpenCV so frames can be read sequentially.

    if not cap.isOpened():  # If the file could not be opened, raise a clear error to the caller.
        raise ValueError(f"Cannot open video: {video_path}")  # Show the exact file path that failed to open.

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total frame count from the video metadata.

    if total_frames < 1:  # If the video has no frames, stop early with a readable message.
        cap.release()  # Release the video handler before exiting.
        raise ValueError(f"Video contains no readable frames: {video_path}")  # Explain that the chosen file is not valid for inference.

    indices = np.linspace(0, max(total_frames - 1, 0), NUM_FRAMES).astype(int)  # Sample evenly spaced frames across the video.
    indices = np.unique(indices)  # Remove duplicates so the model never sees repeated frame selections.

    frames = []  # Start a list to hold normalized RGB frames before stacking them together.
    for idx in indices:  # Loop through the sampled frame positions.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))  # Move the video reader to the selected frame index.
        success, frame = cap.read()  # Read the next frame and store whether it succeeded.

        if not success or frame is None:  # If the frame cannot be read, raise an informative error.
            cap.release()  # Release the OpenCV capture to avoid leaking system resources.
            raise ValueError(f"Could not read frame {idx} from {video_path}")  # Show the failed frame number and path.

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR color space into RGB to match the model's expected channel order.
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))  # Resize the image to the 112x112 input size required by the CNN.
        frame = frame.astype(np.float32) / 255.0  # Convert the pixel values from uint8 to floating point and scale to 0-1.
        frame = (frame - MEAN) / STD  # Normalize the frame using the training mean and standard deviation.
        frames.append(frame)  # Append the normalized frame to the list for later stacking.

    cap.release()  # Always release the video capture once all frames are extracted.

    if not frames:  # If no frames were captured, fail early with a clear message.
        raise ValueError(f"No video frames were extracted from: {video_path}")  # Explain that preprocessing did not yield usable data.

    video = np.stack(frames)  # Stack the individual frames into a single 3D array: frames x height x width x channels.
    video = torch.from_numpy(video).permute(3, 0, 1, 2)  # Convert the array to a PyTorch tensor and reorder to C, T, H, W format.

    return video.unsqueeze(0)  # Add a batch dimension so it matches the model's expected input shape.


@torch.inference_mode()  # Disable gradient tracking during inference to reduce memory use and speed up classification.
def predict_video(model, video_path, verbose=True):  # Run inference for a single video and optionally print the result.
    video = preprocess_video(video_path)  # Convert the video file to a normalized tensor.
    logits = model(video.to(device))  # Pass the tensor through the model to generate class scores.
    probs = torch.softmax(logits, dim=1)[0]  # Convert the raw logits to probabilities for the selected video.

    predicted_index = int(probs.argmax())  # Select the class with the largest probability.
    scores = {name: float(probs[i].item()) for i, name in enumerate(CLASSES)}  # Create a dictionary of per-class confidence values.
    predicted_label = CLASSES[predicted_index]  # Store the text label for the most likely class.
    confidence = float(probs[predicted_index].item())  # Store the top confidence value for later display in the UI.

    if verbose:  # Print the formatted report only when running from the terminal CLI.
        print("\n" + "=" * 45)  # Print a line separator before the prediction report.
        print("VIDEO:", video_path.name)  # Show the video filename that was processed.
        print("PREDICTION:", predicted_label.upper())  # Print the final predicted class in uppercase.

        for i, name in enumerate(CLASSES):  # Loop through each class to display its probability.
            print(f"{name.capitalize()}: {probs[i].item() * 100:.2f}%")  # Print the confidence score as a percentage.

        print("=" * 45)  # Print the closing separator.

    return {"predicted_label": predicted_label, "confidence": confidence, "scores": scores}  # Return a structured result for UI and programmatic use.


def find_video_files(input_path):  # Recursively find supported video files inside a folder.
    input_path = Path(input_path)  # Convert the provided input into a Path object for consistent path operations.

    if input_path.is_file():  # If the user passed a single file, return it directly.
        return [input_path]  # Put the file into a list so the rest of the script can iterate uniformly.

    if input_path.is_dir():  # If the user provided a directory, scan it recursively for videos.
        videos = []  # Create an empty list to hold all valid video files.
        for path in sorted(input_path.rglob("*")):  # Walk through the directory tree in sorted order for deterministic processing.
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in VIDEO_EXTENSIONS:  # Include only supported, visible video files.
                videos.append(path)  # Save the valid video path.
        return videos  # Return the discovered list of file paths.

    raise FileNotFoundError(f"Input path not found: {input_path}")  # Raise a clear error when the user passes an invalid location.


if __name__ == "__main__":  # Execute the CLI workflow only when the script is run directly.
    if len(sys.argv) < 2:  # If no file or folder path was given, default to the built-in test directory.
        default_input = BASE_DIR / "test"  # Set the default dataset folder relative to the project directory.
        if default_input.exists():  # Check whether the default test directory exists.
            target_paths = find_video_files(default_input)  # Discover all supported video files under the test directory.
        else:  # If the default directory is missing, show the expected usage.
            print('Usage: python predict.py "video.mp4" or "test_folder"')  # Display the command-line usage text.
            sys.exit(1)  # Exit with code 1 to signal invalid CLI usage.
    else:  # If the user supplied a path, resolve and validate it.
        user_input = Path(sys.argv[1])  # Convert the provided argument to a Path object.
        if not user_input.is_absolute():  # If the user gave a relative path, resolve it relative to this script.
            user_input = BASE_DIR / user_input  # Build an absolute path from the script directory.
        target_paths = find_video_files(user_input)  # Discover the valid video files from the provided path.

    if not target_paths:  # If no supported videos were found, raise a useful error.
        raise FileNotFoundError(f"No video files were found in: {user_input if len(sys.argv) >= 2 else BASE_DIR / 'test'}")  # Explain which path had no matching videos.

    print("Loading trained model...")  # Announce that the model is about to be loaded.
    model = load_model()  # Load the trained classifier checkpoint.
    print("Model loaded successfully!")  # Confirm that the model is ready to make predictions.

    for video_path in target_paths:  # Process each discovered video one at a time.
        predict_video(model, video_path, verbose=True)  # Run inference and print results for the CLI user.
