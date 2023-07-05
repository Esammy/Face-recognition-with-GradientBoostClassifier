# Face Recognition System
 - Used MTCNN for face detection
 - FaceNet Model is used to create a face embedding for each detected face.
 - Developed a Linear Support Vector Machine for face classification

# Pre-processing
 - Face detection using MTCNN
 - Detect face using and store it as dataset.npz
 - Create face-embedding using Facenet_Keras model for all the images from dataset.npz and store it in embeddings.npz
# Dataset
 - The dataset (photos) used are placed in the dataset folder. splittted into train and val for training photos and validation photos respectively.
## Getting Started

To begin using the face recognition code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirement.txt
   ```

3. Run the application:
   ```bash
   python faceembed.py
   ```
   ```bash
   python classify.py
   ```

## Contributing

We welcome contributions from the community! If you'd like to contribute to the Off-Campus Accommodation Portal, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your forked repository.
4. Submit a pull request describing your changes.

## Contact

If you have any questions, suggestions, or feedback, please feel free to reach out to our team at [egwusamuel2015@gmail.com](mailto:egwusamuel2015@gmail.com).

Thank you for choosing the Off-Campus Accommodation Portal! We hope it provides a valuable service for students seeking off-campus accommodation.
