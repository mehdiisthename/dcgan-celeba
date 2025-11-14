# dcgan-celeba

This repository contains a Jupyter notebook implementing a Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic face images. The model is trained on the CelebA (CelebFaces Attributes) dataset, which is a popular benchmark for face generation tasks.

## Overview

DCGAN is a type of Generative Adversarial Network (GAN) that uses convolutional layers in both the generator and discriminator networks. This architecture helps in producing high-quality images by learning to generate new samples that resemble the training data. In this project, the DCGAN is trained to generate celebrity-like face images using the CelebA dataset.

## Dataset

The CelebA dataset is a large-scale face attributes dataset consisting of:
- **202,599** celebrity face images.
- **10,177** unique identities.
- Each image annotated with **40 binary attributes** (e.g., smiling, wearing glasses, blond hair).
- **5 landmark locations** per image (e.g., eyes, nose, mouth).
- Images feature diverse poses, expressions, and background clutter, making it ideal for training generative models.

The dataset used in this project is sourced from Kaggle: [CelebA Dataset](https://www.kaggle.com/datasets/zuozhaorui/celeba). For more details on the standard CelebA dataset, refer to the original version on Kaggle: [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

**Note:** Download the dataset from the provided Kaggle link and place it in the appropriate directory (e.g., `data/`) before running the notebook. The dataset size is approximately 1.3 GB when unzipped.

## Requirements

To run the notebook, you'll need the following dependencies. It's recommended to use a virtual environment (e.g., via `venv` or `conda`).

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Key libraries (install via `pip`):
  ```
  pip install tensorflow numpy matplotlib
  ```
  (Assuming the notebook uses TensorFlow/Keras; adjust if using PyTorch instead.)

For GPU acceleration (recommended for training), ensure you have CUDA installed and compatible with TensorFlow.


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mehdiisthename/dcgan-celeba.git
   cd dcgan-celeba
   ```

2. Install dependencies:
   ```
   pip install tensorflow
   ```

3. Download the CelebA dataset from [Kaggle](https://www.kaggle.com/datasets/zuozhaorui/celeba) and extract it to a `data/` folder in the project root.

## Usage

1. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `dcgan-celeba.ipynb` in your browser.

3. Run the cells sequentially:
   - Data loading and preprocessing.
   - Model definition (Generator and Discriminator).
   - Training loop.
   - Image generation and visualization.

Training may take several hours depending on your hardware. The notebook includes checkpoints to save model weights.

To generate new images after training:
- Load the saved generator model.
- Use the provided function to sample from the latent space.

Example output: The notebook will display generated face images during training epochs.

## Results

After training, the DCGAN should produce realistic face images. Sample generated images can be found in the `images/` directory (generated during runtime). I trained the model for 20 epochs and the results were not great. For best results, train for at least 50 epochs.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, such as hyperparameter tuning or additional features.

## Acknowledgments

- The DCGAN architecture is based on the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Radford et al. (2015).
- Thanks to the creators of the CelebA dataset.
- Dataset hosted on Kaggle.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. Note that the CelebA dataset has its own usage terms; please review them on Kaggle for commercial or research use.
