# PyTorch Training Pipeline (Work in Progress)

## Overview

This project is a comprehensive PyTorch-based training pipeline designed for **multiclass** and **multilabel** classification tasks. It is integrated with [Weights & Biases (W&B)](https://wandb.ai/site) for streamlined experimentation and monitoring of your machine learning models. Whether you're new to machine learning or an experienced practitioner, this pipeline simplifies the training process and provides flexibility for customizing your experiments.

## Features

- **Easy Setup**: You can get started with this training pipeline quickly. Simply follow the installation instructions provided below.

- **Customizable Architecture**: Configure your network architecture by editing the `architecture_template.yaml` file in the config folder. You can select an architecture and adjust the number of output features and job types to match your specific classification problem.

- **CLIP as zero-shot | fine-tune**: Use CLIP models to create zero-shot classifiers and fine-tune in your custom dataset.

- **Data Configuration**: Specify the path to your dataset and the associated CSV files in the `data_template.yaml` file. These CSV files should contain image paths relative to the dataset path and their corresponding numerical labels.

- **Hyperparameter Tuning**: Customize training and evaluation hyperparameters in the `hyps_template.yaml` file to fine-tune your model's performance.

- **W&B Integration**: Optionally, configure your Weights & Biases project by editing the `wandb_template.yaml` file. This enables you to monitor and visualize your experiments online.

- **Training**: Train your model using the provided `train.py` script. You can specify various training parameters, including the number of epochs, batch size, and more.

- **Resume Training**: Easily resume training from a previous checkpoint with the `--resume` flag.

- **Metrics and Evaluation**: Track a range of metrics, including accuracy, precision, recall, F1 score, and more. Choose the metrics that matter most to you.

- **Device Agnostic**: Train your model on either GPU (`cuda`) or CPU (`cpu`) by specifying the `--device` option.

- **Reproducibility**: Efforts are made to ensure reproducibility across different runs.

- **Contributions Welcome**: This project is open for contributions. Feel free to contribute to its development.

## Installation

To get started, follow these steps:

1. Create a conda environment:

   ```bash
   conda create --name torch_pipe pip python==3.10.6
   conda activate torch_pipe
   ```

2. Navigate to the project folder and install the required packages:

    ```bash
    cd torch_classification
    pip install -r requirements.txt
    ```

## Usage

### Train 

- Configure your network architecture, data settings, and hyperparameters in the respective YAML files.

    - The `csv files` should be in the following format:

        >image path relative to dataset path, numerical label

		<table>
		<tr><th>Multiclass </th><th>Multilabel </th></tr>
		<tr><td>

        |image_name| label |
        |-----------------------|------------|
        |train\class1\img1.png |0 |
        |train\class1\img2.png |0 |
        |train\class2\img1.png |1 |
        |... |... |

		</td><td>

		|image_name| label_1 | label_2 | label_3 | ...|
        |----------|---------|---------|----------|---|
        |train\class1\img1.png |0 |1 |0 |...|
        |train\class1\img2.png |0 |1 |1 |...|
        |train\class2\img1.png |1 |1 |0 |...|
        |...                    |...|...|...|...|

		</td></tr> </table>

    - here is an example of dataset folder structure:

	```plaintext 
	├── dataset
	│   ├── train
	│   │   ├── class1
	│   │   │   ├── image1.png
	│   │   │   ├── image2.png
	│   │   │   └── ...
	│   │   ├── class2
	│   │   │   ├── image1.png
	│   │   │   ├── image2.png
	│   │   │   └── ...
  	│   │   ...
	│   ├── val
 	│   │   ├── class1
	│   │   │   ├── image1.png
	│   │   │   ├── image2.png
	│   │   │   └── ...
 	│   │   ...
	```

- Start training your model using the provided script (example):

	```bash
	python train.py --data data --arch architecture --hyps hyps_none
	```

Refer to the parameter descriptions within the script for more details on each option.

| Parameter          | Description                                            |
| ------------------ | ------------------------------------------------------ |
| --hyps              | .yaml hyperparameter filename                         |
| --arch              | .yaml architecture filename                            |
| --data              | .yaml data filename                                     |
| --mode              | Training mode, online or offline                        |
| --device            | Device to use for training                               |
| --epochs            | Number of epochs to train                                |
| --batch_size        | Total batch size for all GPUs                            |
| --workers           | Max dataloader workers                                   |
| --patience          | How long to wait after the last time validation metric improved |
| --save_weights      | Save weights locally                                    |
| --logging           | Logging level (CRITICAL: 50, ERROR: 40, WARNING: 30, INFO: 20, DEBUG: 10, NOTSET: 0) |
| --weight_decay      | Optimizer weight decay, leave unchanged if set in the hyps file |
| --momentum          | Optimizer momentum, leave unchanged if set in the hyps file |
| --max_lr            | Optimizer max_lr, leave unchanged if set in the hyps file |
| --min_lr            | Optimizer min_lr, leave unchanged if set in the hyps file |
| --fast_mode         | If True, does not upload dataset image files when creating the error analysis table |
| --resume            | Resume last training                                   |
| --resume_info       | Resume training information                            |
| --top_k             | K for top_k precision (e.g., if 2, will use the top 2 confidence classes to compute confusion) |
| --num_samples       | Number of samples to calculate precision@K             |
| --balance_weights   | Set custom weights for loss calculation based on class imbalance |
| --warmup_epochs     | Warming up epochs for model fine-tune |
| --warmup_lr         | Optimizer learning rate for warming up |
| --force_clip		  | Force clip pre-processor (for clip models only) |

### Zero-shot classification

- Configure your network architecture and class settings (zero_shot_template), in the respective YAML files.

- Run the provided script (example)

	```bash
	python zero_shot --clip_path ViT-L-14-336px.pt 
	```

| Parameter          | Description                      |
| ------------------ | -------------------------------- |
| --clip_path        | Pre-trained clip model path      |
| --name             | Model name                       |
| --yaml_path        | .yaml templates file path        |
| --freeze_encoder   | Freeze encoder weights (clip)    |
| --device           | Device to use for training       |

## Project Status and Contributions

<details open>
<summary>Work checklist</summary><br>  

- [X] Project Core Documentation

- [X] Dataloader
    - [X] CSV
    - [ ] Bounding boxes

- [X] Albumentations

- [X] Optimizer
	- [X] Adam
	- [X] SGD
	- [X] RMSprop

- [X] lr scheduler
	- [X] ReduceLROnPlateau
	- [X] CosineAnnealingLR
	- [X] OneCycleLR

- [X] Balance weights

- [X] Early stopping

- [X] Criterion
	- [X] CrossEntropyLoss
	- [X] BCEWithLogitsLoss

- [X] Architectures
	- [X] resnet ::
	- [X] efficientnet ::
	- [X] clip encoder ::

- [X] Torchmetrics
	- [X] Confusion Matrix
		- [X] Accuracy
		- [X] Precision
		- [X] Recall
		- [X] F1 score
	- [X] Average precision
	- [X] Precision@K

- [X] Other metrics
	- [X] Precision top k

- [X] WandB
    - [X] Error analysis table

- [X] Train
    - [X] Offline
	- [X] WandB integration
	- [X] Save model
	- [X] Choose metric for lr scheduler, 
		  model performance tracking and early stopping

- [X] Eval
    - [X] Offline
	- [X] WandB integration

- [X] Resume training
	- [X] Offline
	- [X] WandB integration

- [X] Job type
    - [X] Multiclass
    - [X] Multilabel

- [ ] Logging
	- [X] Offline
	- [X] WandB integration
		- [X] Metrics curves
		- [X] Plot confusion matrix
		- [ ] Plot metrics bar
    
- [X] Warm up

- [X] Device agnostic

- [ ] Reproducibility

- [ ] Alerts

</details>

---

## Version

This is version 1.0 of the project.
