# D-FINE: Inference on Object Detection Models

This repository contains the code to run inference using the D-FINE models, which are pre-trained on the Objects365 dataset and further fine-tuned on the COCO dataset.

## Getting Started

### 1. Clone the Repository

First, clone the repository using the following command:

```bash
git clone git@github.com:floxrobotics/D-FINE.git
```

Navigate into the cloned repository:

```bash
cd D-FINE
```

### 2. Install Basic Dependencies

#### Option 1: Using Conda

Create a new Conda environment with Python 3.11.9 and install the necessary dependencies:

```bash
conda create -n dfine python=3.11.9
conda activate dfine
pip install -r requirements.txt
```

#### Option 2: Using Python Virtual Environment (venv)

Alternatively, you can use Python's built-in `venv` to create a virtual environment:

```bash
python3 -m venv dfine_env
source dfine_env/bin/activate
pip install -r requirements.txt
```

#### Option 3: Without Virtual Environment

You can also install the dependencies directly without using any virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Install Inference Requirements

Once the basic dependencies are installed, install the additional requirements for inference:

```bash
pip install -r tools/inference/requirements.txt
```

### 4. Download Pre-trained Models

The best performing D-FINE models, which were pre-trained on the Objects365 dataset and fine-tuned on COCO, can be downloaded from the following links:

- **D-FINE-S**: [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth)
- **D-FINE-M**: [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth)
- **D-FINE-L**: [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco.pth)
- **D-FINE-X**: [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth)

### 5. Store Models in the Repository

Place the downloaded models in the `models` folder of the repository.

### 6. Prepare Videos for Inference

If you plan to run inference on video files, store the `.mp4` videos in the `test_videos` folder.

### 7. Run Inference on Video

To perform inference on a video file using the x model, use the following command in the main repository folder:

```bash
python tools/inference/torch_inf_video.py -c configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml -r models/dfine_x_obj2coco.pth --device cuda:0 --video test_videos/dock_V.MP4
```

Replace the config file and -r argument in the command accordingly to use the smaller models instead.

### 8. Run Inference on Webcam Footage

To perform inference using your webcam, use this command:

```bash
python tools/inference/torch_inf_video.py -c configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml -r models/dfine_x_obj2coco.pth --device cuda:0
```

Replace the config file and -r argument in the command accordingly to use the smaller models instead.

---

With these steps, you should be able to run inference on both videos and webcam footage using the D-FINE models.
