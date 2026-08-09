# Deep Learning-Based Shoplifting Detection System

This repository contains the code for a Deep Learning-Based Shoplifting Detection System.

## Project Structure
- `config/`: Configuration files
- `data/`: Raw and processed datasets
- `src/`: Source code for data preparation, models, training, evaluation, and inference
- `app/`: Demo application
- `outputs/`: Saved models, logs, and results
- `report/`: Project documentation

## Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`

## Running on Colab
The training pipeline (`Stage 4`) is designed to run on Google Colab with GPU.
Ensure you mount Google Drive and update the `paths.root` variable in `config/config.yaml` to point to the Drive folder.
