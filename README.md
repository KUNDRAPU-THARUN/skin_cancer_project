# Skin Cancer Detection Using Dermoscopic Image Analysis

This project is a B.Tech semester project for skin cancer classification using dermoscopic images.

## Folder structure

```text
skin_cancer_project/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── __init__.py
│   ├── organize_ham10000.py
│   ├── pipeline.py
│   ├── preprocess.py
│   └── train.py
└── venv/
```

## Dataset setup

1. Download the HAM10000 dataset.
2. Place the metadata CSV file, such as `HAM10000_metadata.csv`, in a data folder.
3. Place the image folders (for example `HAM10000_images_part_1` and `HAM10000_images_part_2`) in a dataset folder.
4. Run:

```powershell
python src/organize_ham10000.py --metadata <path-to-metadata.csv> --images <path-to-image-folder> --output data/raw
```

This creates class folders like:

- `data/raw/Actinic keratoses`
- `data/raw/Basal cell carcinoma`
- `data/raw/Benign keratosis`
- `data/raw/Dermatofibroma`
- `data/raw/Melanoma`
- `data/raw/Melanocytic nevi`
- `data/raw/Vascular lesions`

## Full training pipeline

```powershell
python src/pipeline.py --metadata <path-to-metadata.csv> --images <path-to-image-folder>
```

This does the following:

1. Organizes the HAM10000 images into class folders.
2. Applies the DullRazor preprocessing step.
3. Saves processed images into `data/processed`.
4. Trains the MobileNetV2 classifier and saves the model to `models/skin_cancer_model.keras`.

## Run the Streamlit app

```powershell
.\venv\Scripts\python -m streamlit run app.py --server.headless true --server.port 8502
```

Then open the local URL shown in the terminal.
