# Breast Cancer Histopathology Image Classification

This project uses deep learning to classify breast cancer histopathology images using the BreakHis dataset.

## Setting up the Project

### 1. Clone the Repository
```bash
git clone https://github.com/taonaben/Breast-Cancer-Histopathology-Image-Classification.git
cd Breast-Cancer-Histopathology-Image-Classification
```

### 2. Download and Setup the BreakHis Dataset
1. Download the BreakHis dataset from [BreaKHis: Breast Cancer Histopathological Image Classification](https://www.kaggle.com/datasets/ambarish/breakhis)
2. Extract the downloaded dataset
3. Place the dataset files in the project structure as follows:
   ```
   data/
   └── raw/
       └── archive/
           ├── Folds.csv
           └── BreaKHis_v1/
               └── histology_slides/
                   └── breast/
                       ├── benign/
                       │   └── SOB/
                       │       ├── adenosis/
                       │       ├── fibroadenoma/
                       │       ├── phyllodes_tumor/
                       │       └── tubular_adenoma/
                       └── malignant/
                           └── SOB/
                               ├── ductal_carcinoma/
                               ├── lobular_carcinoma/
                               ├── mucinous_carcinoma/
                               └── papillary_carcinoma/
   ```

**Note**: The dataset is not included in this repository due to size constraints and licensing requirements. Please ensure you follow the dataset's terms of use and licensing when downloading.

