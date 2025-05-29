class Config:
    def __init__(self):
        self.train_spec = "Applied-ML-Group-7/project_name/data/spectograms/train"
        self.train_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/train"
        self.valid_spec = "Applied-ML-Group-7/project_name/data/spectograms/valid"
        self.valid_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/valid"
        self.test_spec = "Applied-ML-Group-7/project_name/data/spectograms/test"
        self.test_manual = "Applied-ML-Group-7/project_name/data/manually_extracted_features/test"

        self.data_audio_samples_split = "Applied-ML-Group-7/project_name/data/data_audio_samples_split"
        self.spectograms = "Applied-ML-Group-7/project_name/data/spectograms"
        self.manually_extracted_features = "Applied-ML-Group-7/project_name/data/manually_extracted_features"

        self.RNN_best_model_weights = "project_name/models/model_weights/RNN_best_model.pt"
        self.CNN_best_model_weights = "project_name/models/model_weights/CNN_best_model.pt"
        self.Combined_best_model_weights = "project_name/models/model_weights/combined_best_model.pt"

        self.confusion_matrix = "Applied-ML-Group-7/project_name/models/confusion_matrix"
        self.loss_plots = "Applied-ML-Group-7/project_name/models/loss_plots"

        self.user_uplods = "Applied-ML-Group-7/project_name/models/user_uploads"

        self.pca_components = "project_name/data/pca_components"

        self.drive_url_splits = "https://drive.google.com/file/d/1i-y2OAkiwXbvK3gz-Hq1c1quWCf2fsrJ/view?usp=sharing"
        self.drive_url_spectograms = "https://drive.google.com/file/d/11m2-N52yyOsVt2c6O37LvxRtFoEGmnyc/view?usp=sharing"
        self.drive_url_manual_feats = "https://drive.google.com/file/d/1Zxml-L_ezhXNzRZRszxk7-27GFSrDcp4/view?usp=sharing"
        self.drive_url_pca = "https://drive.google.com/file/d/1mFayQkcY1TP9qfvOhAM66rz1EOpq9Fc3/view?usp=sharing"

        self.correlation = "Applied-ML-Group-7/project_name/features"