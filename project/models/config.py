class Config:
    def __init__(self):
        self.train_spec = "project/data/spectograms/train"
        self.train_manual = "project/data/manually_extracted_features/train"
        self.valid_spec = "project/data/spectograms/valid"
        self.valid_manual = "project/data/manually_extracted_features/valid"
        self.test_spec = "project/data/spectograms/test"
        self.test_manual = "project/data/manually_extracted_features/test"

        self.data_audio_samples_split = "project/data/data_audio_samples_split"
        self.spectograms = "project/data/spectograms"
        self.manually_extracted_features = "project/data/manually_extracted_features"

        self.RNN_best_model_weights = "project/models/model_weights/RNN_best_model.pt"
        self.CNN_best_model_weights = "project/models/model_weights/CNN_best_model.pt"
        self.Combined_best_model_weights = "project/models/model_weights/combined_best_model.pt"

        self.confusion_matrix = "project/models/confusion_matrix"
        self.loss_plots = "project/models/loss_plots"

        self.user_uplods = "project/models/user_uploads"

        self.pca_components = "project/data/pca_components"

        self.drive_url_splits = "https://drive.google.com/file/d/1DUBt-JSpWwfbGG4IE1IkjDBZ5T-5nuvd/view?usp=sharing"
        self.drive_url_spectograms = "https://drive.google.com/file/d/1BN-l01oRxh4ihzg_yjWADs-NGeOdaSEz/view?usp=sharing"
        self.drive_url_manual_feats = "https://drive.google.com/file/d/1gi5L8wk8ApA1Vxy8ggC-H0r-3Kzsm41q/view?usp=sharing"
        self.drive_url_pca = "https://drive.google.com/file/d/17R1ddMkTmOvro3s6dXr5tykoTWSsAZKb/view?usp=sharing"

        self.correlation = "project/features"