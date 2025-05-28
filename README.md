# Applied ML Group 7 🛠️

**Welcome to this Applied Machine Learning!** This code is designed to classify a vechile type out of 8 classes based on a .wav audio file.

## Prerequisites
Make sure you have the following software and tools installed:

- **PyCharm**: We recommend using VsCOde as your IDE, since it offers a highly tailored experience for Python development. 

## Getting Started
### Setting up your own repository
1. Clone this repostiry by running in your terminal :
   ```bash
    git clone https://github.com/alex86590212/Applied-ML-Group-7.git
   ```
2. Install the requirements :
   ```bash
   pip install -r requirements.txt
   ```
3. After cloning the reposity run this command to make sure you are in the right directory :
   ```bash
   cd Applied-ML-Group-7
   ```
4. After this run this command to acceess the API :
    ```bash
    uvicorn main_api:app --reload
    ```
5. Click the API that opened on a local port in the terminal
6. Go to the directory on the opened google tab and add "/docs":
   example : http://000.0.0.0:8000 and add /docs so the final result is : http://000.0.0.0:8000/docs
7. Press the try it out button
8. Where you see the string text enter one of the following to choose which model you want to use to predict : 'CNN', 'RNN', 'Combined'
   make sure you write them exactly as stated here otherwise it will return an error
9. Submit a .wav file and see the result ! Mind the fact that we only accept .wav files at the moment and if you sumbit any other file type it will return an error.

# Training Models Locally

After you’ve processed your data, follow these steps to train any of the three models on your machine:

1. **Configure `main.py`**

   Open `main.py` and set the following variables:
   ```python
   run_local = True           # enable local execution
   TRAIN_FROM_SCRATCH = False # only skip preprocessing if already done
   NUMBER = <model_number>    # 1: RNN, 2: CNN, 3: Combined Model
   MODE = "train"             # switch to training mode

