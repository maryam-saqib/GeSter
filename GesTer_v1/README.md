### GesTer Version 1

This repository contains code for a GesTer - version 1. Follow the steps below to set up and use the system:

#### Setting up Dependencies
After setting up the dependencies, follow the steps below to run the code.

#### Running the Code

1. **Testing the Pre-trained Model**
   - Just run `testModel.py` to run the code.
   - This script uses the already trained `model.pickle` to predict gestures.

2. **Training on Your Own Dataset**
   If you want to train the model on your own dataset, follow these steps:

   - **Step 1: Create Dataset**
     - Run `createDataset.py`. This script will capture 500 pictures of you for each gesture and store them in a new directory named `data`.

   - **Step 2: Prepare Dataset**
     - Run `prepareDataset.py`. This script will extract hand landmarks from the images one-by-one and store them in `data.pickle`. This stage is our data preprocessing.

   - **Step 3: Train the Model**
     - Run `trainModel.py`. This script will retrieve the data from `data.pickle`, train the Random Forest Classifier (RFC) model, and then store the trained model in `model.pickle`.

   - **Step 4: Test the Trained Model**
     - Finally, run `testModel.py`. This script will capture a continuous stream of frames from the webcam and predict the gesture using the trained `model.pickle`.

