After setting up the dependencies, just run "testModel.py" to run the code
This will use already trained "model.pickle" to predict
However, you can also train it on your own dataset
for that,
  -> Run "createDataset", this will take 500 pictures of you for each gesture and store them
      in a new directory named "data".
  -> Run "prepareDataset", this will extract hand landmarks from those images one-by-one and 
      store them in "data.pickel". This stage is our Data-Preprocessing.
  -> Run "trainModel.py", this will get the data from "data.pickle", train RFC Model and then
      store the trained model in "model.pickle"
  -> at Last, "Run testModel.py", this will get continuous stream of frames from webcam and 
      predict the gesture using "model.pickle" 

