# Product Complaint Classification

The repository is structured as follows:

```
├── README.md
├── src
│   ├── data
|   |   |── raw
|   |   |── processed     
|   |   |   |── training
|   |   |   |── testing  
│   ├── models
│   ├── processed  
│   └── raw       
```
## Building the data and the model
From the ```/src``` directory, running ```make train-model``` will run the following sequence of scripts:

- ```data/raw/get_data.py``` -> Will download the data set from the CFPB and save to the raw data directory.
- ```preprocessing.py``` -> Will split the data into training and testing sets, writing the required cleaned documents to text files in the ```data/processed``` directory.
- ```train_conv.py``` -> Will read the processed data and train a model. The relevant serialized objects (```padded_char.pkl``` representing the input features and ```conv_model.pt``` the finalized model) will be written to ```src/outputs```.
## Models and helper scripts
The following helper scripts are also provided:

- ```utilities.py```-> Contains data loader PyTorch classes for converting the text data to tensors and defining the alphabet used for quantization.
- ```models/CLR.py``` -> Contains a PyTorch module for running cyclic learning rates to speed up the process of determining the best learning rate.
- ```models/Stopper.py``` -> Contains a PyTorch module for implementing early stopping to check against the validation data set and make sure loss is decreasing.
- ```models/CharConv.py``` -> The definition of the convolutional neural network used for character-based text classification.
