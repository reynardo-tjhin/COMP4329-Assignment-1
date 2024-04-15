import numpy as np
import json
import zipfile
import os
import shutil

from layer.linear_layer import Linear
from layer.activation_layer import Activation
from layer.dropout_layer import Dropout
from layer.batch_norm_layer import BatchNorm

class MLP:

    def __init__(self, layers) -> None:
        self.layers = layers
        self.training = True
        
        # for preprocessing: saving them for loading the model for testing accuracy
        self.preprocessing_type = None
        self.min = None
        self.max = None
        self.std = None
        self.mean = None

    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input, self.training)
            input = output
        return output
    
    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
    
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = np.argmax(self.forward(x[i]))
        return output
    
    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False
    

    # ===========================================
    # =         miscellaneous operations        =
    # ===========================================
    def save_preprocessing_parameters(
            self,
            type: str = None,
            min: np.ndarray = None,
            max: np.ndarray = None,
            mean: np.ndarray = None,
            std: np.ndarray = None) -> None:
        """
        Save the hyperparameters so that preprocessing can be done to load the model.
        """
        self.preprocessing_type = type
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std
    
    def save(self, name) -> None:
        """
        Save the model to the current directory where python is running.

        1. Save the model's layers and its init values in json file
        2. Save the model's parameters in npy file
        """
        # step 0: create a directory to save the temporary files
        if (not os.path.exists("./temp")):
            os.makedirs("./temp")

        # step 1: create a dictionary to store its type and corresponding string for identification
        dict_type = {
            Linear: 'Linear',
            Activation: 'Activation',
            Dropout: 'Dropout',
            BatchNorm: 'BatchNorm',
        }
        layer_parameters = {
            Linear: ['n_in', 'n_out'],
            Activation: ['activation_fn_str'],
            Dropout: ['probability'],
            BatchNorm: ['num_features', 'affine'],
        }

        # step 2.1: store the miscallaneous data
        filenames = []
        model = {
            'layers': [],
            'preprocessing_type': self.preprocessing_type,
        }
        if (self.preprocessing_type == 'normalization'):
            np.save(f'./temp/min.npy', self.min)
            np.save(f'./temp/max.npy', self.max)
            filenames.append(f'./min.npy')
            filenames.append(f'./max.npy')
        elif (self.preprocessing_type == 'standardization'):
            np.save(f'./temp/std.npy', self.std)
            np.save(f'./temp/mean.npy', self.mean)
            filenames.append(f'./std.npy')
            filenames.append(f'./mean.npy')
        
        # step 2.2: store the layers data
        for i, layer in enumerate(self.layers):
            
            # get the layer's details
            layer_info = vars(layer)
            
            # save it to a '.json' file
            new_dict = {}
            id_item = "layer_" + str(i)
            new_dict['id'] = id_item
            new_dict['name'] = dict_type[type(layer)]
            new_dict['parameters'] = {}
            parameters = layer_parameters[type(layer)]
            for parameter in parameters: # only save the init parameters!
                new_dict['parameters'][parameter] = layer_info[parameter]
            model['layers'].append(new_dict)
        
            # step 3: for each layer with learnable parameters, save the parameters in npy file
            if (type(layer) == Linear):
                np.save(f'./temp/{id_item}_W.npy', layer.W)
                np.save(f'./temp/{id_item}_b.npy', layer.b)
                filenames.append(f'./{id_item}_W.npy')
                filenames.append(f'./{id_item}_b.npy')

        # step 4: save the dictionary as a .json file
        # reference: https://realpython.com/python-json/
        with open("./temp/model_layers_info.json", "w") as write_file:
            json.dump(model, write_file, indent=4)
        filenames.append("model_layers_info.json")

        # step 5: tar/zip the file
        # reference: https://stackoverflow.com/questions/47438424/python-zip-compress-multiple-files
        compression = zipfile.ZIP_DEFLATED
        zf = zipfile.ZipFile(f"{name}.zip", mode="w")
        try:
            for file_name in filenames:
                zf.write(f"./temp/{file_name}", file_name, compress_type=compression)
        except FileNotFoundError:
            print("An error occurred")
        finally:
            zf.close()

        # step 6: remove any temporary files
        shutil.rmtree("./temp")
        
        print("model saved!")

    def load(self, path: str):
        """
        Loads the saved model. Create a new model.
        """
        # step 0: create a directory to store the temporary files
        if (not os.path.exists("./temp")):
            os.makedirs("./temp")

        # step 1: unzip the file
        # reference: https://stackoverflow.com/questions/3451111/unzipping-files-in-python
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall("./temp")

        # step 2: get the .json file
        # reference: https://realpython.com/python-json/
        with open("./temp/model_layers_info.json", "r") as read_file:
            data = json.load(read_file)
        
        # step 3: create the model based on the .json file
        layers = []
        for layer in data['layers']:
            # create the layer and append it
            if (layer['name'] == 'Linear'):
                layers.append(Linear(
                    n_in=layer['parameters']['n_in'],
                    n_out=layer['parameters']['n_out'],
                    W=np.load(f"./temp/{layer['id']}_W.npy"),
                    b=np.load(f"./temp/{layer['id']}_b.npy"),
                ))
            elif (layer['name'] == 'Activation'):
                layers.append(Activation(
                    activation_fn=layer['parameters']['activation_fn_str'],
                ))
            elif (layer['name'] == 'Dropout'):
                layers.append(Dropout(
                    probability=layer['parameters']['probability'],
                ))
            elif (layer['name'] == 'BatchNorm'):
                layers.append(BatchNorm(
                    num_features=layer['parameters']['num_features'],
                    affine=layer['parameters']['affine'],
                ))

        # create the model
        model = MLP(layers=layers)
        
        # for preprocessing
        model.preprocessing_type = data['preprocessing_type']
        if (model.preprocessing_type == "normalization"):
            model.min = np.load("./temp/min.npy")
            model.max = np.load("./temp/max.npy")
        
        elif (model.preprocessing_type == "standardization"):
            model.std = np.load("./temp/std.npy")
            model.mean = np.load("./temp/mean.npy")

        # step 4: remove the temp directory
        shutil.rmtree("./temp")

        print("model loaded!")
        return model


# testing
if (__name__ == "__main__"):

    # testing on saving and loading
    model = MLP(
        layers=[
            Linear(5, 4),
            BatchNorm(4),
            Activation('relu'),
            Dropout(0.3),
            Linear(4, 3),
            Activation('relu'),
        ]
    )
    model.save("model")
    model.load("./model.zip")