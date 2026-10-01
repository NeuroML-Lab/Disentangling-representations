import numpy as np
import torch
import os 
import pickle 
import random
from itertools import combinations
# from skimage.transform import resize
from torchvision import transforms
from PIL import Image


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Geirhos_Conflicting_Cues(torch.utils.data.Dataset):
    def __init__(self):
        self.images = np.load('inference_datasets/geirhos_conflicting_cues/images.npy')
        self.content_label = np.load('inference_datasets/geirhos_conflicting_cues/content_label.npy')
        self.style_label = np.load('inference_datasets/geirhos_conflicting_cues/style_label.npy')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.content_label) 
    
    def __getitem__(self, index):
        
        X = preprocess(Image.open(self.images[index]))
        c_label = self.content_label[index]
        s_label = self.style_label[index]
        return X, c_label, s_label
    

class Geirhos_Filled_Silhoutte(torch.utils.data.Dataset):
    def __init__(self):
        self.images = np.load('inference_datasets/geirhos_filled_silhouttes/images.npy')
        self.content_label = np.load('inference_datasets/geirhos_filled_silhouttes/content_label.npy')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.content_label) 
    
    def __getitem__(self, index):
        
        X = preprocess(Image.open(self.images[index]))
        img_name = self.images[index].rsplit("/", 1)[-1] [:-4]
        c_label = self.content_label[index]
        return X, c_label, img_name


class Geirhos_Edges(torch.utils.data.Dataset):
    def __init__(self):
        self.images = np.load('inference_datasets/geirhos_edge_cues/images.npy')
        self.content_label = np.load('inference_datasets/geirhos_edge_cues/content_label.npy')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.content_label) 
    
    def __getitem__(self, index):
        
        X = preprocess(Image.open(self.images[index]))
        img_name = self.images[index].rsplit("/", 1)[-1] [:-4]
        c_label = self.content_label[index]
        return X, c_label, img_name
    

class Stylised_COCO_Conflicting_Cues(torch.utils.data.Dataset):
    def __init__(self, splits = 'test'):
        self.contents = np.load(f'raw_training_data/mscoco_stylised/splits/content_{splits}.npy')
        self.styles = np.load(f'raw_training_data/mscoco_stylised/splits/style_{splits}.npy')
        self.content_labels, self.style_labels = [],[]
        for c in self.contents:
            for s in self.styles:
                self.content_labels.append(c[:-4])
                self.style_labels.append(s[:-4])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.content_labels) 
    
    def __getitem__(self, index):
        c_label, s_label = self.content_labels[index], self.style_labels[index]
        image_name = 'raw_training_data/mscoco_stylised/images/' + c_label + '_' + s_label + '.jpg'
        X = preprocess(Image.open(image_name))
        return X, c_label, s_label

class MNIST_content_style(torch.utils.data.Dataset):
    def __init__(self, splits = 'test'):
        content_splits = np.load('raw_training_data/mnist_stylised/splits/content_'+splits+'.npy', allow_pickle = True)
        label_splits = np.load('raw_training_data/mnist_stylised/splits/label_'+splits+'.npy', allow_pickle = True)
        style_splits = np.load('raw_training_data/mnist_stylised/splits/style_'+splits+'.npy', allow_pickle = True)
        image_path = 'raw_training_data/mnist_stylised/images/'
        self.content_labels = []
        self.style_labels = []
        self.label_list = []
        for c,l in zip(content_splits, label_splits):
            for s in style_splits:
                self.content_labels.append(c)
                self.style_labels.append(s[:-4])
                self.label_list.append(l)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.content_labels) 
    
    def __getitem__(self, index):
        c, s, l = self.content_labels[index], self.style_labels[index], self.label_list[index]
        image_name = 'raw_training_data/mnist_stylised/images/' + f"{int(c):05d}" + '_label_' + str(l) + '_' + str(s) + '.jpg'
        X = preprocess(Image.open(image_name))
        return X, c, s, l


class NSD_Dataset_visual(torch.utils.data.Dataset):

    def __init__(self, mode = 'train', brain_region = 'ventral_visual_data', splits = 'splits_0',  
                data_path = '/home/khoslalab/Documents/response-optimised-models/data/', model_name='swinb', transformation_type='invertible', unit_ratio='0.4375_0.4375', units_type='content'):
           
        self.splits = np.load('inference_datasets/NSD_image_embeddings/splits/' + splits + '.npy', allow_pickle=True).item() 
        self.ids = self.splits[mode]
        self.data_path = data_path
        with open(os.path.join(data_path,brain_region+'_1257_filtered.pickle'), 'rb') as pickle_file:
            self.stimuli_response_dict = pickle.load(pickle_file) 
        self.resp_sizes = np.load(os.path.join(data_path,brain_region+'_resp_sizes_1257_filtered.npy'))
        self.total_size = len(self.ids) 
        self.n_neurons =  sum(self.resp_sizes)
        self.pretrained_model_name = model_name
        self.unit_ratio = unit_ratio
        self.pretrained_transformation = transformation_type
        self.units_type = units_type
        data_ix = self.ids[0]
        X = np.load('inference_datasets/NSD_image_embeddings/' + self.pretrained_model_name + '/' + self.unit_ratio + '/' + self.pretrained_transformation + '/' + self.units_type + '/' + str(data_ix) + '.npy')
        self.hidden_dim = X.shape[-1]


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids) 
  
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_ix = self.ids[index]
        X = np.load('inference_datasets/NSD_image_embeddings/' + self.pretrained_model_name + '/' + self.unit_ratio + '/' + self.pretrained_transformation + '/' + self.units_type + '/' + str(data_ix) + '.npy')
        # X = np.asarray(image).astype('float32') 
        y = []
        subjects = [1,2,5,7]
        for i,sub in enumerate(subjects):
            respi = self.stimuli_response_dict[data_ix][sub]
            resp_size = self.resp_sizes[i]
            if respi is None:
                respi = np.ones(resp_size)*(-999.0) ##[-999.0 for i in range(resp_size)]
            y = np.concatenate((y, respi))
        X = torch.tensor(X)
        return data_ix, X, y
