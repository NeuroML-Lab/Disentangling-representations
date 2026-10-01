import numpy as np
import torch
import os 
import pickle 
import random
from itertools import combinations
from skimage.transform import resize
from torchvision import transforms
from PIL import Image

def get_pereira_2018_minimal_pairs(split_type):
    all_data = np.load('raw_training_data/pereira_2018/jabberwocky_nonsense/'+split_type+'.npy', allow_pickle = True)
    return all_data

def get_pereira_2018_meaningful_jabberwocky_minimal_pairs(split_type):
    all_data = np.load('raw_training_data/pereira_2018/jabberwocky_meaningful/'+split_type+'.npy', allow_pickle = True)
    return all_data

def get_pereira_2018_syn_sem_minimal_pairs(split_type):
    all_data = np.load('raw_training_data/pereira_2018/semantic_syntactic/'+split_type+'.npy', allow_pickle = True)
    return all_data

def get_msr_paraphrase_minimal_pairs(split_type):
    all_data = np.load('raw_training_data/msr_paraphrase_corpus/splits/'+split_type+'.npy', allow_pickle = True)
    return all_data

class Dataset_language(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', datasets = ['pereira_2018']):
        self.data = np.array([], dtype=object)
        if 'pereira_2018' in datasets:
            self.data = get_pereira_2018_minimal_pairs(mode)
        if 'pereira_2018_meaningful' in datasets:
            self.data = np.concatenate((self.data, get_pereira_2018_meaningful_jabberwocky_minimal_pairs(mode)))
        if 'pereira_2018_syn_sem' in datasets:
            self.data = np.concatenate((self.data, get_pereira_2018_syn_sem_minimal_pairs(mode)))
        if 'msr_paraphrase_corpus' in datasets:
            self.data = get_msr_paraphrase_minimal_pairs(mode)
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data) 
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        org = str(X[0])
        semantic = str(X[1])
        syntax = str(X[2])
        label = int(X[3])
        return org, semantic, syntax, label
    
def get_mscoco_stylized_shape_texture_minimal_pairs(split_type):
    content_splits = np.load('raw_training_data/mscoco_stylised/splits/content_'+split_type+'.npy', allow_pickle = True)
    style_splits = np.load('raw_training_data/mscoco_stylised/splits/style_'+split_type+'.npy', allow_pickle = True)
    image_path = 'raw_training_data/mscoco_stylised/images/'
    return content_splits, style_splits, image_path

def get_shuffled_mscoco_stylized_shape_texture_minimal_pairs(split_type):
    content_splits = np.load('raw_training_data/shuffled_mscoco_stylised/splits/content_'+split_type+'.npy', allow_pickle = True)
    style_splits = np.load('raw_training_data/shuffled_mscoco_stylised/splits/style_'+split_type+'.npy', allow_pickle = True)
    image_path = 'raw_training_data/shuffled_mscoco_stylised/images/'
    return content_splits, style_splits, image_path

def get_mnist_stylised_shape_texture_minimal_pairs(split_type):
    content_splits = np.load('raw_training_data/mnist_stylised/splits/content_'+split_type+'.npy', allow_pickle = True)
    label_splits = np.load('raw_training_data/mnist_stylised/splits/label_'+split_type+'.npy', allow_pickle = True)
    content_labels = {content:label for content,label in zip(content_splits, label_splits)}
    content_splits = np.array([str(k)+'.jpg' for k in content_labels])
    label_splits = np.array([int(content_labels[k]) for k in content_labels])
    style_splits = np.load('raw_training_data/mnist_stylised/splits/style_'+split_type+'.npy', allow_pickle = True)
    image_path = 'raw_training_data/mnist_stylised/images/'
    return content_splits, label_splits, style_splits, image_path

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Dataset_vision(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', datasets = ['mscoco_stylized_shape_texture'], functional_dissociations = 'content_style'):
        self.datasets = datasets[0]
        self.functional_dissociations = functional_dissociations
        self.label_splits = None
        print('datasets :: ', datasets, functional_dissociations)
        if 'mscoco_stylized_shape_texture' in datasets:
            self.content_splits, self.style_splits, self.image_paths = get_mscoco_stylized_shape_texture_minimal_pairs(mode)
        elif 'shuffled_mscoco_stylized_shape_texture' in datasets:
            self.content_splits, self.style_splits, self.image_paths = get_shuffled_mscoco_stylized_shape_texture_minimal_pairs(mode)
        elif 'mnist_stylised_shape_texture' in datasets:
            self.content_splits, self.label_splits, self.style_splits, self.image_paths = get_mnist_stylised_shape_texture_minimal_pairs(mode)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.content_splits) 
    
    def return_content_style_minimal_pairs(self, index):
        index2 = random.choice([i for i in range(len(self.content_splits)) if i != index])

        index3 = random.choice([i for i in range(len(self.style_splits))])
        index4 = random.choice([i for i in range(len(self.style_splits)) if i != index3])
        
        c1, c2 = self.content_splits[index][:-4], self.content_splits[index2][:-4]
        if 'mnist' in self.datasets:
            c1, c2 = f"{int(c1):05d}", f"{int(c2):05d}"
        s1, s2 = self.style_splits[index3][:-4], self.style_splits[index4][:-4]
        if self.label_splits is not None:
            l1, l2 = self.label_splits[index], self.label_splits[index2]
        image_names = [c1 + '_' + s1 + '.jpg', c1 + '_' + s2 + '.jpg', c2 + '_' + s1 + '.jpg', c2 + '_' + s2 + '.jpg']
        if self.label_splits is not None:
            image_names = [c1 + '_label_' + str(l1) + '_' + s1 + '.jpg', c1 + '_label_' + str(l1) + '_' + s2 + '.jpg', c2 + '_label_' + str(l2) + '_' + s1 + '.jpg', c2 + '_label_' + str(l2) + '_' + s2 + '.jpg']
        X = [preprocess(Image.open(os.path.join(self.image_paths, img))) for img in image_names]
        X = torch.stack(X, dim=0)
        content_labels = torch.tensor([int(c1), int(c1), int(c2), int(c2)])
        style_labels = torch.tensor([int(s1), int(s2), int(s1), int(s2)])
        return X, content_labels, style_labels
    
    def return_content_class_minimal_pairs(self, index):

        content1 = self.content_splits[index][:-4]
        label1 = int(self.label_splits[index])

        label2 = int(random.choice([self.label_splits[i] for i in range(len(self.label_splits)) if self.label_splits[i] != label1]))

        
        style1 = random.choice([i for i in self.style_splits])
        style2 = random.choice([i for i in self.style_splits if i != style1])
        style3 = random.choice([i for i in self.style_splits if i != style1 and i != style2])
        style4 = random.choice([i for i in self.style_splits if i != style1 and i != style2 and i != style3])

        style1, style2, style3, style4 = style1[:-4], style2[:-4], style3[:-4], style4[:-4]

        content2 = random.choice([self.content_splits[i] for i in range(len(self.content_splits)) if self.label_splits[i]==label2 ])
        content3 = random.choice([self.content_splits[i] for i in range(len(self.content_splits)) if self.label_splits[i]==label2 and self.content_splits[i]!=content2 ])

        content2, content3 = content2[:-4], content3[:-4]
        if 'mnist' in self.datasets:
            content1, content2, content3 = f"{int(content1):05d}", f"{int(content2):05d}", f"{int(content3):05d}"
        
        image_names = [f'{content1}_label_{label1}_{style1}.jpg', 
                       f'{content1}_label_{label1}_{style2}.jpg', 
                       f'{content2}_label_{label2}_{style3}.jpg', 
                       f'{content3}_label_{label2}_{style4}.jpg']
        
        X = [preprocess(Image.open(os.path.join(self.image_paths, img))) for img in image_names]
        X = torch.stack(X, dim=0)
        content_labels = torch.tensor([int(content1), int(content1), int(content2), int(content3)])
        class_labels = torch.tensor([int(label1), int(label1), int(label2), int(label2)])
        return X, content_labels, class_labels

    
    def __getitem__(self, index):
        if self.functional_dissociations == 'content_style':
            return self.return_content_style_minimal_pairs(index)
        elif self.functional_dissociations == 'content_class':
            return self.return_content_class_minimal_pairs(index)
        


        
