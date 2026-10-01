from math import tau
from utils.model_defs_utils import *
from scipy.optimize import linear_sum_assignment

from transformers import GPT2Tokenizer, GPT2Model,SwinModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torchvision.models import *
import torch.nn.functional as F
import torch.nn as nn
import geotorch
import torch

def load_gpt2_xl():
    model_name = "gpt2-xl"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name)
    return tokenizer, model, 1600

def load_gemma3():
    model_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model, 2560

def load_pythia_160M():
    model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-160m-deduped",
    revision="step143000",
    cache_dir="./pythia-160m-deduped/step143000",
    )

    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-160m-deduped",
    revision="step143000",
    cache_dir="./pythia-160m-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, 768

def load_pythia_410M():
    model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-410m-deduped",
    revision="step143000",
    cache_dir="./pythia-410m-deduped/step143000",
    )

    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-410m-deduped",
    revision="step143000",
    cache_dir="./pythia-410m-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, 1024


def load_pythia_1B():
    model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-1b-deduped",
    revision="step143000",
    cache_dir="./pythia-1b-deduped/step143000",
    )

    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1b-deduped",
    revision="step143000",
    cache_dir="./pythia-1b-deduped/step143000",
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer, model, 2048
    

def load_language_model(model_name):
    if model_name == 'gpt2_xl':
        return load_gpt2_xl()
    elif model_name == 'gemma3':
        return load_gemma3()
    elif model_name == 'pythia_160m':
        return load_pythia_160M()
    elif model_name == 'pythia_410m':
        return load_pythia_410M()
    elif model_name == 'pythia_1b':
        print('''Loading Pythia 1B model, this may take a while...''')
        return load_pythia_1B()
    
def get_vision_model_embeddings(model, model_name, x):
    if 'swin' in model_name:
        return model(x).pooler_output
    elif 'vit' in model_name and 'dino' not in model_name:
        return model.forward_features(x)[:, 0]
    return model(x)
    
class Disentangled_Invertible_Visual_Model(nn.Module):
    def __init__(self, model_name, transformation_type, hidden_layer=None):
        super().__init__()
        self.model_name = model_name
        self.model, self.hidden_dim = load_vision_model(model_name, hidden_layer)
        for param in self.model.parameters():
            param.requires_grad = False

        self.linear_invertible = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.transformation_type = transformation_type
        if transformation_type == 'orthogonal':
            print("Using orthogonal transformation")
            geotorch.orthogonal(self.linear_invertible, "weight")
        elif transformation_type == 'invertible':
            print("Using invertible transformation")
            geotorch.invertible(self.linear_invertible, "weight")
        elif transformation_type == 'nonlinearity_tanh':
            self.linear_invertible = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim*2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim*2, self.hidden_dim),
            )

    def forward(self, x):
        with torch.no_grad():
            outputs = get_vision_model_embeddings(self.model, self.model_name, x).reshape(x.shape[0],-1)
        outputs = self.linear_invertible(outputs)
        return outputs
    
    def reconstruct(self, x):
        return get_vision_model_embeddings(self.model, self.model_name, x).reshape(x.shape[0],-1)
    # def reconstruct(self, x):
    #     return (x-self.linear_invertible.bias)@torch.linalg.inv(self.linear_invertible.weight.T)
    

class DoublyStochasticLinear(nn.Module):
    def __init__(self, dim, n_iters=10):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.randn(dim, dim))
        self.n_iters = n_iters

    def sinkhorn(self, W):
        W = torch.exp(W)  # ensure positivity
        for _ in range(self.n_iters):
            W = W / W.sum(dim=1, keepdim=True)  # row normalize
            W = W / W.sum(dim=0, keepdim=True)  # col normalize
        return W
    
    def to_permutation_matrix(self):
        """Convert doubly stochastic matrix to permutation matrix via Hungarian algorithm."""
        W = self.sinkhorn(self.weight_raw)
        # Hungarian algorithm maximizes assignment, so negate for max
        cost = -W.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        
        perm = torch.zeros_like(W)
        perm[row_ind, col_ind] = 1.0
        return perm

    def forward(self, x):
        W = self.sinkhorn(self.weight_raw)
        return x @ W.T
    

class Disentangled_Permutation_Visual_Model(nn.Module):
    def __init__(self, model_name, transformation_type='permutation', hidden_layer=None):
        super().__init__()
        self.model_name = model_name
        self.model, self.hidden_dim = load_vision_model(model_name, hidden_layer)
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear_invertible = DoublyStochasticLinear(self.hidden_dim)

    
    def forward(self, x,):
        with torch.no_grad():
            outputs = get_vision_model_embeddings(self.model, self.model_name, x).reshape(x.shape[0],-1)
        outputs = self.linear_invertible(outputs)
        return outputs
    
    def inference_forward(self, x):
        with torch.no_grad():
            outputs = get_vision_model_embeddings(self.model, self.model_name, x).reshape(x.shape[0],-1)
        P = self.linear_invertible.to_permutation_matrix().to(outputs.device)
        outputs = outputs @ P.T
        return outputs
    

class Disentangled_Permutation_Language_Model(nn.Module):
    def __init__(self, model_name, hidden_layer, token_processing_type, alpha, transformation_type='permutation'):
        super().__init__()
        self.model_name = model_name
        self.tokenizer, self.llm, self.hidden_dim = load_language_model(model_name)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.hidden_layer = hidden_layer
        self.token_processing_type = token_processing_type
        self.linear_invertible = DoublyStochasticLinear(self.hidden_dim)

    
    def forward(self, x,):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        outputs = self.linear_invertible(outputs)
        return outputs
    
    def inference_forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        P = self.linear_invertible.to_permutation_matrix().to(outputs.device)
        P = P.to(outputs.dtype)
        outputs = outputs @ P.T
        return outputs
    
class Disentangled_Invertible_Language_Model(nn.Module):
    def __init__(self, model_name, hidden_layer, token_processing_type, alpha, transformation_type):
        super().__init__()
        self.tokenizer, self.llm, self.hidden_dim = load_language_model(model_name)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.hidden_layer = hidden_layer
        self.token_processing_type = token_processing_type

        self.linear_invertible = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.alpha = alpha

        self.transformation_type = transformation_type
        if transformation_type == 'orthogonal':
            print("Using orthogonal transformation")
            geotorch.orthogonal(self.linear_invertible, "weight")
        elif transformation_type == 'invertible':
            print("Using invertible transformation")
            geotorch.invertible(self.linear_invertible, "weight")
    
    def forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        outputs = outputs.to(self.linear_invertible.weight.dtype)
        outputs = self.linear_invertible(outputs.cuda())
        return outputs

    def llm_forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        return outputs
    
    def reconstruct(self, x):
        return (x-self.linear_invertible.bias)@torch.linalg.inv(self.linear_invertible.weight.T)

class Disentangled_Linear_Model(nn.Module):
    def __init__(self, model_name, hidden_layer, token_processing_type, alpha):
        super().__init__()
        self.tokenizer, self.llm, self.hidden_dim = load_language_model(model_name)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.hidden_layer = hidden_layer
        self.token_processing_type = token_processing_type

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_reconstruct = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.alpha = alpha
        
    
    def forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        outputs = self.linear(outputs) ## add RELU here
        reconstructed = self.linear_reconstruct(outputs)
        return outputs, reconstructed

    def llm_forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        return outputs

    
class Disentangled_Model(nn.Module):
    def __init__(self, model_name, hidden_layer, token_processing_type, alpha, transformation_type):
        super().__init__()
        self.tokenizer, self.llm, self.hidden_dim = load_language_model(model_name)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.hidden_layer = hidden_layer
        self.token_processing_type = token_processing_type
        
        self.alpha = alpha
        self.linear_reconstruct = nn.Identity()

        self.transformation_type = transformation_type
        if transformation_type == 'orthogonal':
            print("Using orthogonal transformation")
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            geotorch.orthogonal(self.linear, "weight")
        elif transformation_type == 'invertible':
            print("Using invertible transformation")
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            geotorch.invertible(self.linear, "weight")
        elif transformation_type == 'GL_differente_weights':
            print("Using linear transformation with different weights")
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear_reconstruct = nn.Linear(self.hidden_dim, self.hidden_dim)
        elif transformation_type == 'GL_same_weights':
            print("Using linear transformation with same weights")
            self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        outputs = self.linear(outputs)
        reconstructed = self.linear_reconstruct(outputs)
        return outputs, reconstructed
    
    def llm_forward(self, x):
        encoded_inputs1 = self.tokenizer(x, return_tensors='pt', padding=True)
        with torch.no_grad():
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs1.items()}
            outputs = self.llm(**encoded_inputs, output_hidden_states=True).hidden_states[self.hidden_layer]
        if self.token_processing_type == 'avg':
            outputs = outputs.mean(dim=1)
        return outputs
    
    def reconstruct(self, x):
        if self.transformation_type in ['orthogonal', 'invertible']:
            return (x-self.linear.bias)@torch.linalg.inv(self.linear.weight.T)
        elif self.transformation_type == 'GL_same_weights':
            return F.linear(x, self.linear.weight.T)
        elif self.transformation_type == 'GL_differente_weights':
            return self.linear_reconstruct(x)
        
    def transform(self, x):
        return self.linear(x)

    def l2_regularization(self):
        l2_reg = torch.tensor(0.).cuda()
        for param in self.linear.parameters():
            l2_reg += torch.norm(param)
        return self.alpha * l2_reg
    

class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
    
    def forward(self, x):
        return self.linear(x)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.).cuda()
        for param in self.parameters():
            l2_reg += torch.norm(param)
        return self.alpha * l2_reg
    
class LinearProbe(nn.Module):
    def __init__(self, in_dim=2048, num_classes=47):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)