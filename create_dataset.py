import os
import urllib.request
from traceback import print_tb

import torch

dataset_url = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt" #My text data
DATA_PATH = "data/shakespeare.txt"

def dataset_download():#I'm getting dataset with urllib.request
    if os.path.exists(DATA_PATH):
        print("Data is exist")
        return

    os.makedirs(name="data",exist_ok=True)

    urllib.request.urlretrieve(dataset_url,filename=DATA_PATH)

class Create_Tokenizer:
    def __init__(self,text:str):

        self.character_text = sorted(list(set(text))) #firstly I get unique char with set  , then I listed char, then sorted
        #print(self.character_text) #unique char

        self.vocab_size = len(self.character_text)
        #print(self.vocab_size) #size of char

        self.char_to_id = {} # "a" -->1
        self.id_to_char={} #1 --> "a"

        for index,char in enumerate(self.character_text):
            self.char_to_id[char] = index # \n': 0, ' ': 1, '!': 2, '$': 3,
            self.id_to_char[index]=char # 0: '\n', 1: ' ', 2: '!', 3:

        #print(self.char_to_id)
        #print(self.id_to_char)

    def encode(self, text_char_to_id): #Encode can=[41, 39, 52]
        ids=[]
        for char in text_char_to_id:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self,id_to_text_char):#Decode  [41, 39, 52] =can
        char_to_ids = []
        if isinstance(id_to_text_char,torch.Tensor):
            id_to_text_char = id_to_text_char.tolist()

        for ids in id_to_text_char:
            char_to_ids.append(self.id_to_char[ids])
        return ''.join(char_to_ids)

def loaded_data(data_split:float=0.9):
    with open(DATA_PATH,"r",encoding="utf-8") as file:
        text_file=file.read()

    tokenizer = Create_Tokenizer(text=text_file)

    ids_tokenizers = tokenizer.encode(text_file)# encode all token in my dataset
    data = torch.tensor(ids_tokenizers,dtype=torch.long) #convert to Tensor

    index_split = int(data_split*len(data))#out data 1115394 we get 1003854 split data

    train_split_data = data[:index_split] #torch.Size([1003854])
    test_split_data = data[index_split:] #torch.Size([111540])

    #print(test_split_data.size())
    #print(train_split_data.size())
    return train_split_data,test_split_data,tokenizer


def create_batch(data:torch.Tensor,batch_size:int,block_size:int):

    # safety buffer to guarantee a full sequence window (block_size + 1) can be sliced without exceeding data limits
    max_start =len(data) - block_size -1

    positon = torch.randint(low=0,high=max_start,size=(batch_size,))#we are choosing random 256 size for block size in data

    x_list = []
    y_list = []

    for pos in positon:
        x_list.append(data[pos : pos + block_size ]) # position + 256(block_size)
        y_list.append(data[pos+1 : pos + block_size +1]) # target always predicts the next one.

    #print(x_list) #5 tensor 256 size
    #print(y_list) #5 tensor 256 size

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    #print(x.shape)  # 1 tensor ,size of tensor [5,256]
   # print(y.shape) # 1 tensor ,size of tensor [5,256]

    return x , y


















if __name__ =="__main__":
    train_data, test_data,tokenizer = loaded_data()

    create_batch(data=train_data,batch_size=5,block_size=256)



