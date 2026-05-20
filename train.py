import os
import time
import torch
from torchgen.api.types import voidT

from create_dataset import dataset_download,loaded_data,create_batch
from GPT_TransformersBlock import GPT_Transformers_Block

Embedding_dim =384 #embedding dim
NUM_heads=6
Num_layers = 6
Block_size = 256
dropouts = 0.1

Batch_size=64
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4
WARMUP_ITERS =100

Device = "mps" if torch.backends.mps.is_available() else "cpu"

model_point = "model_point/model.pt"

def learning_rate_created(iteration):
    #In paper lr starting small then to make them take it step by step
    if iteration < WARMUP_ITERS:
        return learning_rate * (iteration / WARMUP_ITERS)

    else:
        return  learning_rate

def evaluate_model(model,train_data,test_data):
    model.eval() #evaluate mode

    result = {} #show result in dict

    with torch.inference_mode():
        #if name of my data "train" I use for loop for train
        for name,data in [("train",train_data),("test",test_data)]:
            total_loss = 0.0

            #this is  eval area
            for _ in range(100):
                #I get my x and y  x[5x256]
                x,y=create_batch(data=data,batch_size=Batch_size,block_size=Block_size)

                x = x.to(Device)
                y = y.to(Device)

                logits,loss = model(x,y)

                total_loss += loss.item()
            result[name] = total_loss / 100

        model.train()
        return  result

@torch.no_grad()

def generate_sample(model,tokenizer):
    model.eval()

    prompts = "ROMEO: "
    prompts_ids = tokenizer.encode(prompts) # encode the prompts

    input_ids = torch.tensor(prompts_ids,dtype=torch.long,device=Device).unsqueeze(dim=0) # We take batch_size so unsqueeze

    output_ids = model.generation(input_ids,max_new_token=200,temperature=0.8)#generete answer

    model.train()

    return tokenizer.decode(output_ids[0])#we choose first decode char



def train_step():
    # I take train_data,test_data and tokenizer

    train_data,test_data,tokenizer = loaded_data()

    vocab_size = tokenizer.vocab_size

    model = GPT_Transformers_Block(
        vocab_size=vocab_size,
        embedding_dim=Embedding_dim,
        num_layers=Num_layers,
        number_heads=NUM_heads,
        block_size=Block_size
    )
    model = model.to(Device)

    #create optimizer in paper uses AdamW
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

    os.makedirs("model_point",exist_ok=True)

    start_time = time.time()

    #this is train step
    for iteration in range(max_iters):
        lr = learning_rate_created(max_iters)

        for params_groups in optimizer.param_groups: #lr is dynamic now
            params_groups["lr"] = lr

        x,y=create_batch(data=train_data,batch_size=Batch_size,block_size=Block_size) #Take x and y
        x = x.to(Device)
        y = y.to(Device)

        logits,loss = model(x,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #I print when eval_interval iterations have occurred  or iteration +1 = max_iters
        if iteration % eval_interval ==0 or iteration ==max_iters -1:
            losses = evaluate_model(model,train_data,test_data)
            elapsed = time.time() -start_time
            print(f"Iter:{iteration}")
            print(f"Train Loss:{losses['train']}")
            print(f"Test Loss:{losses['test']}")
            print(f"Learning Rate:{lr}")
            print(f"time:{elapsed}")
            if iteration > 0:
                print(generate_sample(model, tokenizer))#if iter >0 create sample

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'iteration': max_iters,
        'val_loss': losses['test'],
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': Embedding_dim,
            'num_heads': NUM_heads,
            'num_layers': Num_layers,
            'block_size': Block_size,
        }
    }
    torch.save(checkpoint, model_point)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final test loss: {losses['test']:.4f}")
    print(f"Model saved to: {model_point}")


if __name__ == "__main__":
    train_step()








