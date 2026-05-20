import torch
from GPT_TransformersBlock import GPT_Transformers_Block
from create_dataset import dataset_download, Create_Tokenizer, DATA_PATH

def load_model(checkpoint_path: str = "checkpoints/model.pt"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    dataset_download()
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenizer = Create_Tokenizer(text)

    model = GPT_Transformers_Block(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        number_heads=config['num_heads'],
        num_layers =config['num_layers'],
        block_size=config['block_size'],
        drop_outs=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()
    print(f"Model loaded!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training iterations: {checkpoint['iteration']}")

    return model, tokenizer, device

@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_tokens: int = 500, temperature: float = 0.8):
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    generated_text = tokenizer.decode(output_ids[0])

    return generated_text

if __name__ == "__main__":
    model, tokenizer, device = load_model("checkpoints/model.pt")
    while True:
        try:
            prompt = input("\nYour prompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Farewell!")
                break

            if not prompt:
                continue

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break