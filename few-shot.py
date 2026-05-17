import torch
from torch import cuda
from torch.cpu import is_available
from transformers import AutoModelForCausalLM, AutoTokenizer


    
def ask(prompt: str, max_tokens: int=5, debug: bool=False) -> str:
    messages = [
        {"role": "user",
         "content": prompt}
    ]

    # text = tokenizer.apply_chat_template(messages,
    #                               tokenize=False,
    #                               add_generation_prompt=True)
    
    # if debug: print(f'chat template applied {text}')

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if debug: print(f'tokenized inputs: {inputs}')

    with torch.no_grad():
        output_ids = model.generate(inputs['input_ids'], 
                                    max_new_tokens=max_tokens,
                                    do_sample=False,
                                    temperature=None,
                                    top_p=None,
                                    pad_token_id=tokenizer.eos_token_id)
        
        if debug: print(f'output_ids: {output_ids}')
        return tokenizer.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()




if __name__=='__main__':
    debug = False    
    # MODEL_NAME = "EleutherAI/pythia-70m"
    MODEL_NAME = "EleutherAI/pythia-410m"

    device = 'cuda' if cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                dtype=torch.float16 if device=='cuda' else torch.float32,
                                device_map = "auto" if device=='cuda' else None,
                                )

    model.eval()

    user_input = input('?')

    while user_input:
        with open('example_file.txt') as f:
            template = f.read()
            ready_for_ai = template.replace('{sentence}', user_input)
            # print(f'ready for ai {ready_for_ai}')
        print(ask(ready_for_ai, max_tokens=20, debug=debug))
        user_input = input('?')


