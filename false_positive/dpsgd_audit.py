import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from datasets import load_dataset
import itertools
import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer

eps = 1.0
delta=1e-4
batch_size = 10
per_device_batch_size=10
assert batch_size % per_device_batch_size == 0
grad_accumulate_num = batch_size // per_device_batch_size
audit_res_file = 'audit_res_t5privtranstesttest.csv'
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
text = ds['train']['text']
# print(len(text))
print(text[1])

model_name = "allenai/t5-small-next-word-generator-qoogle"

tokenizer = T5Tokenizer.from_pretrained(model_name,local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(model_name,local_files_only=True).cuda()
vocab = tokenizer.get_vocab()
sequences=[]
for se in text:
    word_list = se.split()
    if len(word_list) < 10:
        continue
    extra_id = vocab.get('<extra_id_0>')
    end = vocab.get('</s>')
    # import pdb;pdb.set_trace()
    se_ids = tokenizer.encode(se, truncation=True, max_length=10)
    sequences.append((torch.tensor(se_ids[:-2]+[extra_id, end]), torch.tensor([extra_id,se_ids[-2], end])))


# sequences = [(encode_sequence(seq), word_to_ix[label]) for seq, label in sequences]

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return sequences[idx]

dataset = TextDataset(sequences)
dataloader = DataLoader(dataset, batch_size=per_device_batch_size, shuffle=True)
dataloader2 = DataLoader(dataset, batch_size=per_device_batch_size, shuffle=True)

embedding_dim = 10
hidden_dim = 20
num_layers = 8
observe_num = 10000


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
from fastDP.accounting import accounting_manager
manager = accounting_manager.RDPManager(alphas = accounting_manager.DEFAULT_ALPHAS)
sigma = manager.compute_sigma(eps, delta,sample_rate=1,steps=1) # single step eps
model.train()

privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(dataset),
    epochs=num_epochs,
    max_grad_norm=1.0,
    noise_multiplier=sigma,
    # target_epsilon=eps,
)
privacy_engine.attach(optimizer)
# attaching to optimizers is not needed for multi-GPU distributed learning

# exit()
#----- standard training pipelin
o_list, o_prime_list = [],[]
train_loader_iter =None
train_loader2_iter =None
total_step = observe_num+2
cur_step = 0
model.train()
step = 0
while step < observe_num and cur_step < total_step:
    cur_step += 1
    success=1
    
    optimizer.zero_grad()
    # pbar = tqdm.tqdm(enumerate(zip(train_loader,train_loader2), 1), total=len(train_loader), ncols=100)
    del train_loader_iter
    del train_loader2_iter
    train_loader_iter = iter(dataloader)
    train_loader2_iter = iter(dataloader2)
    epoch_finish = False
    while not epoch_finish and step < observe_num:
        optimizer.zero_grad()
        try:
            inputs, targets = next(train_loader_iter)
        except StopIteration:
            epoch_finish = True
            continue
        # print(inputs)
        # print(targets)
        # import pdb;pdb.set_trace()
        outputs = model(input_ids=inputs.cuda(), labels=targets.cuda())
        loss = outputs.loss
        # loss.backward()
        optimizer.virtual_step(loss=loss)

        if epoch_finish:
            break
        # privacy_engine._create_noisy_clipped_gradient()
        # import pdb;pdb.set_trace()
        param = next(itertools.islice(model.named_parameters(), 2,3)) # next(itertools.islice(model.named_parameters(), 4, 5))
        print("auditing param:", param[0])
        k = 0
        # import pdb;pdb.set_trace()
        o_prime = param[1].summed_grad.flatten()[k]
        o_prime += 1/batch_size
        # privacy_engine.unlock()
        optimizer.zero_grad()

        del loss
        del outputs
        del inputs
        del targets

        try:
            inputs, targets = next(train_loader2_iter)
        except StopIteration:
            epoch_finish = True
            continue
        outputs = model(input_ids=inputs.cuda(), labels=targets.cuda())
        loss = outputs.loss
        # loss.backward()
        optimizer.step(loss=loss)

        if epoch_finish:
            break
        # privacy_engine._create_noisy_clipped_gradient()
        param = next(itertools.islice(model.named_parameters(), 2,3)) # next(itertools.islice(model.named_parameters(), 4, 5))
        # param = next(itertools.islice(model.named_parameters(), 4, 5))
        print("auditing param:", param[0])
        k = 0
        o = param[1].summed_grad.flatten()[k]
        # privacy_engine.unlock()

        del loss
        del outputs
        del inputs
        del targets
        
        optimizer.zero_grad()


        if success:
            step += 1
            ### dp_step end ###
            ### dp_step is the function in fastDP/privacy_engine.py, which can be called directly by optimizer.step(). For auditing, we need to additionally observe the gradients of the shuffle-able parameters.

            o_list.append(o.item())
            o_prime_list.append(o_prime.item())
            with open(audit_res_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([o.item(), o_prime.item()])
            del o
            del o_prime

