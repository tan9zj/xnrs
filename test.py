
import torch
from os.path import join

base_dir = '/var/scratch/zta207/experiments/mind_small/'  # e.g., cfg.dir + cfg.name
epoch = 0  # Replace with the desired epoch number

predictions_path = join(base_dir, 'predictions', f'predictions_{epoch}')

scores_dict = torch.load(predictions_path, map_location='cpu', weights_only=False)  # use 'cpu' unless you need GPU

targets = scores_dict['targets']
scores = scores_dict['scores']
stats = scores_dict['stats']

print("Targets:", targets)
print("Scores:", scores)
print("Stats:", stats)
