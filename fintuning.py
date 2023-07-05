
# Author: Shilong
# Creat Data: 2023-7-4
# Modify Date:2023-
# Description:Fine-tuning示例代码
from BEATs import BEATs, BEATsConfig

# load the fine-tuned checkpoints
checkpoint = torch.load(r'E:\Shilong\murmur\03_Classifier\LM\BEATs\BEATs_iter2_finetuned_on_AS2M_cpt2.pt')
# checkpoint = torch.load('/path/to/model.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

# predict the classification probability of each class
audio_input_16khz = torch.randn(3, 10000)
padding_mask = torch.zeros(3, 10000).bool()

probs = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=2))):
    top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
    print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')