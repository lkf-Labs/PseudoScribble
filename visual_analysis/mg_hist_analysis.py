# IoU analysis
import pickle
import numpy as np
import matplotlib.pyplot as plt

click_id = 5
dataset_name = 'MeibomianGlands'  # MeibomianGlands  / MGD-1K

# plt.figure(dpi=300)
# Load the list from the Pickle file
with open(f'{dataset_name}_model_pseudoscribble_clicks_{click_id}.pkl', 'rb') as file:
    data_pseu_scribble = pickle.load(file)

with open(f'{dataset_name}_model_RITM_h18_clicks_{click_id}.pkl', 'rb') as file:
    data_ritm_h32 = pickle.load(file)

with open(f'{dataset_name}_model_FCFI_r101_clicks_{click_id}.pkl', 'rb') as file:
    data_fcfi_r101 = pickle.load(file)

with open(f'{dataset_name}_model_simpleclick_vit_huge_clicks_{click_id}.pkl', 'rb') as file:
    data_simpleclick_vith = pickle.load(file)

# Plot
plt.hist(sorted(data_ritm_h32), bins=50, range=(0, 1.0), histtype='step', label='RITM-H32', alpha=0.8, color='black', density=True)
# plt.hist(sorted(sbd_iou_20_clicks_sfb3), bins=50, range=(0, 1.0), histtype='step', label='FoclickClick-SF-B3 (C+L)', alpha=0.8, color='black')
plt.hist(sorted(data_fcfi_r101), bins=50, range=(0, 1.0), histtype='step', label=r'baseline',  alpha=1.0, color='green', density=True)
# plt.hist(sorted(sbd_iou_20_clicks_vitb), bins=50, range=(0, 1.0), label='Ours-ViT-B (SBD)', alpha=0.5, color='#e85300')
plt.hist(sorted(data_simpleclick_vith), bins=50, range=(0, 1.0), histtype='step', label='SimpleClick-ViT-H', alpha=1.0, density=True)
plt.hist(sorted(data_pseu_scribble), bins=50, range=(0, 1.0), histtype='step', label='pseudo-scribble-u2net', alpha=1.0, color='#e85300', density=True)

plt.xlabel(f'IoU@{click_id}', fontsize='x-large')
plt.ylabel('Count (%)', loc='center', fontsize='x-large')
plt.yticks(fontsize='large')
plt.xticks(fontsize='large')

plt.title('MG-203', fontsize='x-large')
plt.legend(loc='upper left', fontsize='large')
plt.grid(True)
plt.savefig(f'mg_hist_{click_id}.png', format='png', dpi=300)
plt.show()
plt.close()
