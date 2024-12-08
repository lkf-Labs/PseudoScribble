# IoU analysis
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Load the list from the Pickle file
# with open('model_simpleclick_vit_base_clicks_20_plot.pkl', 'rb') as file:
#     loaded_list = pickle.load(file)
#
# # Print the loaded list to verify
# print(len(loaded_list))
# print(loaded_list)

click_id = 20
dataset_name = 'MeibomianGlands'  # MeibomianGlands  / MGD-1K
# Load the list from the Pickle file
with open(f'{dataset_name}_model_pseudoscribble_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_pseu_scribble = pickle.load(file)

with open(f'{dataset_name}_model_RITM_h32_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_ritm_h32 = pickle.load(file)

with open(f'{dataset_name}_model_RITM_h18_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_ritm_h18 = pickle.load(file)

with open(f'{dataset_name}_model_RITM_u2net_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_ritm_u2net = pickle.load(file)

with open(f'{dataset_name}_model_focalclick_segb0_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_focal_b0 = pickle.load(file)

with open(f'{dataset_name}_model_focalclick_segb3_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_focal_b3 = pickle.load(file)

with open(f'{dataset_name}_model_focuscut_r50_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_focus_r50 = pickle.load(file)

with open(f'{dataset_name}_model_focuscut_r101_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_focus_r101 = pickle.load(file)

with open(f'{dataset_name}_model_FCFI_r101_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_fcfi_r101 = pickle.load(file)

with open(f'{dataset_name}_model_FCFI_h18s_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_fcfi_h18s = pickle.load(file)

with open(f'{dataset_name}_model_simpleclick_vit_base_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_simpleclick_vitb = pickle.load(file)

with open(f'{dataset_name}_model_simpleclick_vit_large_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_simpleclick_vitl = pickle.load(file)

with open(f'{dataset_name}_model_simpleclick_vit_huge_clicks_{click_id}_plot.pkl', 'rb') as file:
    data_simpleclick_vith = pickle.load(file)

plt.figure(figsize=(12, 9))
plt.plot(1 + np.arange(click_id), [x * 100 for x in data_pseu_scribble], linewidth=2, label='Ours', linestyle='-', color='#0080ff')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_ritm_h32], linewidth=2, label='RITM-HRNet32', linestyle='-', color='#008000')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_ritm_h18], linewidth=2, label='RITM-HRNet18', linestyle='-', color='#ff0000')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_ritm_u2net], linewidth=2, label='RITM-U2net', linestyle='-', color='#0000ff')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_focal_b0], linewidth=2, label='FocalClick-SegF-B0', linestyle='-', color='#6A5ACD')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_focal_b3], linewidth=2, label='FocalClick-SegF-B3', linestyle='-', color='#A2142F')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_focus_r50], linewidth=2, label='FocusCut-ResNet-r50', linestyle='-', color='#EDB120')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_focus_r101], linewidth=2, label='FocusCut-ResNet-r101', linestyle='-', color='#4DBEEE')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_fcfi_r101], linewidth=2, label='FCFI-ResNet-101', linestyle='-', color='#000000')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_fcfi_h18s], linewidth=2, label='FCFI-HRNet18s', linestyle='-', color='#FFA500')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_simpleclick_vitb], linewidth=2, label='SimpleClick-ViT-B', linestyle='-', color='#77AC30')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_simpleclick_vitl], linewidth=2, label='SimpleClick-ViT-L', linestyle='-', color='#7E2F8E')
plt.plot(1 + np.arange(click_id),  [x * 100 for x in data_simpleclick_vith], linewidth=2, label='SimpleClick-ViT-H', linestyle='-', color='#D95319')

plt.title('MG-203', fontsize=22)
plt.grid()
plt.legend(loc=4, fontsize='x-large')

min_val, max_val, step = (62, 77, 1)
plt.yticks(np.arange(min_val, max_val, step=step), fontsize='xx-large')
plt.xticks(1 + np.arange(click_id), fontsize='xx-large')
plt.xlabel('Number of Clicks', fontsize='xx-large')
plt.ylabel('mIoU score (%)', fontsize='xx-large')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.savefig('mg_plot.png', format='png', dpi=300)
plt.show()
plt.close()
