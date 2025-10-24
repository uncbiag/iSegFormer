import os
import gdown

# download models
save_folder ='saves'
os.makedirs(save_folder, exist_ok=True)

# image for demo
data_url = 'https://drive.google.com/uc?id=1LSqMjRR8uslNivVQGDSGkQ5Z6JLpGj-0'
output = f'{save_folder}/image.nii.gz'
gdown.download(data_url, output, quiet=False)

# click-based interaction module
weight_fbrs_url = 'https://drive.google.com/uc?id=1L16obYAba5LZPd8Y-TaMCBIkbLWvEnnh'
output = f'{save_folder}/fbrs.pth'
gdown.download(weight_fbrs_url, output, quiet=False)

# scribble-based interaction module
weight_s2m_url = 'https://drive.google.com/uc?id=1KIQnp2Pmwy0c9Ui8f8ETS6zi5eb60ro5'
output = f'{save_folder}/s2m.pth'
gdown.download(weight_s2m_url, output, quiet=False)

# propagation modules
weight_stcn_url = 'https://drive.google.com/uc?id=1gZzI00UXGbOOXGFFVeeR5e04JUvi4T2A'
output = f'{save_folder}/stcn.pth'
gdown.download(weight_stcn_url, output, quiet=False)

weight_stcn_no_cycle_url = 'https://drive.google.com/uc?id=1AtbmPPnrwQX0934Sbp0joXPu6KI0X0sc'
output = f'{save_folder}/stcn_ft_without_cycle.pth'
gdown.download(weight_stcn_no_cycle_url, output, quiet=False)

weight_stcn_cycle_url = 'https://drive.google.com/uc?id=171dlMGw3eYyfFFdbRYEb7y_iGenOYKeF'
output = f'{save_folder}/stcn_ft_with_cycle.pth'
gdown.download(weight_stcn_cycle_url, output, quiet=False)

# progation fusion module
weight_fusion_stcn_url = 'https://drive.google.com/uc?id=103leTCako9Ba5ThLcibQovvBjf732IOM'
output = f'{save_folder}/fusion_stcn.pth'
gdown.download(weight_fusion_stcn_url, output, quiet=False)

# download data
data_folder ='data'
os.makedirs(data_folder, exist_ok=True)

# AbdomenCT-1k
data_url = 'https://drive.google.com/uc?id=1JFBzDxuURj1T0i7ztxDE0KmstMdvrspb'
output = f'{data_folder}/AbdomenCT-1K.zip'
gdown.download(data_url, output, quiet=False)
