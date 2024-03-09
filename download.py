import os
import gdown

save_folder ='saves'
os.makedirs(save_folder, exist_ok=True)

# image for demo
data_url = 'https://drive.google.com/uc?id=1OrhlObKL4UwWkUIpwNeAIsz-koDr3OQw'
output = f'{save_folder}/image.nii.gz'
gdown.download(data_url, output, quiet=False)

# click-based interaction module
weight_fbrs_url = 'https://drive.google.com/uc?id=1smsbmxmfNBituJHlWFSYVaj4CECXUHbH'
output = f'{save_folder}/fbrs.pth'
gdown.download(weight_fbrs_url, output, quiet=False)

# scribble-based interaction module
weight_s2m_url = 'https://drive.google.com/uc?id=1g78flBPhsJkBzUtg4WRAcAM_bMkMYtNV'
output = f'{save_folder}/s2m.pth'
gdown.download(weight_s2m_url, output, quiet=False)

# propagation modules
weight_stcn_url = 'https://drive.google.com/uc?id=1xdnNXD-5uGj3BoIMrh-0mwI4vCkU10nr'
output = f'{save_folder}/stcn.pth'
gdown.download(weight_stcn_url, output, quiet=False)

weight_stcn_no_cycle_url = 'https://drive.google.com/uc?id=1mSei91L6L7c263nUgGVtnam6VUaPIMHb'
output = f'{save_folder}/stcn_ft_without_cycle.pth'
gdown.download(weight_stcn_no_cycle_url, output, quiet=False)

weight_stcn_cycle_url = 'https://drive.google.com/uc?id=1nMoeSUDRSBHKoFfxA92UnxoNErxc0x6r'
output = f'{save_folder}/stcn_ft_with_cycle.pth'
gdown.download(weight_stcn_cycle_url, output, quiet=False)

# progation fusion module
weight_fusion_stcn_url = 'https://drive.google.com/uc?id=1hf3yHizbgcOQx6LLFCX_rvHR30p4ioVp'
output = f'{save_folder}/fusion_stcn.pth'
gdown.download(weight_fusion_stcn_url, output, quiet=False)
