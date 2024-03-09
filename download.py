import gdown

data_url = 'https://drive.google.com/uc?id=1OrhlObKL4UwWkUIpwNeAIsz-koDr3OQw'
output = 'image.nii.gz'
gdown.download(data_url, output, quiet=False)

weight_stcn_url = 'https://drive.google.com/uc?id=1xdnNXD-5uGj3BoIMrh-0mwI4vCkU10nr'
output = 'stcn.pth'
gdown.download(weight_stcn_url, output, quiet=False)

weight_stcn_no_cycle_url = 'https://drive.google.com/uc?id=1mSei91L6L7c263nUgGVtnam6VUaPIMHb'
output = 'stcn_ft_without_cycle.pth'
gdown.download(weight_stcn_no_cycle_url, output, quiet=False)

weight_stcn_cycle_url = 'https://drive.google.com/uc?id=1nMoeSUDRSBHKoFfxA92UnxoNErxc0x6r'
output = 'stcn_ft_with_cycle.pth'
gdown.download(weight_stcn_cycle_url, output, quiet=False)

