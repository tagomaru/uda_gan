# uda_gan

Use GAN for unsupervised domain adaptation.
@Stanford CS236g

## Prerequites
- Python 3.5
- PyTorch 

## Usage
Clone the repository
``` Ruby
https://github.com/jingxiaoliu/uda_gan.git
cd uda_gan
```

## Get the dataset
Send an email to
liujx@stanford.edu

## Run the implementation
Open the jupyter notebook
``` Ruby
run_udagan_shm.ipynb
```

There are four sections in the notebook. Run one-by-one to test the implementation.
- Form dataset
Change the data path
``` Ruby
filePath
```
to your own directory that saves the data
- CycleGAN: this section uses a CycleGAN [[2]](#2) to align source and target domain data in pixel-level.
- DANN [[1]](#1) this section uses a DANN model to align source and target domain data in feature-level.
- Generate to adapta [[3]](#3): this section presents a baseline model for feature-level adaptation.


## References
<a id="1">[1]</a>
Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lem-pitsky.   Domain-adversarial  training  of  neural  networks.The  Journal  of  Machine  Learning  Research,17(1):2096–2030, 2016.

<a id="2">[2]</a> 
C. Yunjey. MNIST-to-SVHN and SVHN-to-MNIST. https://github.com/yunjey/mnist-svhn-transfer

<a id="3">[3]</a> 
S. Sankaranarayanan, Y. Balaji, C. D. Castillo, and R. Chellappa. Generate to adapt:  Aligning domainsusing generative adversarial networks.  InProceedings of the IEEE Conference on Computer Vision andPattern Recognition, pages 8503–8512, 2018.
