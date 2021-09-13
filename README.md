# SRDRL (A Blind Super-Resolution Framework With Degradation Reconstruction Loss)
By Zongyao He, Zhi Jin, Yao Zhao

SRDRL is a blind SR framework without prior knowledge that can handle multiple degradations.

By using an efficient SR network, a degradation simulator, and a novel degradation reconstruction loss, SRDRL provides satisfactory SR results on multi-degraded datasets.

#### BibTex
```
@article{he2021srdrl,
  title={SRDRL: A Blind Super-Resolution Framework With Degradation Reconstruction Loss},
  author={He, Zongyao and Jin, Zhi and Zhao, Yao},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```

## Dependencies 
* Python >= 3.7 (Recommend to use Anaconda or Miniconda)
* PyTorch >= 1.0
* lmdb
* numpy
* opencv-python

## Test
To test the pre-trained degradation simulator (generating fake LR images), run:
```
python test_degnet.py
```

To test the pre-trained SRDRL (generating SR images), run:
```
python test.py
```

The testing results will be in the ./results folder. To test your own models and on your own datasets, you can modify the configuration json file in the ./options/test folder.

## Train
Download the datasets from the [official DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

First you need to crop the DIV2K HR images into fixed size image pathces, put the DIV2K_train_HR and DIV2K_valid_HR datasets in the ./downsampling folder and run:
```
cd scripts/
python sub_images.py
```

Then you need to degrade the DIV2K HR images with different blur, noise, and downsampling, use Matlab to run ./scripts/generate_degradated_LR.m.

After generating the DIV2K_train_LR and DIV2K_valid_LR dataset you want, put the training dataset in the ./datasets/DIV2K800 folder, and put the validation dataset in the ./datasets/DIV2K100 folder.

Once the dataset preparation is finished, you can train the degradation simulator, run:
```
python train_degnet.py
```

Put the degradation simulator model in ./experiments/pretrained_models folder. TNow you can use the degradation reconstruction loss to train the SR network, run:
```
python train.py
```

The training results will be in the ./experiments folder.
