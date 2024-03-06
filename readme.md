# ArIB-BPS

### <p align="center"> Learned Lossless Image Compression based on Bit Plane Slicing</p>
####  <p align="center"> CVPR 2024</p>
####  <p align="center"> Zhe Zhang, Huairui Wang, Zhenzhong Chen, Shan Liu</p>

![framework](./assets/overview.png)

## Environments
```
conda env create -f arib_bps.yml
```

## Entropy Coder Compiling
```
cd src/utils/coder
sh compile.sh
```

## Folder Structure
Model weights are stored in a folder containing "sig.pth", which is the model for significant planes and "ins.pth", which is the model for insignificant planes. Each model has a corresponding configuration file, refer to src/config folder.

We use three kinds of folder structures for dataset:
- CIFAR10 dataset (torchvision.datasets.CIFAR10), where images should be download using argument --download=True.
- ImageFolder dataset (torchvision.datasets.ImageFolder), where images are stored in subfolders of the dataset root.
- FileDataset (utils.dataset.FileDataset), where images are stored in the dataset root.

## Encoding and Decoding
```
python compress.py (--encode | --decode) --input [path to input file] --output [path to output file] --config [path of config file] --model [path to model folder]
```

## Testing
```
python test.py --dataset [cifar10|imagenet32|imagenet64|imagenet64_small] --dataset_type [cifar10|imagefolder|filedataset] --data_dir [path to dataset folder] --model [path to model folder] --mode [inference|single|dataset|speed] <--batchsize> [batch size for inference, or size of dataset for dataset compression setting] <--log_path> [logger path for dataset/single compression setting]
```

- Inference mode: evaluate the thereotical compression performance.
- Single mode: evaluate the single-image compression performance.
- Dataset mode: evaluate the dataset compression performance.
- Speed mode: evaluate the inference speed of the model.

## Training
Models for significant planes and insignificant planes are trained separately. 

```
python train.py --num_gpus [number of gpus] --mode [sig|ins|finetune] --dataset_type [cifar10|imagefolder|filedataset] --data_dir [path to dataset folder] --valid_num [size of validation set] --config [path of config file] --dropput [probability of dropout] --save_dir [path to save log and weights] --lr [learning rate] --batch_size [batch size for each gpu] --num_iters [number of iterations] --log_interval [interval of logging] --valid_interval [interval of validation] --decay_rate [decay rate of learning rate] --decay_interval [interval of decaying learning rate] <--resume> [path of pretrained checkpoints] <--qp_path> [relative path to qplist, only used for finetune mode] <--master_port> [master port for ddp]
```

- Sig mode: train the model for significant planes.
- Ins mode: train the model for insignificant planes.
- Finetune mode: finetune the model for significant planes using discretized sampling. To use this mode, a pretrained model for both significant and insignificant planes should be provided, and a dataset with qplist should be provided, which could be generated using
```
python generate_finetune_dataset.py --dataset_type [cifar10|filedataset] --data_dir [path to dataset folder] --model [path to model folder] <--data_dst>[path to folder of qp list and imgs, only required for cifar10 dataset]  <--qp_path> [relative path to qplist]
``` 

## Pretrained Models
Pretrained models are available [here](https://drive.google.com/drive/folders/1RiI2Fzqu0lhjHSpjrOVPb0eOzvnJD9XC?usp=sharing). Related configuration files are available in src/config folder.

## Citation

```
@InProceedings{zhang2024learned,
    author    = {Zhang, Zhe and Wang, Huairui and Chen, Zhenzhong and Liu, Shan},
    title     = {Learned Lossless Image Compression based on Bit Plane Slicing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}