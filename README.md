# Hyperbolic Busemann Learning with Ideal Prototypes

This is the implementation of paper [Hyperbolic Busemann Learning with Ideal Prototypes (NeurIPS2021)](https://arxiv.org/pdf/2106.14472.pdf).

![Figure 1](additional_files/Figure1.png)

## How to use?

### Ideal Prototype Learning
As the first step, you should learn ideal prototypes for the classes in the ultimate task.
To run the code and learn the prototypes, the `number of classes` and `output dimension` should be specified.

To learn prototypes with 50 output dimensionality and 100 classes, use the code,

```shell
python prototype_learning.py -d 50 -c 100
```

The output will be `prototypes-50d-100c.npy` saved in `prototypes` directory.

### Main code
Once the prototypes are ready, it's time to run the main code.
To run `HBL.py`, the parameters in the argparser should be specified,
```shell
python HBL.py --data_name cifar100 -e 1110 -s 128 -r adam -l 0.0005 -c 0.00005 --mult 0.1 --datadir data/ --resdir runs/output_dir/cifar/ --hpnfile prototypes/prototypes-50d-100c.npy --logdir test --do_decay True --drop1 1000 --drop2 1100 --seed 100
```

Further explanation will be added soon.

### Citation
Please consider citing this work using this BibTex entry,

```bibtex
@inproceedings{atigh2021hyperbolic,
  title={Hyperbolic Busemann Learning with Ideal Prototypes},
  author={Atigh, Mina Ghadimi and Keller-Ressel, Martin and Mettes, Pascal},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
