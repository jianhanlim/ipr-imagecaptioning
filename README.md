# Image Captioning IPR Protection

[Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320321004659) | [ArXiv](https://arxiv.org/abs/2008.11009)

### Official implementation of the paper: "Protect, Show, Attend and Tell: Empowering Image Captioning Models with Ownership Protection"

#### Pattern Recognition

16 Sept. 2021

## Description

<p align="justify"> By and large, existing Intellectual Property (IP) protection on deep neural networks typically i) focus on image classification task only, and ii) follow a standard digital watermarking framework that was conventionally used to protect the ownership of multimedia and video content. This paper demonstrates that the current digital watermarking framework is insufficient to protect image captioning tasks that are often regarded as one of the frontiers AI problems. As a remedy, this paper studies and proposes two different embedding schemes in the hidden memory state of a recurrent neural network to protect the image captioning model. From empirical points, we prove that a forged key will yield an unusable image captioning model, defeating the purpose of infringement. To the best of our knowledge, this work is the first to propose ownership protection on image captioning task. Also, extensive experiments show that the proposed method does not compromise the original image captioning performance on all common captioning metrics on Flickr30k and MS-COCO datasets, and at the same time it is able to withstand both removal and ambiguity attacks.</p>

## Preparation
### Dataset
- **MSCOCO**: Download the COCO train2014 and val2014 data from [here](http://cocodataset.org/#download)
- **Flickr30k**: Download from [here](http://shannon.cs.illinois.edu/DenotationGraph/)
- **Flickr8k**: Download from [here](https://illinois.edu/fb/sec/1713398)

### Pretrained ResNet50
- Follow this [repo](https://github.com/DeepRNN/image_captioning) to download the pretrained ResNet50 net [here](https://app.box.com/s/17vthb1zl0zeh340m4gaw0luuf2vscne) to use it to initialize the CNN part.

## How to run

For compability issue, a docker image has been created and pushed to docker hub. Run the command to download it:

```docker pull limjianhan/tensorflow:1.13.1-gpu```

Two folders in this repo:

```addition_bi``` is the implementation of element-wise addition model

```multiplication_bi``` is the implementation of element-wise multiplication model

### To start the docker

Refer to the ```scripts/run_docker.sh``` in the respective folders. Please set the absolute path to the ```CODE_DIR, CNN_DIR, DATA_DIR and COCO_EVAL_DIR```. Copy the command and paste in terminal to start the docker.

### To train the model

Refer to the ```scripts/training_steps.txt``` in the respective folders. Copy the command and paste in terminal to train the model for MSCOCO, Flickr30k or Flickr8k dataset. The evaluation will be run automatically after the training is complete. The result is saved to ```tmp``` folder.  

### To attack the model

Refer to the ```scripts/attack_key_steps.txt``` to attack the model with forged key.

Refer to the ```scripts/attack_sign_steps.txt``` to attack the model with fake signature.



## Citation
If you find this work useful for your research, please cite
```
@article{IcIPR,
  author    = {Jian Han Lim and
               Chee Seng Chan and
               Kam Woh Ng and
               Fixin Fan and
               Qiang Yang},
  title     = {Protect, show, attend and tell: Empowering image captioning models with ownership protection},
  journal   = {Pattern Recognit.},
  year      = {2021},
  url       = {https://doi.org/10.1016/j.patcog.2021.108285},
  doi       = {10.1016/j.patcog.2021.108285},
}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to
`jianhanl98 at gmail.com` or `cs.chan at um.edu.my`.

## References
* The baseline implementation of "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015) was based on this [repo](https://github.com/DeepRNN/image_captioning)
* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
* [Microsoft COCO dataset](http://mscoco.org/)
* [Flickr30k dataset](http://shannon.cs.illinois.edu/DenotationGraph/)
* [Flickr8k dataset](https://illinois.edu/fb/sec/1713398)

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file).

&#169;2021 Universiti Malaya and WeBank.
