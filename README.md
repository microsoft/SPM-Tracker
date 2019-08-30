## SPM-Tracker

This is an inference-only implementation of SPM-Tracker

* SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking, CVPR 2019

-----------

### Installation

#### Prerequisite Lib

* **Python3** is required for this project.  

* You should install **PyTorch** and **Torchvision** lib firstly. The installation can refer to official repo https://pytorch.org/ .

* The other Python dependences can be installed by **pip**.

```bash
pip install -r requirements.txt
```

#### Compile Cython Extensions

Some components are written in Cython & CUDA. It's necessary to manually compile them.

```bash
bash compile.sh
```

Known issue for **torch >= 1.0.0** on *Win32* platform: the MSCV compiler cannot handle `#warning`, see this pull request: 
https://github.com/pytorch/pytorch/pull/20484

------------

### Demo

* `tools/track_video.py` shows an example to track a video

You may just simply call

```bash
python tools/track_video.py --video data/demo/boy.avi --cfg configs/spm_tracker/alexnet_c42_otb.yaml
```

------------------------------------

