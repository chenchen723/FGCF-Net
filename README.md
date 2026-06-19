
# Frequency-domain Guided Cross-layer Fusion Network for Polyp Segmentation (FGCF-Net)

[cite_start]This repository contains the official PyTorch implementation of the paper: **"Frequency-domain guided cross-layer fusion network for polyp segmentation"**[cite: 3].

## 📖 Overview
[cite_start]Accurate polyp segmentation is vital for the early diagnosis of colorectal cancer[cite: 8]. [cite_start]However, it remains challenging due to scale variations and indistinct boundaries[cite: 8]. [cite_start]To address this, we propose **FGCF-Net**, a novel Frequency-Domain Guided Cross-Layer Fusion Network designed to enhance feature representations and boundary precision[cite: 10]. 

[cite_start]Our framework explicitly injects frequency-domain priors into spatial representation learning and achieves decoupled, conflict-free multi-scale feature fusion[cite: 81, 83].

### Key Components:
* [cite_start]**Differential Spatial-Frequency Interaction (DSFI) Module:** Leverages frequency-domain priors and directional differential operators to bridge spatial representation learning with high-frequency detail extraction, significantly improving boundary sensitivity[cite: 11].
* [cite_start]**Cross-layer Fusion (CLF) Module:** Employs a dual-branch attention mechanism to independently model semantic consistency and structural discrepancies, effectively resolving cross-scale feature conflicts[cite: 18].
* [cite_start]**Semantic Aggregation (SA) Module:** Aggregates multi-scale features to mitigate semantic information loss caused by progressive downsampling, providing consistent global contextual guidance[cite: 12, 17].

## 📊 Datasets
[cite_start]Our model was trained and evaluated on five publicly available colorectal polyp datasets[cite: 603]. [cite_start]The benchmark datasets used in this study are publicly available at their respective repositories[cite: 948, 949]:

* [cite_start]**CVC-ClinicDB:** [https://polyp.grand-challenge.org/CVCClinicDB/](https://polyp.grand-challenge.org/CVCClinicDB/) [cite: 949, 950]
* [cite_start]**Kvasir-SEG:** [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/) [cite: 950]
* [cite_start]**CVC-ColonDB:** [http://mv.cvc.uab.es/projects/colon-qa/cvc-colondb/](http://mv.cvc.uab.es/projects/colon-qa/cvc-colondb/) [cite: 952, 953]
* [cite_start]**ETIS:** [https://polyp.grand-challenge.org/EtisLarib/](https://polyp.grand-challenge.org/EtisLarib/) [cite: 953]
* [cite_start]**CVC-300:** [http://adas.cvc.uab.es/endoscene/](http://adas.cvc.uab.es/endoscene/) [cite: 955]

*(Note: For a centralized collection of these dataset links, please refer to our companion data repository: [https://github.com/chenchen723/Polyp-Dataset](https://github.com/chenchen723/Polyp-Dataset))*

## 🚀 Getting Started

### Prerequisites
[cite_start]The model is implemented using the **PyTorch** framework[cite: 624].
* Python 3.x
* PyTorch
* torchvision
* *(Add any other specific libraries like opencv, numpy, etc., here)*

### Installation
```bash
git clone [https://github.com/chenchen723/FGCF-Net.git](https://github.com/chenchen723/FGCF-Net.git)
cd FGCF-Net
# Install required packages
pip install -r requirements.txt

```

### Training & Testing

*(Please provide your specific command-line instructions here. For example:)*

```bash
# Example training command
python train.py --dataset_path ./data --batch_size 16

# Example testing command
python test.py --weights ./checkpoints/best_model.pth

```

## 🏆 Quantitative Results

Extensive experiments demonstrate that FGCF-Net achieves robust performance across multiple datasets.

* 
**CVC-ClinicDB:** 93.4% Dice Score, 88.8% IoU 


* 
**Kvasir-SEG:** 91.3% Dice Score, 86.2% IoU 


* 
**CVC-300:** 90.5% Dice Score, 84.2% IoU 



These results indicate that explicit spatial-frequency modeling significantly enhances the accuracy and robustness of polyp segmentation, offering a reliable tool for clinical endoscopy.

## 🤝 Acknowledgments

This work was supported by the National Key Research and Development Program of China (Grant No. 2017YFE0135700), Hebei Key Laboratory of Industrial Intelligent Perception (No. SZX2021013), and the Science and Technology Project of Hebei Education Department (Grant No. ZD2022102).

```

```
