# REDAffectiveLM <br>
<img src= 'images/model_architecture.png' style="max-width: 100%;"> 

**Distinguishing Natural and Computer-Generated Images using Multi-Colorspace fused EfficientNet** </br>
Manjary P Gangan, Anoop K, and Lajish V L </br>
Computational Intelligence and Data Analytics (CIDA Lab) </br>
Department of Computer Science </br>
University of Calicut, India

:memo: Paper : https://www.sciencedirect.com/science/article/abs/pii/S2214212622001247 </br>
:earth_asia: Link: https://dcs.uoc.ac.in/cida/projects/dif/mceffnet.html

**Abstract**: The problem of distinguishing natural images from photo-realistic computer-generated ones either addresses _natural images versus computer graphics_ or _natural images versus GAN images_, at a time. But in a real-world image forensic scenario, it is highly essential to consider all categories of image generation, since in most cases image generation is unknown. We, for the first time, to our best knowledge, approach the problem of distinguishing natural images from photo-realistic computer-generated images as a three-class classification task classifying natural, computer graphics, and GAN images. For the task, we propose a Multi-Colorspace fused EfficientNet model by parallelly fusing three EfficientNet networks that follow transfer learning methodology where each network operates in different colorspaces, RGB, LCH, and HSV, chosen after analyzing the efficacy of various colorspace transformations in this image forensics problem. Our model outperforms the baselines in terms of accuracy, robustness towards post-processing, and generalizability towards other datasets. We conduct psychophysics experiments to understand how accurately humans can distinguish natural, computer graphics, and GAN images where we could observe that humans find difficulty in classifying these images, particularly the computer-generated images, indicating the necessity of computational algorithms for the task. We also analyze the behavior of our model through visual explanations to understand salient regions that contribute to the model's decision making and compare with manual explanations provided by human participants in the form of region markings, where we could observe similarities in both the explanations indicating the powerful nature of our model to take the decisions meaningfully. 

For other inquiries, please contact: </br>
Anoop K, University of Calicut, Kerala, India. :email: anoopk_dcs@uoc.ac.in </br> 
Deepak P., Queen’s University Belfast, Northern Ireland, UK. :email: deepaksp@acm.org </br>
Manjary P Gangan, University of Calicut, Kerala, India. :email: manjaryp_dcs@uoc.ac.in </br>
Savitha Sam Abraham, School of Science and Technology, Örebro University, Örebro, Sweden. :email:  savitha.sam-abraham@oru.se </br>
Lajish V. L., University of Calicut, Kerala, India. :email: lajish@uoc.ac.in :earth_asia: [website](https://dcs.uoc.ac.in/index.php/dr-lajish-v-l)

## Citation
```
will update soon 
```

## Acknowledgement
The frst author wishes to dedicate this work to the ever-loving memory of his father Ayyappan K. The authors thankfully acknowledge Arjun K. Sreedhar, Dheeraj K., Sarath Kumar P. S., and Vishnu S., the postgraduate students of the Department of Computer Science, University of Calicut, who have been involved in dataset procurement. The third author would like to thank the Department of Science and Technology (DST) of the Government of India for fnancial support through the Women Scientist Scheme-A (WOS-A) for Research in Basic/Applied Science under the Grant SR/WOS-A/PM-62/2018. The authors thankfully acknowledge the popular leading digital media company RAPPLER for the data source of news data along with associated emotions from their online portal that very relevantly helped to conduct this research.

