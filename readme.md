# iMaterialist (Fashion) 2019 at FGVC6
test task for Int20h hackathon 
data_soScientists - Illia Ovcharenko, Anton Babenko, Alexey Kuzmenko

## Data and task descriptions

Kaggle training dataset was randomly splited into train/test/val parts.
All information about attributes was dropped.
Metric - mAP 

## Proposed solution 
As a baseline solution we consider task as a multiclass segmenatation problem.  For segmentation we use Unet architecture with EfficientNet B2 encoder. To get separate class instances we use ```from scipy.ndimage import label```. 

Sigmoid activation with Focal loss to somehow deal with high class inbalance.
As a training metric we use dice score just to have some understading about current segmentation quality.

Learning rate scheduling using ReduceOnPlatue.
Gradiend accumulation to simulate higher batch size and smoother convergence.
All the experiments are logged using tensorbord and save into ```logs``` folder.


## Next steps
- some time for network training - due to a hard deadline we don't have enough time to train our network
- write evaluation code
- do something with image loading time cause it is the main bottle neck during traing. Maybe a good decision would be replace OpenCV with another library for image loading or just save all images in raw .npy format. This should alow to quickly iterare throught differerent idead and this is vital for any machine learning task.
- increase image size to detect some tiny classes like ribbon, rivet, etc.
- stratified by outfit splits
- move to mask-rcnn network

