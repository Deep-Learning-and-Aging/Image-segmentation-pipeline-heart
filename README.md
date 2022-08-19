Image-segmentation-pipeline-heart

This pipeline aims to extract the heart from 4 chambers view MRI images and run the Deep Learning age predictors in order to improve the biological relevance of the Deep Learning model and better capture heart aging.

There are two main directories.

scripts: where the python scripts can be found. This is the core of the pipeline.

bash: where the bash scripts to submit the jobs using slurm can be found.

There are two steps in this segmentation pipeline. 

The first step consists of developing an unsupervised model to extract the heart from MRIs.
Functions: model_label, model_seg_heart, first_seg.

The second step improves the segmentation by adding three processing steps.
1. Functions: cut, border1, border2, second_seg. This processing step excludes the parts that are 10 pixels away from the heart that represent unnecessary information. 
2. Functions: positions_seg, cover. This processing step recovers the surrounding pixels on the outside of the heart in the segmented image and also the possible excluded parts that are in the middle.
3. Functions: count_pxls_seg, metric. This processing step aims to avoid overfitting that could be caused by the unsupervised model.