#!/bin/bash
index_imgs=()

for ((i=0; i<=46086; i+=200)); do
   index_imgs[$i]=${i}
done


time=0-11:00
memory=24G



for index in ${index_imgs[@]}; do
	last=46000
        if [ $index -eq $last ]
        then
                index2=46087
        else
                index2=$(($index + 200))
        fi
        version=SEGMENTATION_${index}_${index2}
	
	job_name="$version.job"
	out_file="/n/groups/patel/Hamza/eo_f/$version.out"
	err_file="/n/groups/patel/Hamza/eo_f/$version.err"
		
	sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time segmentation.sh $index $index2
done




