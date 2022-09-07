# Generating Oversampled Datasets

1) Set up environment (be in the right conda environment and run export PYTHONPATH="$PYTHONPATH:/home/roigvilamalam/projects/deepproblog/")

2) Run generate_all.sh PATH_TO_FOLDERS. This automatically runs the code "generate_data.py" for each folder in PATH_TO_FOLDERS

3) Run make_oversamplings.sh PATH_TO_INPUT_FOLDER PATH_TO_OUTPUT_FOLDER. This removes all datasets already in PATH_TO_OUTPUT_FOLDER. Then, it takes the datasets generated in PATH_TO_INPUT_FOLDER and makes the oversampled versions in PATH_TO_OUTPUT_FOLDER. 
   1) Alternatively, use add_oversamplings.sh PATH_TO_INPUT_FOLDER PATH_TO_OUTPUT_FOLDER to do the same without removing the datasets already in PATH_TO_OUTPUT_FOLDER (this will still overwrite datasets with the same name) 

4) Run clean_datasets.sh PATH_TO_FOLDER. This will generate the clean version for each training dataset in PATH_TO_FOLDER. This is done by making a copy of the training file that does not contain any of the training cases where no complex event occurs.
