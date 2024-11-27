#!/bin/bash

# Check if DATA_PATH and OUTPUT_DIR arguments are provided
if [ -z "$1" ]; then
    echo "Usage: ./convert.sh /path/to/your/data"
    exit 1
fi

# Assign DATA_PATH from the first argument
DATA_PATH="$1"

# Predefined list of dataset subfolders with their corresponding dataset names
datasets="ocr:icdar-2017,icdar-2019,impresso-nzz asr:quaero-broadcast"

# Mapping of dataset names to dataset pseudo names
dataset_pseudo_names="quaero-broadcast:quaero icdar-2017:icdar icdar-2019:icdar impresso-nzz:nzz"

# Function to split datasets string into an array
split_datasets() {
    local datasets_str="$1"
    local delimiter=","
    local datasets_array

    IFS="$delimiter" read -ra datasets_array <<< "$datasets_str"

    echo "${datasets_array[@]}"
}

# Iterate through each subfolder in the predefined list
for subfolder_datasets in $datasets; do
    IFS=":" read -r subfolder datasets_str <<< "$subfolder_datasets"
    datasets_array=( $(split_datasets "$datasets_str") )

    # Iterate through each dataset in the current subfolder
    for dataset in "${datasets_array[@]}"; do
        input_dir="$DATA_PATH/$subfolder/original/$dataset"
        echo "$input_dir"
        # Check if the input_dir exists and is a directory
        if [ -d "$input_dir" ]; then
            for dataset_pseudo in $dataset_pseudo_names; do
                IFS=":" read -r dataset_name pseudo_name <<< "$dataset_pseudo"

                if [[ "$dataset" == "$dataset_name" ]]; then
                    echo "Converting dataset: $dataset (Pseudo name: $pseudo_name)"
                    python "${pseudo_name}_converter.py" --input_dir "$input_dir" --output_dir "$DATA_PATH/$subfolder/converted/"
                    break
                fi
            done
        else
            echo "Dataset $dataset not found in $input_dir"
        fi
    done
done
