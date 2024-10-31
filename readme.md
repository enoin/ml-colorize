

## create conda env

    conda create --name ml-colorize --file requirements.txt

or

    conda env create -f environment.yml


## export installed packages

    conda list -e > requirements.txt
    conda env export > environment.yml 


## run
    conda activate ml-colorize 
    python main.py train
    python main.py test
