# AMPlify

AMPlify is an attentive deep learning model for antimicrobial peptide prediction.

For more information, please refer to the preprint: https://www.biorxiv.org/content/10.1101/2020.06.16.155705v1

### Dependencies
* CentOS Linux 7
* Python 3
* Keras 2.2.4
* Tensorflow 1.11/1.12/1.13
* Numpy
* Pandas
* Scikit-learn
* Biopython

No non-standard hardware required.

### Installation
Download the zip file from the [Releases](https://github.com/bcgsc/AMPlify/releases) section, and unzip the file to the chosen directory for the software installation.

Typical install time: < 1 minute.

### Datasets
Datasets for training and testing are stored in the `data` folder. Please specify the directory if you would like to use those datasets for training or testing the model.

### Pre-trained sub-models
Weights for 5 pre-trained sub-models are stored in the `models` folder. Please specify the directory if you would like to use those models for prediction.

### Train
Usage: `python train_amplify.py [-h] -amp_tr AMP_TR -non_amp_tr NON_AMP_TR [-amp_te AMP_TE] [-non_amp_te NON_AMP_TE] -out_dir OUT_DIR -model_name MODEL_NAME`
```
optional arguments:
  -h, --help            Show this help message and exit
  -amp_tr AMP_TR        Training AMP set, fasta file
  -non_amp_tr NON_AMP_TR
                        Training non-AMP set, fasta file
  -amp_te AMP_TE        Test AMP set, fasta file, optional
  -non_amp_te NON_AMP_TE
                        Test non-AMP set, fasta file, optional
  -out_dir OUT_DIR      Output directory
  -model_name MODEL_NAME
                        File name of trained model weights
```
Example: `python train_amplify.py -amp_tr ../data/AMP_train_20190414.fa -non_amp_tr ../data/non_AMP_train_20190414.fa -amp_te ../data/AMP_test_20190414.fa -non_amp_te ../data/non_AMP_test_20190414.fa -out_dir ../models/ -model_name model`

Expected output: 1) The model weights trained using the specified data; 2) Test set performance, if test sequences have been specified.

Typical run time on the entire training set of `AMP_train_20190414.fa` + `non_AMP_train_20190414.fa`: ~ 1 day.


### Predict
Usage: `python AMPlify.py [-h] -md MODEL_DIR [-m MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME] -s SEQS [-od OUT_DIR] [-of {txt,xlsx}]`
```
optional arguments:
  -h, --help            Show this help message and exit
  -md MODEL_DIR, --model_dir MODEL_DIR
                        Directory of where models are stored
  -m MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME, --model_name MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME
                        File names of 5 trained models (optional)
  -s SEQS, --seqs SEQS  Sequences for prediction, fasta file
  -od OUT_DIR, --out_dir OUT_DIR
                        Output directory (optional)
  -of {txt,xlsx}, --out_format {txt,xlsx}
                        Output format, txt or xlsx (optional)
```
Example: `python AMPlify.py -md ../models/ -s ../data/AMP_test_20190414.fa`

Expected output: Prediction scores, classes and attention scores of the input sequences.

Typical run time on the AMP test set `AMP_test_20190414.fa`: ~ 6 minutes.


### Author

Chenkai Li (cli@bcgsc.ca)

### Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact us.
