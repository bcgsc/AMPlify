# AMPlify

AMPlify is an attentive deep learning model for antimicrobial peptide prediction.

For more information, please refer to:

### Dependencies
* Python 3
* Keras
* Numpy
* Pandas
* Scikit-learn
* Biopython

### Train
Usage: 'python train_amplify.py [-h] -amp_tr AMP_TR -non_amp_tr NON_AMP_TR [-amp_te AMP_TE] [-non_amp_te NON_AMP_TE] -out_dir OUT_DIR -model_name MODEL_NAME'
'''
optional arguments:
  -h, --help            show this help message and exit
  -amp_tr AMP_TR        Training AMP set, fasta file
  -non_amp_tr NON_AMP_TR
                        Training non-AMP set, fasta file
  -amp_te AMP_TE        Test AMP set, fasta file, optional
  -non_amp_te NON_AMP_TE
                        Test non-AMP set, fasta file, optional
  -out_dir OUT_DIR      Output directory
  -model_name MODEL_NAME
                        File name of trained model weights
'''

### Predict
Usage: 'python AMPlify.py [-h] -md MODEL_DIR [-m MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME] -s SEQS [-od OUT_DIR] [-of OUT_FORMAT]'
'''
optional arguments:
  -h, --help            show this help message and exit
  -md MODEL_DIR, --model_dir MODEL_DIR
                        Directory of where models are stored
  -m MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME, --model_name MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME MODEL_NAME
                        File names of 5 trained models
  -s SEQS, --seqs SEQS  Sequences for prediction, fasta file
  -od OUT_DIR, --out_dir OUT_DIR
                        Output directory, optional
  -of OUT_FORMAT, --out_format OUT_FORMAT
                        Output format, txt or xlsx, optional
'''

### Author

Chenkai Li (cli@bcgsc.ca)

### Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact us.
