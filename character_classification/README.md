# img2latex: character classification 

This module explore the character classification part of img2latex. 

## Deployment:
We have made a flask app that takes input a image and outputs the corresponding character in the image. The flask app can be found at `./app.py`.   
The app accepts POST requests on `/predict`. The input image should be in `PIL` format.   
The predicts are given in form of a json. If `json['success']==true`, then `json['predictions']` contains the predicted symbol.  

## Training
Training information can be found in `./notebooks/character_classification.ipynb`

## Dataset Used
- https://www.kaggle.com/xainano/handwrittenmathsymbols
- https://www.nist.gov/srd/nist-special-database-19


