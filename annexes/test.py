# exploring data

# extract key info from receipts :
# extract and distinguish two lines : TTC (to pay) and kms (browsed)

# to that, we could use NER method (Named Entity Recognition) based on exploring the semantic context
# in text sequences (TTC and km..)

# here, we will try to use the effective information from both semantic context and spatial distribution
# of texts.

# CUTIE (Convo Universal Text Information Extractor) : consist on applying a CNN on gridded texts where
# texts are embedded as features with semantical connotations "sur les zones de textes quadrillées,
# où les textes sont integrés comme des caractéristiques ayant des connotations sémantiques"

# why ? To employ NER models, text words in the original document are aligned as a long paragraph based
# on a line-based rule. => involve the spatial info 

# the line-based feature extraction method can not achieve its best performance when document texts
# are not perfectly aligned. Moreover the RNN-based classifier, bi-directional LSTM model in CloudScan, 
# has limited ability to learn the relationship among distant words.

# Bidirectional Encoder Representations from Transformers (BERT) is a recently proposed model that is pre-trained
# on a huge dataset and can be fine-tuned for a specific task, including Named Entity Recognition (NER),
# which outperforms most of the state of the art results in several NLP tasks Since the previous learning based 
# methods treat the key information extraction problem as a NER problem, applying BERT can achieve a better result
# than the bi-LSTM in CloudScan.

# method to creat grid data for model training

# grid positional mapping : use OCR to aquire texts and theire absolut/relative position.
# on considere l'image I(w,h), le cadre minimal autour du ième texte "si" est noté "bi", le cadre bi est restreint
# par 2 points : le point supérieur à gauche (xi_left,yi_top) et le point inférieur à droite (xi_right,yi_bottom).
# pour éviter le chauvauchement des cadres, et avoir la position relative réelle entre les textes, on considère le
# centre du ième cadre : (ci_x,ci_y)
# Dans le tableau (matrice) des mots, la zone de délimitation est divisée horizontalement en plusieurs zones, et
# la position des zones délimitées (ligne et colonne) sont calculées par les équations (1) et (2) : 

# CUTIE A VS CUTIE B







# image preparation and processing to build and train a CNN to

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt

# organiser les données dans des fichiers séparés (données training/validation)

os.chdir('C:/Users/issla/Desktop/data/large-receipt-image-dataset-SRD') # change directory
if os.path.isdir('train') is False :
	os.makedirs('train')
	os.makedirs('validation')

	for c in random.sample(glob.glob('*receipt.jpg'),160):
		shutil.move(c,'train')



data = 
# données (éparse) textes dissperssé dans le document, avec des echelles différentes => inclure la donnée spatiale + multi-scale
# ségmentation sémantique, intégrer une capacité de traitement du contexte sémantique à plusieurs échelles.
# pour ce faire, il existe 2 méthodes : "image pyramid" and "the encoder-decoder structure"de la structure de l'encodeur-décodeur
# (les -) la résolution spatiale est réduite dans le processus d'encodage, et le processus de décodage n'exploite que les caractéristiques
# de haute résolution mais de bas niveau pour récupérer l'info spatiale,
# (les +) champs de réception larges, charges de calcul réduite, capture des détails fins et récupre progressivement l'info spatiale

# instead, the field of view of filters can also be effectively enlarged and multi-scale contexts can be captured by combining 
# multi-resolution features, or by applying atrous convolution

# . To capture long distance connection and avoid potential information loss in the encoding process, we propose two different 
# network architectures and compare their performance in Section 4. In fact, we had experimented with various types of model 
# structures and only detail two of them here to avoid being a tedious paper. Specifically, the proposed CUTIE-A is a high 
# capacity convolutional neural network that fuses multi-resolution features without losing high-resolution features, the 
# proposed CUTIE-B is a convolutional network with atrous convtion for enlarging the field of view and Atrous Spatial Pyramid 
# Pooling (ASPP) module to capture multi-scale contexts.

# The cross entropy loss function is applied to compare the predicted token class grid and the ground truth grid.

# CUTIE-A

# AP and softAP

# learning rate of 1e-3 with Adam optimiser and step decay learning strategy The learning rate is dropped to 1e − 4 and 1e − 5
# on the 15, 000-th and 30, 000-th steps, the training is terminated within 40, 000 steps with batch size of 32.
#  Instance normalization and hard negative mining are involved to facilate training. . Dropout (l'abandon) is applied with
# keep probability of 0.9
# 75 % training and 25% test 
# The default embedding size is 128, target augmentation shape is 64 for both row and column. No post or pre-processing on CUTIE







