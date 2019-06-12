import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import string
from PIL import Image, ImageDraw, ImageFont,ImageFilter

# function to plot the convergence of accuracy in train set and test set
def convergence_plot(history_dict):
  fig, ax = plt.subplots(1,2, figsize=(20,7))
  ax[0].plot(history_dict['first_acc'], label = 'first')
  ax[0].plot(history_dict['second_acc'], label = 'second')
  ax[0].plot(history_dict['third_acc'], label = 'third')
  ax[0].plot(history_dict['fourth_acc'], label = 'fourth')
  ax[0].plot(history_dict['fifth_acc'], label = 'fifth')
  ax[0].plot(history_dict['sixth_acc'], label = 'sixth')
  ax[0].plot(history_dict['seventh_acc'], label = 'seventh')
  ax[0].plot(history_dict['eighth_acc'], label = 'eighth')
  ax[0].plot(history_dict['nineth_acc'], label = 'nineth')
  ax[0].plot(history_dict['tenth_acc'], label = 'tenth')
  ax[0].legend()
  ax[0].set_title('trainin set')

  ax[1].plot(history_dict['val_first_acc'], label = 'first')
  ax[1].plot(history_dict['val_second_acc'], label = 'second')
  ax[1].plot(history_dict['val_third_acc'], label = 'third')
  ax[1].plot(history_dict['val_fourth_acc'], label = 'fourth')
  ax[1].plot(history_dict['val_fifth_acc'], label = 'fifth')
  ax[1].plot(history_dict['val_sixth_acc'], label = 'sixth')
  ax[1].plot(history_dict['val_seventh_acc'], label = 'seventh')
  ax[1].plot(history_dict['val_eighth_acc'], label = 'eighth')
  ax[1].plot(history_dict['val_nineth_acc'], label = 'nineth')
  ax[1].plot(history_dict['val_tenth_acc'], label = 'tenth')
  ax[1].legend()
  ax[1].set_title('validation set')
  
# function to read all images from directory
def read_images_from_dir(blur_factor=1, path = None, height = 20, width =200):
  
  if path==None:
    file_path = 'data_mining_course/deep_learning/blur_factor_'+str(blur_factor)
    file_names = os.listdir(file_path)
    file_names = sorted(file_names)
    
  else:
    file_path = path + '/blur_factor_'+str(blur_factor)
    file_names = os.listdir(path)
    file_names = sorted(file_names)
  
  data = np.empty((len(file_names), height, width, 1))
  
  for i in range(len(file_names)):
    img = Image.open(file_path+'/'+file_names[i])
    data[i] = (1 - np.array(img).reshape(height, width, 1))/255.0
  
  return data

# function to convert the predicted array to text
def array_to_text(prediction, character_count=10):
  alphabet= string.ascii_uppercase # create a string series A-Z
  pred_len = len(prediction[0])
  prediction_text = []
  
  for pred in range(pred_len):
    text_series = ''
    for char in range(character_count):
      char_pos = prediction[char][pred].argmax() # get the max probability 
      text_series += alphabet[char_pos]
    prediction_text.append(text_series)
   
  return prediction_text


# function to take the prediction array and convert to data frame with desired output
def text_prediction(model, blur_factors=[1], path = None, height = 20, width =200, character_count=10):
  
  prediction_df = pd.DataFrame()
  
  for blur_factor in blur_factors:  
    if path==None:
      file_path = 'data_mining_course/deep_learning/blur_factor_'+str(blur_factor)
      file_names = os.listdir(file_path)
      file_names = sorted(file_names)

    else:
      file_path = path +'blur_factor_'+str(blur_factor)
      file_names = os.listdir(file_path)
      file_names = sorted(file_names)
    # load data
    test_data = read_images_from_dir(blur_factor=blur_factor, path = path, height = height, width =width)
    num_prediction = model.predict(x = test_data)
    text_prediction = array_to_text(prediction =num_prediction, character_count=character_count)
    file_text_df = pd.DataFrame({'file':file_names, 'text':text_prediction})

    # sort file name
    file_text_df = file_text_df.assign(num = file_text_df.file.apply(lambda x: int(re.split('\.',x)[0])))
    file_text_df = file_text_df.sort_values('num')
    file_text_df = file_text_df.drop(columns =['num'])
    file_text_df = file_text_df.assign(file = 'blur_factor_'+str(blur_factor)+'/'+file_text_df.file)
    # append to the prediction df
    prediction_df = prediction_df.append(file_text_df, ignore_index = True)
  
  prediction_df = prediction_df.reset_index(drop = True)
    
  return prediction_df