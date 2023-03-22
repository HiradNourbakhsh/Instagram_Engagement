#!/usr/bin/env python
# coding: utf-8

# In[17]:


df = pd.read_excel('/Users/hiradnourbakhsh/INSY_670/Individual_assignment/Insta_download.xlsx')


# In[18]:


df = df.drop(columns = ['Caption', 'Comments'])
df


# In[19]:


df.to_excel('/Users/hiradnourbakhsh/Desktop/INSY 670/Individual Assignment/image_url.xls', index = False)


# In[21]:


import xlrd
from google.cloud import vision
import os
import pandas as pd

Application_Credentials = 'creds.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Application_Credentials
client = vision.ImageAnnotatorClient()
image = vision.Image()

loc = ("/Users/hiradnourbakhsh/Desktop/INSY 670/Individual Assignment/image_url.xls")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
df = pd.DataFrame()
# loop through every url, retrieve the image and send to google vision
for i in range(sheet.nrows):
    image_src_temp = sheet.cell_value(i, 0)
    image.source.image_uri = image_src_temp
    response = client.label_detection(image=image)
    labels = response.label_annotations
    l = []
    for label in labels:
        l.append(label.description)
    s = ' '.join(l)
    print("s")
    print(s)
    df = df.append({'URL': image_src_temp, 'Labels': s}, ignore_index=True)
df.to_excel("GV_Output.xlsx",index=False)


# In[20]:


pd.read_excel('/Users/hiradnourbakhsh/Desktop/INSY 670/Individual Assignment/image_url.xls')

