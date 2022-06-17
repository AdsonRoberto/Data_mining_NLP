
#Explorando os logs
#Metricas
 #Nº mensagens do STUART e alunos
 #Distribuição de mensagens por alunos
 #Distribuição de quantidade de mensagens
 #Nº PCD
 #Proporção STUART/ALUNOS
 #Top mensagens do STUART
 #Série temporal mensagens de alunos/STUART (linhas separadas)
 #Análise linguística de mensagens de alunos
 #Análise do diálogo
 #Agradecimentos ao STUART
 #Análise de sentimento

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!pip install unidecode

!python -m spacy download pt_core_news_sm
from unidecode import unidecode 

!python3 -m spacy download pt

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re, os
from statsmodels.tsa.seasonal import seasonal_decompose

%matplotlib inline
sns.set(style="whitegrid")
#pd.set_option('display.max_colwidth', None)

#load all files
files_path = '/content/drive/MyDrive/Mineracao/Logs' 
files_list = os.listdir(files_path)
dfs = []
for log in files_list:
  print(log)
  file_path = files_path + '/' + log
  df = pd.read_csv(file_path, error_bad_lines=False, sep=',')
  dfs.append(df)
df = pd.concat(dfs)

# ordem cronológica
df['data_hora_mensagem'] = pd.to_datetime(df['data_hora_mensagem'])
df = df.sort_values(by='data_hora_mensagem')
df.head(3)

df.sample(10)

print(len(df))

# create vertical barchart with annotation
def annotate_barchart(values, labels, title, size = (8,5), col = None, rotate_xticks = False):
  plt.figure(figsize = size)
  plt.title(title)
  if rotate_xticks:
    plt.xticks(values, labels, rotation='vertical')
  if type(col) == str or col == None:
    g = sns.barplot(x=labels, y=values, color = col)
  elif type(col) == list:
    g = sns.barplot(x=labels, y=values, palette = col)
    
  for p in g.patches:
      g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                                p.get_height()), ha = 'center', 
                va = 'center', xytext = (0, 5), textcoords = 'offset points')
  plt.show()

#clrs = ['grey' if (x < max(values)) else 'red' for x in values ]
#sb.barplot(x=idx, y=values, palette=clrs) # color=clrs)

# create donnut chart
def donnut(values, labels, size = (10,8), col = None):
  plt.figure(figsize=(8, 8))
  my_circle = plt.Circle( (0,0), 0.6, color='white')
  plt.pie(values, labels=labels, autopct='%1.1f%%',startangle=90, pctdistance=0.4, colors = col)
  p = plt.gcf()
  p.gca().add_artist(my_circle)
  plt.show()


#Qte de msg
size = len(df)
nans = df['mensagem'].isna().sum()
nan_ratio = (nans/size)*100 

print('Total de mensagens: {a:1d}'.format(a=size))
print('Total de mensagens nulas: {a:1d}'.format(a=nans))
print('Porcentagem de mensagens nulas: {a:.2f}%'.format(a=nan_ratio))