
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