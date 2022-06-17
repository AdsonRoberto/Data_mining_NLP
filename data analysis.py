
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

df[df['autor_da_mensagem']!='STUART']['remetente'].nunique()

#Remetentes
stuart_msgs = len(df[df['autor_da_mensagem']=='STUART'])
users_msgs = len(df[df['autor_da_mensagem']=='USUÁRIO'])

#plt.style.use('default')
plt.style.use('bmh')
values = [stuart_msgs, users_msgs]
labels = ['STUART','Usuários']

annotate_barchart(values,labels,title='Quantidade de mensagens')
donnut(values, labels)
values

active_students = set(df[df['autor_da_mensagem']!='STUART']['remetente'])
total_studentes = active_students.union(set(df[df['autor_da_mensagem']=='STUART']['destinatario']))
values = [len(active_students), len(total_studentes)]
labels = ['alunos que enviaram mensagens para o STUART', 'alunos que o stuart interagiu']
donnut(values, labels)

print('mensagens únicas do stuart')
df[df['autor_da_mensagem']=='STUART']['mensagem'].nunique()

#Mensagens enviadas por estudantes

students_series = df.groupby('remetente').count().sort_values(by='autor_da_mensagem', ascending=False)['autor_da_mensagem'][1:]
print('Quantidade de estudantes:',len(students_series))
print('Média de mensagens por estudante:',students_series.mean())

top_students = students_series[0:100]
students_IDs = list(top_students.index)
labels = [str(a) for a in students_IDs]
values = top_students.values
annotate_barchart(values, labels, title = 'Top 100 estudantes que enviaram mais mensagens', size = (30,10), col='C9', rotate_xticks=True)


ids = students_series.index
n_messages = students_series.values
deficiency = []
pcd_bool = []
for id in ids:
  id_def = df[df['remetente']==id]['deficiencia'].unique()[0]
  deficiency.append(id_def)
  if id_def == 'Nenhuma':
    pcd_bool.append(False)
  else:
    pcd_bool.append(True)

df_students = pd.DataFrame({'id':ids, 'número de mensagens': n_messages, 'deficiencia': deficiency, 'PcD': pcd_bool})

# swarmplot
#sns.set(style="darkgrid")

#plt.style.use('default')
#plt.style.use('bmh')

plt.figure(figsize=(15,10))
plt.title('Quantidade de mensagens por aluno')
sns.swarmplot(x="PcD", y="número de mensagens", palette=['C0','C1'],size=6, data=df_students) #['C2','C5']


#Estudante que mais enviou msg p chatbot
top_student = df[df['remetente']==students_IDs[1]] 
top_student.head(1)

print('Mensagens quebradas')
list(top_student['mensagem'])

#id tutores
tutores = {355074062, 276365414, 395739605}
daniel = 395739605

#pcd que mais enviou msg
pcd_series = df[df['deficiencia'] != 'Nenhuma'].groupby('remetente').count().sort_values(by='autor_da_mensagem', ascending=False)['autor_da_mensagem']
top_student_pcd = df[df['remetente']==pcd_series.index[0]] 
top_student_pcd.head(1)

list(top_student_pcd['mensagem'])

# plot all together

#plt.style.use('default')
plt.style.use('bmh')
sns.set(style="darkgrid")

# prepare data
df_students_msgs = df[df['autor_da_mensagem'] !='STUART'] 

# students
pcds = df_students_msgs[df_students_msgs['deficiencia'] != 'Nenhuma']
not_pcd = df_students_msgs[(df_students_msgs['deficiencia'] == 'Nenhuma')]
n_msgs = [pcds['remetente'].nunique(), not_pcd['remetente'].nunique()]
pcd_label = ['PcD','Não PcD']

# type of deficency
deficiency_type = pcds['deficiencia'].value_counts()
def_count = []
for deficiency in deficiency_type.index:
  count = pcds[pcds['deficiencia']==deficiency]['remetente'].nunique()
  def_count.append(count)

def_type = deficiency_type.index

# messages
pcd_msgs = len(df_students_msgs[df_students_msgs['deficiencia'] != 'Nenhuma'])
non_pcd_msgs = len(df_students_msgs[(df_students_msgs['deficiencia'] == 'Nenhuma')])
msgs_pcd = [pcd_msgs, non_pcd_msgs]
msgs_pcd_labels = ['PcD','Não PcD']

# subplots
plt.figure(figsize=(15,12))

# barras alunos
plt.subplot(321)
plt.title('Alunos')
g = sns.barplot(x=pcd_label, y=n_msgs, palette= ['C2','C5'])
for p in g.patches:
    g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                               p.get_height()), ha = 'center', 
               va = 'center', xytext = (0, 5), textcoords = 'offset points')

# barras mensagens
plt.subplot(322)
plt.title('Mensagens')
g = sns.barplot(x=msgs_pcd_labels, y=msgs_pcd, palette= ['C6','C7'])
for p in g.patches:
    g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                               p.get_height()), ha = 'center', 
               va = 'center', xytext = (0, 5), textcoords = 'offset points')

#donnut alunos
plt.subplot(323)
my_circle=plt.Circle( (0,0), 0.6, color='white')
plt.pie(n_msgs, labels=pcd_label, autopct='%1.1f%%',startangle=90, pctdistance=0.4, colors = ['C2','C5'])
p=plt.gcf()
p.gca().add_artist(my_circle)

# donnut mensagens
plt.subplot(324)
my_circle=plt.Circle( (0,0), 0.6, color='white')
plt.pie(msgs_pcd, labels=msgs_pcd_labels, autopct='%1.1f%%',startangle=90, pctdistance=0.4, colors = ['C6','C7'])
p=plt.gcf()
p.gca().add_artist(my_circle)

# barras deficiencia alunos
plt.subplot(325)
plt.xticks(values, labels, rotation='vertical')
plt.title('Quantidade de alunos por tipo de deficiência')
g = sns.barplot(x=def_type, y=def_count, color= 'C4')
for p in g.patches:
    g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                               p.get_height()), ha = 'center', 
               va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
# barras deficiencia mensagens
plt.subplot(326)
plt.xticks(values, labels, rotation='vertical')
plt.title('Quantidade de mensagens por tipo de deficiência')
g = sns.barplot(x=deficiency_type.index, y=deficiency_type.values, color= 'C1')
for p in g.patches:
    g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                               p.get_height()), ha = 'center', 
               va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.2, 
                    hspace=0.1)    
plt.show()

sum =  df_students[df_students['PcD']==False]['número de mensagens'].sum()
stu = df_students[df_students['PcD']==False]['id'].nunique()
print(sum,stu)

pcd_type = deficiency_type.to_frame(name='mensagens')
pcd_type['alunos'] = def_count
non_pcd = pd.DataFrame({'mensagens':sum, 'alunos': stu},index = ['Não-PcD'])
pcd_type = pcd_type.append(non_pcd)
pcd_type['mensagens/aluno'] = pcd_type['mensagens']/pcd_type['alunos']
pcd_type.sort_values(by='alunos', ascending=False).round(1)

plt.style.use('default')
#plt.style.use('bmh')
sns.set(style="darkgrid")

pcd_type = deficiency_type.to_frame(name='mensagens')
pcd_type['alunos'] = def_count
non_pcd = pd.DataFrame({'mensagens':sum, 'alunos': stu},index = ['Não-PcD'])
pcd_type = pcd_type.append(non_pcd)
pcd_type['mensagens/aluno'] = pcd_type['mensagens']/pcd_type['alunos']
pcd_type = pcd_type.sort_values(by='mensagens/aluno', ascending=False).round(1)
pcd_type.reset_index(inplace=True)
cols = list(pcd_type.columns)
cols[0] = 'Tipo de deficiência'
pcd_type.columns = cols
#pcd_type
plt.figure(figsize=(10,4))
plt.xticks(values, labels, rotation='vertical')
g = sns.barplot(x='Tipo de deficiência',y = 'mensagens/aluno',data=pcd_type, color = 'C1')
for p in g.patches:
    g.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., 
                                               p.get_height()), ha = 'center', 
               va = 'center', xytext = (0, 5), textcoords = 'offset points')

plt.figure(figsize=(10,4))
plt.xticks(values, labels, rotation='60')
g = sns.barplot(y='Tipo de deficiência',x = 'mensagens/aluno',data=pcd_type, color = 'C1')
for p in g.patches:
        g.annotate(format(p.get_width(), '.1f'), (p.get_width(), 
                                                    p.get_y() + p.get_height()/2.), ha = 'center', 
                    va = 'center', xytext = (15, 0), textcoords = 'offset points')

# estatisticas
df_students_statistics = df_students[df_students['PcD']==False].describe()['número de mensagens'].to_frame()
df_students_statistics['PcD'] = df_students[df_students['PcD']==True].describe()['número de mensagens']
df_students_statistics.columns = ['Não-PcD','PcD']
df_students_statistics.round(2)
#df_students_statistics = df_students_statistics.transpose()
#df_students_statistics.style.background_gradient(cmap ='Blues')

plt.figure(figsize=(8,8))
plt.title('Distribuições de mensagens por aluno')
sns.boxplot(x="PcD", y="número de mensagens", palette=['C0','C1'], data=df_students) #['C2','C5']