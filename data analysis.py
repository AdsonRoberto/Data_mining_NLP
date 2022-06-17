
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

#Mensagens enviadas pelo chatbot

import re
def generalize(text):
  # 307725127
  student_regex = '\d\d\d\d\d\d\d\d\d'
  return None

def truncate(text, threshold = 6):
  words = text.split()
  trunc_words = words[0:threshold]
  text = ' '.join(trunc_words)
  if len(words) > threshold:
     text = text + '...'
  return text

  sns.set(style="whitegrid")
plt.style.use('bmh')
df_stuart = df[df['autor_da_mensagem']=='STUART']
df_stuart['mensagem'] = [text.replace('CHAT.STUART_INFORMATION_USEFUL','Recomendação') for text in df_stuart['mensagem']]
#df_stuart['mensagem_geral'] = []

msgs_stuart = df_stuart['mensagem'].value_counts()
print('Total de mensagens únicas do STUART:', len(msgs_stuart))
top_stuart = msgs_stuart[1:10]
trunc_msg = [truncate(t,10) for t in top_stuart.index]

plt.figure(figsize=(8,8))
#plt.title('Mensagens mais frequentes do STUART')
g = sns.barplot(x=top_stuart.values,y=trunc_msg, color = 'C0')
for p in g.patches:
    g.annotate(format(p.get_width(), '.0f'), (p.get_width(), 
                                               p.get_y() + p.get_height()/2.), ha = 'center', 
               va = 'center', xytext = (15, 0), textcoords = 'offset points')

sns.set(style="whitegrid")
plt.style.use('bmh')
df_stuart = df[df['autor_da_mensagem']=='STUART']
df_stuart['mensagem'] = [text.replace('CHAT.STUART_INFORMATION_USEFUL','Recomendação') for text in df_stuart['mensagem']]
df_stuart['mensagem'] = [text.replace('CHAT.STUART_STUDENT_SATISFACTION','Recomendação') for text in df_stuart['mensagem']]
#df_stuart['mensagem_geral'] = []

msgs_stuart = df_stuart['mensagem'].value_counts()
print('Total de mensagens únicas do STUART:', len(msgs_stuart))
top_stuart = msgs_stuart[5:15]
trunc_msg = [truncate(t,10) for t in top_stuart.index]

plt.figure(figsize=(8,8))
#plt.title('Mensagens mais frequentes do STUART')
g = sns.barplot(x=top_stuart.values,y=trunc_msg, color = 'C0')
for p in g.patches:
    g.annotate(format(p.get_width(), '.0f'), (p.get_width(), 
                                               p.get_y() + p.get_height()/2.), ha = 'center', 
               va = 'center', xytext = (15, 0), textcoords = 'offset points')
              
top_stuart

#analise das respostas de duvidas de conteudo dos alunos
counter = 0
for msg in list(df_stuart['mensagem']):
  if 'se quiser saber mais' in msg.lower() or 'veja' in msg.lower() or 'look what'in msg.lower():
    counter+=1
    print(msg)

print(counter)
print('Dúvidas sobre conteúdo foram utilizadas {a:1d} vezes'.format(a=counter))
print(counter/len(df)*100)

#Serie temporal

df['timestamp'] = pd.to_datetime(df['data_hora_mensagem']) #, format='%d/%m/%y %H:%M')
frame = '24H'
timeseries = df[df['autor_da_mensagem']!='STUART'].groupby('timestamp').count()['remetente']
timeseries = timeseries.resample(frame).sum()

print('Total de dias: ', len(timeseries))
print('Média de mensagens enviadas para o STUART por dia:',timeseries.mean())

fig = px.line(x=timeseries.index, y=timeseries.values)
fig.show()

df.sample(3)

# stuart and students
timeseries_stuart = df[df['autor_da_mensagem'] =='STUART'].groupby('timestamp').count()['remetente'].resample(frame).sum().to_frame()
timeseries_users = df[df['autor_da_mensagem'] !='STUART'].groupby('timestamp').count()['remetente'].resample(frame).sum().to_frame()
df_timeseries = pd.concat([timeseries_stuart,timeseries_users])
df_timeseries['autor'] = ['STUART']*len(timeseries_stuart) + ['USUÁRIOS']*len(timeseries_users)
df_timeseries.reset_index(level=0, inplace=True)

print('Média de mensagens enviadas pelo STUART por dia:',timeseries_stuart.mean())
print('Média de mensagens enviadas por alunos por dia:',timeseries_users.mean())

fig = px.line(df_timeseries, x='timestamp', y='remetente',
              color="autor", line_group="autor", hover_name="autor")
fig.show()

#Análise linguística das mensagens enviadas por alunos

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import string
import spacy

nltk.download('stopwords')
#!python -m spacy download pt

punct = list(string.punctuation)

def get_top_n_ngrams(corpus, ngram = (1,1), n=None, reverse = True):
    if type(corpus) == str:
        corpus = [corpus]
    vec = CountVectorizer(ngram_range = ngram).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=reverse)
    return words_freq[:n]

def build_ngram_df(corpus, ngram = (1,1), n=None, ascend = False):
    reverse = not(ascend)
    ngrams = get_top_n_ngrams(corpus, ngram, n, reverse = reverse)
    df1 = pd.DataFrame(ngrams, columns = ['ngrams' , 'count'])
    df1 = df1.groupby('ngrams').sum()['count'].sort_values(ascending=ascend)
    return df1

def plotNgrams(ngrams, col = 'C0', orientation = 'vertical'):
    labels = list(ngrams.index)
    values = list(ngrams.values)
    #
    if orientation == 'vertical':
        g = sns.barplot(y=labels, x=values, color = col)
        for p in g.patches:
            g.annotate(format(p.get_width(), '.0f'), (p.get_width(), 
                                                       p.get_y() + p.get_height()/2.), ha = 'center', 
                       va = 'center', xytext = (15, 0), textcoords = 'offset points')
    else:
        plt.xticks(values, labels, rotation='vertical')
        g = sns.barplot(x=labels, y=values, color = col) 
        for p in g.patches:
            g.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., 
                                                       p.get_height()), ha = 'center', 
                       va = 'center', xytext = (0, 5), textcoords = 'offset points')
            
# remoção de stop words
stop_words = list(stopwords.words('portuguese'))
stop_words = stop_words + ['pra', 'como', 'ola', 'olá', 'oi', 'oii'] #, 'obrigado', 'obrigada','ok']

def remove_punctuation(text):
  for p in punct:
    text = text.replace(p,'')
  return text
  
def remove_stopwords(text):
    words = text.split()
    # [f(x) for x in sequence if condition]
    words = [w for w in words if w not in stop_words]
    text = ' '.join(words)
    return text

def remove_greetings(text):
  greetings_list = ['bom dia','boa tarde', 'boa noite', 'thank you']
  for g in greetings_list:
    text = text.replace(g,'')
  return text

#lemmatization
# nlp = spacy.load('pt')
def lemmatization(text):    
    doc = nlp(text)
    for token in doc:
        if token.text != token.lemma_:
            text = text.replace(token.text, token.lemma_)
    return text



def preprocess(text):
  text = text.lower()
  text = remove_punctuation(text)
  #text = remove_greetings(text)
  text = remove_stopwords(text)
  #text = lemmatization(text)
  return text

msgs_students = df[df['autor_da_mensagem'] !='STUART']['mensagem'].dropna()

preprocessed = [preprocess(t) for t in msgs_students]

total = ' '.join(list(preprocessed))
plt.figure(figsize=(20, 10))
plt.title('Top palavras mais frequentes nas mensagens enviadas por alunos')
total_bigrams = build_ngram_df(total, ngram = (1,1), n=20)
plotNgrams(total_bigrams, col='C7', orientation = 'vertical')
plt.show()

msgs_students = df[df['autor_da_mensagem'] !='STUART']['mensagem'].dropna()

preprocessed = [preprocess(t) for t in msgs_students]

total = ' '.join(list(preprocessed))
print('palavras usadas somente uma vez por alunos')
#plt.figure(figsize=(20, 10))
#plt.title('Palavras menos frequentes nas mensagens enviadas por alunos')
total_bigrams = build_ngram_df(total, ngram = (1,1), n=30, ascend=True)
#total_bigrams
#plotNgrams(total_bigrams, col='C8', orientation = 'vertical')
#plt.show()

# analise de mensagens de gratiluz e dúvidas
import string 
punct = list(string.punctuation)
#punct
import pt_core_news_sm
nlp = pt_core_news_sm.load()

def remove_punct(text):
  for p in punct:
    text = text.replace(p,'')
  return text


def clean_text(text, punctuation = False):
  if punctuation:
    text = remove_punct(text) 
  text = text.lower()
  text = unidecode(text)
  text = text.strip()
  return text

def is_query(text):
  words = text.split()
  if len(words) > 4 or len(words) < 1:
    return False

  if greeting(text) or understood(text) or appreciation(text) or negative(text):
    return False

  doc = nlp(text)
  counter_noun = 0
  for token in doc:
    if token.pos_ == 'NOUN':
      counter_noun += 1
  noun_ratio = counter_noun/len(words)
  if noun_ratio >= 0.5:
    return True
  else:
    return False


def is_thanks(text):
  thanks_list = ['brigad','vlw','valeu','obg','thank','obrigdo','obrigada', 'thx']
  og_text = text
  text = clean_text(text)
  for th in thanks_list:
    if th in text and 'é obrigado' not in og_text: 
      return True
  return False

def appreciation(text):
  appreciation_list = ['!',
                       'legal',
                       'otimo',
                       'bacana',
                       'show',
                       'muito bom', 
                       ':)', 
                       '^^', 
                       'que bom', 
                       'ebaa', 
                       'uhu', 
                       'kkk', 
                       '\o/', 
                       's2', 
                       ';)', 
                       'fofo', 
                       ':-)', 
                       ':d',
                       ';)']

  thanks_list = ['brigad',
                 'vlw',
                 'valeu',
                 'obg',
                 'thank',
                 'obrigdo',
                 'obrigada', 
                 'thx']

  appreciation_list += thanks_list
  text = clean_text(text)
  for ap in appreciation_list:
    if ap in text and 'é obrigado' not in text: #and len(text.split() < 5): 
      return True
  return False

def interrogation(text):
  doubt_list = ['?', 
                'queria saber', 
                'como fa', 
                'o que', 
                'qual', 
                'quanto', 
                'quando', 
                'que horas', 
                'como', 
                'oque', 
                'quais', 
                'onde', 
                'how', 
                'what is',
                'quantas',
                'tem mais',
                'quem',
                'duvida']
  cl_text = clean_text(text)
  for db in doubt_list:
    if db in cl_text: 
      return True  
  # nouns
  return is_query(cl_text)

def understood(text):
  oks = ['ok', 
         'tudo bem', 
         'certo', 
         'sim', 
         'yes', 
         'positivo', 
         'ah sim', 
         'ah', 
         'blz', 
         'beleza', 
         'eh', 
         'entend',
         '👍',
         'isso']

  text = clean_text(text, True)
  for ok in oks:
    if ok in text and len(text.split()) < 5: 
      return True
  return False

def greeting(text):
  greetings = ['ola',
               'oi',
               'oii',
               'hello', 
               'bom dia', 
               'boa tarde', 
               'bom dai', 
               'boa noite', 
               'tchau', 
               'bye', 
               'eae',
               'abraco',
               'te mais']

  text = clean_text(text, True)
  for ap in greetings:
    if ap in text and len(text.split()) < 5: 
      return True
  return False

def negative(text):
  negatives = ['horr',
               ':(', 
               'ruim', 
               'pessimo', 
               'burr', 
               'triste', 
               'incapaz', 
               'bugado', 
               'dificil', 
               'insensivel', 
               'cansei', 
               'cansado',  
               'nao gosto', 
               'nao quero', 
               '¬¬',
               'desanimado',
               'estou mal',
               'sinto mal',
               'estress',
               'burocra']

# 'oque é uma notação hexadecimal?

  text = clean_text(text)
  for neg in negatives:
    if neg in text:
      return True
  return False

def trouble(text):
  negatives = ['erro',
              'nao consig',              
              'nao estou conseguindo',
              'sistema esta fora',
              'travou',
              'nao recebi',
              'nao entend',
              'nao compreendi',
              'erro',
              'errad',
              'bug',
              'pane',
              'problema', 
              'dificuldade',              
              'nao vou conseguir',
              'atrasa',
              'nao funciona',
              'forma errada',
              'nao to conseguindo',
              'nao consegui',
              'incorret',
               'nao entendi']
  text = clean_text(text)
  for neg in negatives:
    if neg in text:
      return True
  return False
  
  

    
df_students_msgs = df[df['autor_da_mensagem'] !='STUART']
#df_students_msgs['agradecimento'] = [is_thanks(text) for text in df_students_msgs.loc[:,'mensagem']]
df_students_msgs['dúvida'] = [interrogation(text) for text in df_students_msgs['mensagem']] 
df_students_msgs['apreciação'] = [appreciation(text) for text in df_students_msgs['mensagem']]
df_students_msgs['depreciação'] =  [negative(text) for text in df_students_msgs['mensagem']]
df_students_msgs['saudação'] = [greeting(text) for text in df_students_msgs['mensagem']]
df_students_msgs['compreensão'] = [understood(text) for text in df_students_msgs['mensagem']]
df_students_msgs['problemas'] = [trouble(text) for text in df_students_msgs['mensagem']]

# apreciação
print(df_students_msgs['apreciação'].sum())
df_students_msgs[df_students_msgs['apreciação']]['mensagem'].sample(10).values
#list(df_students_msgs[df_students_msgs['apreciação']]['mensagem'])

# depreciação
print(df_students_msgs['depreciação'].sum())
df_students_msgs[df_students_msgs['depreciação']]['mensagem'].sample(20).values

# dúvidas
print(df_students_msgs['dúvida'].sum())
df_students_msgs[df_students_msgs['dúvida']]['mensagem'].sample(20).values

# trouble
print(df_students_msgs['problemas'].sum())
df_students_msgs[df_students_msgs['problemas']]['mensagem'].sample(20).values

print('saudação:',df_students_msgs['saudação'].sum())
df_students_msgs[df_students_msgs['saudação']]['mensagem'].sample(10).values

print('compreensão:',df_students_msgs['compreensão'].sum())
df_students_msgs[df_students_msgs['compreensão']]['mensagem'].sample(10).values

"""
Dúvidas peculiares
Temas mais frequentes:
Média/nota mínima
Certificado
Datas e prazos
Resiliência
Dúvidas sobre conteúdos dos cursos
Erros no sistema ('Eu fiz a avaliação da aula 1.3 e não liberou a próxima aula, qual o motivo?')
Usabilidade
'Por que no Smartphone é mais difícil de acessar as aulas ?'
'Oi boa tarde! Tem intérprete? Ou eu ir lá Dell ou ficar casa câmera web?'
Corações apaixonados
'vc é casado ?'
'né que eu amo a Helena?'
'vamos ser amigos?'
Privacidade hackeada
'você é monitorado?'
STUART, a calculadora
'quanto é a integral de uma função do tipo f(x)=2x + 1 ?'
'quanto é 2+2'
Que deselegante
'e dai?'
'meu *au é grande?'
Pesado
'Você é programado para falar sobre suicidio?'

"""
df_students_msgs.columns

labels = ['apreciação','depreciação','dúvida', 'saudação', 'compreensão', 'problemas']
values = []
outros = pd.Series([True]*len(df_students_msgs))
for l in labels:
  value = df_students_msgs[l].sum()
  values.append(value)
  outros = outros & (df_students_msgs[l]==False).reset_index(drop=True)

labels.append('outros')
values.append(len(df_students_msgs[outros.values]))

s = pd.Series(values, index=labels).sort_values(ascending=False)
s

annotate_barchart(s.values, s.index, title = None, size = (10,5), col='C6', rotate_xticks=False)

outros_series = df_students_msgs[outros.values]['mensagem']
print(len(outros_series))
outros_series.sample(20).values

request = df_students_msgs['dúvida'].reset_index(drop=True)  | df_students_msgs['problemas'].reset_index(drop=True)
solved =  df_students_msgs['apreciação'].reset_index(drop=True)  |  df_students_msgs['compreensão'].reset_index(drop=True) 
not_solved =  df_students_msgs['depreciação'].reset_index(drop=True) 

val = [request.sum(),solved.sum(),not_solved.sum()]
labels = ['Solicitação', 'Contentamento', 'Descontentamento']
annotate_barchart(val, labels, title = None, size = (10,5), col='C7', rotate_xticks=False)

req = df_students_msgs[list(request)]['mensagem']
sol = df_students_msgs[list(solved)]['mensagem']
not_sol = df_students_msgs[list(not_solved)]['mensagem']
print('Exemplos de mensagem de solicitação:')
print(list(req.sample(10)))
print()
print('Exemplos de mensagem de solicitação solucionada:')
print(list(sol.sample(10)))
print()
print('Exemplos de mensagem de descontentamento:')
print(list(not_sol.sample(10)))
print()

for p in list(sol.sample(100)):
  print(p)


#Amostragem de dúvidas

sample = list(req.sample(116))
import pickle

"""
Outros tipos de mensagens interessantes
Aniversários errados
'hoje não é meu aniversario stuart',
'relaxa meu brother'
Não pode ofender o coleguinha
'stuart burro'
'parece o miguel'
'vou te chamar de eduardo',
'não quero mais',
'não gosto de robos'
'Burro'
Bad vibes
'estou triste',
'você é insensível'
'stuart eu tirei 2 na avaliação por não estudar :('
'eu nao entendi minha aula',
'mas nao eh um problema',
'eu sou burra',
'mas nao falei de vocec',
'desculpa',
'pois me ajuda',
'eu nao entendi minha aula ;-;',
'eu nao compreendi minha aula',
'minha aula esta dificil',
'to triste',
'eu sou incapaz',
Caoma senhora
'MEU DEUS ME DEIXA ESCREVERRRRRRRRRR'
'AQUI NAO VAI'
'ESSE CHAT É HORRIVEL'
'NAO CONSIGO ENVIAR UMA MENSAGEM SIMPLES'
STUART não entende risadas
'huahuahuha'
'eu estava rindo'
Só elogios
'Ai q fofo'
'Você é bonito '
'brabo dms esse bot em'
Observação: muitos usuários conversam em mensagens "quebradas"

"""

doubts = df_students_msgs[df_students_msgs['dúvida']]['mensagem']
doubts = df_students_msgs['mensagem']

preprocessed = [preprocess(t) for t in doubts]


total = ' '.join(list(preprocessed))
total = total.lower()
total = unidecode(total)

# agradecimentos
total = total.replace('obrigado','')
total = total.replace('obrigada','')

# pergunta
total = total.replace('o que é','')
total = total.replace('quais','')
total = total.replace('onde','')
total = total.replace('quanto','')
total = total.replace('conseguir','')
total = total.replace('consigo','')
total = total.replace('posso','')
total = total.replace('fazer','')
total = total.replace('faco','')
total = total.replace('sobre','')
total = total.replace('gostaria','')
total = total.replace('saber','')
total = total.replace('significa','')
total = total.replace('boa tarde','')
total = total.replace('bom dia','')
total = total.replace('boa noite','')
total = total.replace('sim','')
total = total.replace('vou','')
total = total.replace('nao','')
total = total.replace('ok','')
total = total.replace('ainda','')
total = total.replace('vai','')
total = total.replace('stuart','')

total = [t if len(t) > 3 else '' for t in total.split()]
total = ' '.join(total)

# lemma
total = total.replace('cursos','curso')
total = total.replace('aulas','aula')
total = total.replace('media','média')

plt.figure(figsize=(10, 5))
#plt.title('Temas mais frequentes nas mensagens enviadas por alunos')
total_bigrams = build_ngram_df(total, ngram = (1,1), n=15)
plotNgrams(total_bigrams, col='C5', orientation = 'vertical')
plt.show()


#Localizando conversas pesquisando pelo texto

def query_in(df,query):
  messages = list(df['mensagem'])
  match = []
  query = query.lower()
  for m in messages:
    if query in m.lower():
      match.append(True)
    else:
      match.append(False)
  return df[match]

def search_conversation(df, text, begin, end):
  author = query_in(df,text)['autor_da_mensagem'].values[0]
  sender = query_in(df,text)['remetente'].values[0]
  reciever = query_in(df,text)['destinatario'].values[0]

  # message sent by student; student = sender
  if author != 'STUART':    
    student = sender
  # message sent by stuart. 
  else:
    student = reciever

  df_conversation = df[(df['remetente']==student) | (df['destinatario']==student)].reset_index(drop=True) 
  idx = query_in(df_conversation,text).index[0]
  return df_conversation.loc[idx-begin:idx+end]

text = 'CHAT.STUART_INFORMATION_USEFUL'
#search_conversation(df,text,2,10)

df[df['mensagem']=='CHAT.STUART_INFORMATION_USEFUL']








