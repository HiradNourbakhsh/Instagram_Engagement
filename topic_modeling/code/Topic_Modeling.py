#!/usr/bin/env python
# coding: utf-8

# # Company: Massimo Dutti

# In[1]:


get_ipython().system(' pip install pyLDAvis')


# In[ ]:





# In[2]:


import gensim
from gensim.models import ldamulticore
from pprint import pprint
from gensim.models import coherencemodel


# In[3]:


import pandas as pd
import numpy as np

from ast import literal_eval


# In[4]:


import nltk; nltk.download('stopwords')
import re
from pprint import pprint
import spacy


# In[5]:


from nltk.tokenize import word_tokenize


# In[6]:


import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


# In[7]:


import pyLDAvis


# In[8]:


import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec


# In[9]:


import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)


# In[10]:


df = pd.read_excel('/content/GV_Output.xlsx')


# In[11]:


df


# In[12]:


df = df.iloc[1:, :]


# In[13]:


df


# In[ ]:





# # Topic Modeling

# In[14]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import nltk
nltk.download('punkt')


# In[ ]:





# In[15]:


stop = stopwords.words('english')
df['Labels'] = df['Labels'].astype(str)
df['label_tokens'] = df['Labels'].apply(lambda each_post: word_tokenize(re.sub(r'[^\w\s]',' ',each_post.lower())))
df['label_tokens'] = df['label_tokens'].apply(lambda list_of_words: [x for x in list_of_words if x not in stop])


# In[16]:


df


# In[17]:


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


# In[18]:


def get_corpus(df):
    bigram = bigrams(df.label_tokens)
    bigram = [bigram[review] for review in df.label_tokens]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


# In[19]:


train_corpus, train_id2word, bigram_train = get_corpus(df)


# # Finding optimal number of topics

# In[20]:


df.label_tokens[1]


# In[21]:


for k in range(2,20):
  print('k = ' + str(k))
  LDA = gensim.models.ldamulticore.LdaMulticore
  ldamodel = LDA(corpus = train_corpus,
      num_topics = k,
      id2word = train_id2word,
      iterations = 100,
      chunksize = 10000,
      workers = 7,
      passes = 20,
      eval_every = 10,
      per_word_topics = True
  )
  ldamodel.save(f"ldamodel_for_{k}topics_Run_10")
  pprint(ldamodel.print_topics())


# In[22]:


coherence = []
for k in range(2, 20):
  LDA = gensim.models.ldamulticore.LdaMulticore
  ldamodel = LDA.load(f'ldamodel_for_{k}topics_Run_10')
  cm = gensim.models.coherencemodel.CoherenceModel(model = ldamodel, texts = bigram_train, dictionary = train_id2word, coherence = 'c_v')
  coherence.append((k, 'default', 'default', cm.get_coherence()))


# In[23]:


pd.DataFrame(coherence, columns = ['LDA_model', 'alpha', 'eta', 'coherence_score']).to_csv('coherence_matrix.csv', index = False)
mat = pd.read_csv('coherence_matrix.csv')
mat.reset_index(drop = True)
x= range(2,20)
plt.plot(x, mat['coherence_score'])
plt.xlabel('Num Topics')
plt.ylabel('Coherence score')
plt.legend(('coherence_values'), loc = 'best')
plt.show()


# The ideal number of topics is 7 (highest coherence score). I found this number by iterating through values of k (representing the number of topics for the LDA model) from 2 to 20.

# In[24]:


import math


# In[25]:


k = 7
insta_lda = LDA(train_corpus, num_topics = k, id2word = train_id2word, passes=20)

def plot_top_words(lda=insta_lda, nb_topics=k, nb_words=25):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.25, hspace=0.25)
    plt.figure(figsize=(45,40))
    
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='orange', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic: "+str(i))


# In[26]:


plot_top_words()


# ### Topic names: 
# 
# Topic 0: dress shirt
# 
# Topic 1: modeling
# 
# 
# Topic 2: outerwear
# 
# Topic 3: vintage
# 
# Topic 4: eyewear
# 
# Topic 5: nature
# 
# Topic 6: formal wear

# # Engagement

# In[27]:


df


# In[28]:


comments = pd.read_excel('/content/Insta_download.xlsx')


# In[29]:


comments


# In[30]:


insta_lda.print_topics()


# In[31]:


df_lda = pd.DataFrame(insta_lda.print_topics(), columns = ['Topic', 'Weights'])


# In[32]:


df_lda


# In[33]:


df_lda.to_csv('topic_weights.csv')


# In[34]:


train_vecs = []
for i in range(len(df.label_tokens)):
    top_topics = insta_lda.get_document_topics(train_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(len(top_topics))]
    train_vecs.append(topic_vec)


# In[35]:


len(train_vecs)


# In[36]:


train_vec_df = pd.DataFrame(train_vecs)
train_vec_df.columns = ['Topic_' + str(i) for i in range(7)]


# In[37]:


train_vec_df.head()


# # Joining dataframes

# In[38]:


df.head(3)


# In[39]:


comments.head(3)


# In[40]:


df = df.reset_index(drop = True)
df.head(3)


# In[41]:


df


# In[42]:


merged_inner = pd.merge(left = df, right = comments, left_on = 'URL', right_on = 'URL')
merged_inner.head(10)


# In[43]:


merged_inner.shape


# In[44]:


final_df = pd.concat([merged_inner.reset_index(drop = True), train_vec_df.reset_index(drop  =True)], axis = 1)
final_df


# In[45]:


sorted_df = final_df.sort_values(by = ['Comments'], ascending = False)
sorted_df.head(5)


# In[46]:


lq = np.percentile(final_df.Comments, 25)
hq = np.percentile(final_df.Comments, 75)
print(lq, hq)


# In[47]:


final_df.Comments.value_counts().sort_values(ascending = False)


# In[48]:


top = final_df[final_df.Comments >= hq]
top


# In[49]:


low = final_df[final_df.Comments <= lq]
low


# In[50]:


avtop = top.iloc[:, 5:].mean(axis = 0)
avtop


# In[51]:


avlow = low.iloc[:, 5:].mean(axis = 0)
avlow


# In[52]:


quartop = pd.concat([avtop, avlow], axis = 1)
quartop.columns = ['Top Quartile', 'Bottom Quartile']
quartop


# ### In the top quartile, the highest weighted topics are topic 2, and topic 1. These correspond to outerwear and modeling, respectively. In the bottom quartile, the highest weighted topics are topic 0 and topic 5. These correspond to dress shirt and nature. Based on these findings, in order to increase engagement, I would recommend Massimo Dutti post more photographs of outerwear and their models (of either gender) in different poses, as those appear to increase engagement with their instagram followers. Photographs pertaining to nature or dress shirts are associated with lower engagement among their followers, thus I would dissuade them from posting such photos frequently.

# In[ ]:





# In[ ]:




