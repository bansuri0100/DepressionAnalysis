from flask import Flask, redirect, render_template, request, url_for, session
import time
import re
import pandas as pd
import numpy as np
import csv
import spacy

app = Flask(__name__)

# Secret key for sessions
app.secret_key = "secret"

@app.route("/", methods=['GET', 'POST'])
def index():
	# Initialises balance and transaction count from session variables
	# If session keys do not exist, KeyError raised - (occurs the first time program is run)
	# Session variables initialised
	# To clear session - session.clear()


	if request.method == "GET":
		return render_template("index.html", balance=balance, msg="")

	if request.method == "POST":
		if request.form["speech"] == "" :
			msg = "Sorry...Could you say that again!!!"
			return render_template("interface.html",msg=msg)



		text = request.form["speech"]
		fh=open('isear.txt')
		label=[]
		sen=[]
		for line in fh:
    		lis=re.findall('[a-zA-Z]+',line)
    		label.append(lis[0])
    		sen.append(' '.join(lis[1:]))

    		print(label)

			print(sen)

		with open ('data2.csv','w') as f:
    		writer=csv.writer(f)
    		writer.writerows(zip(label,sen))


		df=pd.read_csv('data2.csv')


		df.head()

		df.rename(columns={'ID':'label','CITY COUN SUBJ SEX AGE RELI PRAC FOCC MOCC FIEL EMOT WHEN LONG INTS ERGO TROPHO TEMPER EXPRES MOVE EXP EXP EXP PARAL CON EXPC PLEA PLAN FAIR CAUS COPING MORL SELF RELA VERBAL NEUTRO Field Field Field MYKEY SIT STATE':'sentence'},inplace=True)


		df.head()



		df.isnull().sum()




		nlp=spacy.load('en_core_web_sm')


		print(nlp.Defaults.stop_words)

		corpus=[]
		for i in range(7666):
		    sentence=re.sub('[^a-zA-Z]', ' ',df['sentence'][i])
		    sentence=sentence.lower()
		    sentence=sentence.split()
		    
		    sentence=[s for s in sentence if not nlp.vocab[s].is_stop]
		    sentence=' '.join(sentence)
		    corpus.append(sentence)


			corpus2=[]
			for i in range(7666):
			    sent=nlp(corpus[i])
			    
			    sent2=[s.lemma_ for s in sent ]
			    sentence2=' '.join(sent2)
			    corpus2.append(sentence2)


# In[16]:


corpus2


# In[17]:


df.head()


# In[18]:


df['cleaned_sentence']=corpus2


# In[19]:


df.head()


# In[20]:


df.label.value_counts()


# In[ ]:


#WordCloud Analysis


# In[21]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import matplotlib.cm
import matplotlib.pyplot as plt


# In[22]:


depressive_words = ' '.join(list(df[df['label'] == 'sadness']['cleaned_sentence']))
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap=matplotlib.cm.inferno).generate(depressive_words)
plt.figure(figsize = (8, 6), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[23]:


depressive_words = ' '.join(list(df[df['label'] == 'fear']['cleaned_sentence']))
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap=matplotlib.cm.inferno).generate(depressive_words)
plt.figure(figsize = (8, 6), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[24]:


df['emotion'] = df['label'].apply(lambda c: 'pos' if c =='sadness' else 'neg')


# In[25]:


df['emotion'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


df2=pd.read_csv('sentiment_tweets3.csv')


# In[28]:


df2.head()


# In[29]:


df2 = df2.drop(['Unnamed: 0'],axis=1)


# In[ ]:





# In[30]:


corpus=[]
for i in range(len(df2)):
    sentence=re.sub('[^a-zA-Z]', ' ',df2['message'][i])
    sentence=sentence.lower()
    sentence=sentence.split()
    
    sentence=[s for s in sentence if not nlp.vocab[s].is_stop]
    sentence=' '.join(sentence)
    corpus.append(sentence)


# In[31]:


corpus2=[]
for i in corpus:
    sent=nlp(i)   
    sent2=[s.lemma_ for s in sent ]
    sentence2=' '.join(sent2)
    corpus2.append(sentence2)


# In[32]:


df2['cleaned_sentence']=corpus2


# In[33]:


df2=df2[['label','message','cleaned_sentence']]


# In[34]:


df2.head()


# In[35]:


df.head()


# In[36]:


df2.rename(columns={'message':'sentence'},inplace=True)


# In[37]:


df3=df2[df2['label']>0]


# In[38]:


df.info()


# In[39]:


df4=df.append(df3)


# In[40]:


df4.head()


# In[41]:


df4['emotion'] = df4['label'].apply(lambda c: 'pos' if c ==1 or c=='sadness' else 'neg')


# In[42]:


df4.info()


# In[43]:


df4['emotion'].value_counts()


# In[44]:


df4


# In[45]:


from sklearn.model_selection import train_test_split



X = df4['cleaned_sentence']
y = df4['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# In[46]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train) 


# In[47]:


def process(str):
    corpus=[]
    
    sentence=re.sub('[^a-zA-Z]', ' ',str)
    sentence=sentence.lower()
    sentence=sentence.split()
    
    sentence=[s for s in sentence if not nlp.vocab[s].is_stop]
    sentence=' '.join(sentence)
    
    
    
    sent=nlp(sentence)   
    sent2=[s.lemma_ for s in sent ]
    sentence2=' '.join(sent2)
    return(sentence2)


# In[56]:


string=str(input("Enter Message :"))
string2=process(string)   
z=pd.Series(string2)
predictions = text_clf.predict(z)
predictions


# In[63]:


predictions2=text_clf.predict(X_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions2))


# In[65]:


print(metrics.classification_report(y_test,predictions2))


# 

# In[66]:


print(metrics.accuracy_score(y_test,predictions2))


# In[67]:


depressive_words = ' '.join(list(df4[df4['emotion'] == 'pos']['cleaned_sentence']))
depressive_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="Set1").generate(depressive_words)
plt.figure(figsize = (10,8), facecolor = 'k')
plt.imshow(depressive_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


  










		

if __name__ == '__main__':
	app.run(host='localhost', port=80, debug=True)