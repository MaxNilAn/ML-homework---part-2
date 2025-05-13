#!/usr/bin/env python
# coding: utf-8

# ## Определение спама

# In[7]:


import pandas as pd

data = pd.read_csv('./spam.csv')
print(data)
print(data.columns)
data.info()

data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(data.head())


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(data['Message'])

w = vect.get_feature_names_out()
print(w, len(w), w[1000])

print(X[:, 1000])


# In[15]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = Pipeline([('vect', CountVectorizer()), ('NB', MultinomialNB())])
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Spam'], test_size=0.3)


# In[18]:


model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))


# In[26]:


msg = [
    'Hi! How are you?',
    'Free subscription',
    'Win the lottery call us',
    'Call me this evening'
]

print(model.predict(msg))


# In[30]:


from sklearn.naive_bayes import GaussianNB
model = Pipeline([('vect', CountVectorizer()), ('NB', GaussianNB())])
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Spam'], test_size=0.3)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))


# ## Фишинг

# In[33]:


data = pd.read_csv('./phishing.csv')
print(data.columns)
data.info()


# In[35]:


X = data.drop(columns='class')
print(X.columns)


# In[41]:


from sklearn.tree import DecisionTreeClassifier

Y = pd.DataFrame(data['class'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
dt = DecisionTreeClassifier()
model = dt.fit(X_train, y_train)
dt_predict = model.predict(X_test)
print(accuracy_score(y_test, dt_predict))


# In[42]:


Y = pd.DataFrame(data['class'].apply(lambda x: 1 if x == 'spam' else 0))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
dt = MultinomialNB()
model = dt.fit(X_train, y_train)
dt_predict = model.predict(X_test)
print(accuracy_score(y_test, dt_predict))


# ## Парсинг

# In[86]:


from bs4 import BeautifulSoup as bs

html_content = '''<html>
<title>Data Science is Fun</title>

<body>
    <h1>Data Science is Fun</h1>
    <div id='paragraphs' class='text'>
        <p id='paragraph 0'>Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 </p>
        <p id='paragraph 1'>Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 </p>
        <p id='paragraph 2'>Here is a link to <a href='https://www.mail.ru'>Mail ru</a></p>
    </div>
    <div id='list' class='text'>
        <h2>Common Data Science Libraries</h2>
        <ul>
            <li>NumPy</li>
            <li>SciPy</li>
            <li>Pandas</li>
            <li>Scikit-Learn</li>
        </ul>
    </div>
    <div id='empty' class='empty'></div>
</body>

</html>'''

soup = bs(html_content, "lxml")
title = soup.find('title')
print(title)
print(type(title))
print(title.text)
print(soup.body.text)
print(soup.body.p)
pList = soup.body.find_all('p')
for p in pList:
    print(p.text)
    print('------------')

print([bullet.text for bullet in sup.body.find_all('li')])

p2 = soup.find(id='paragraph 2')
print(p2.text)

divAll = soup.find_all('div')
print(divAll)

divClassText = soup.find_all('div', class_ = 'text')
print(divClassText)

for div in divClassText:
    id = div.get('id')
    print(id)
    print(div.text)
    print('------------------')


# In[89]:


soup.body.find(id='paragraph 1').decompose()
soup.body.find(id='paragraph 2').decompose()


# In[94]:


print(soup.body.find(id='paragraphs'))

new_p = soup.new_tag('p')
print(new_p)
print(type(new_p))

new_p.string = 'Новое содержание'
print(new_p)

soup.find(id='paragraph 0').append(new_p)
print(soup)


# In[98]:


from urllib.request import urlopen
import lxml

url = 'https://ya.ru'
html_content = urlopen(url).read()
print(html_content[0:1000])

sp = bs(html_content, "lxml")
print(sp.title.text)

