
# coding: utf-8

# In[1]:


import model_1


# In[3]:


sc = model_1.predict('x ray data/CXR623_IM-2205-2002.png','x ray data/CXR623_IM-2205-3003.png')


# In[4]:


sc = model_1.predict('x ray data/CXR626_IM-2206-1001.png','x ray data/CXR626_IM-2206-2001.png')


# In[5]:


from helpers import *
sc = model_1.score('x ray data/CXR623_IM-2205-2002.png','x ray data/CXR623_IM-2205-3003.png',Data['x ray data/CXR623_IM-2205-2002.png'])


# In[6]:


sc = model_1.score('x ray data/CXR626_IM-2206-1001.png','x ray data/CXR626_IM-2206-2001.png',Data['x ray data/CXR626_IM-2206-1001.png'])

