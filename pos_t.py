from stanfordcorenlp import StanfordCoreNLP
import codecs,nltk,re,random
from tashaphyne.stemming import ArabicLightStemmer
ArListem = ArabicLightStemmer()

def classification(name,classifier):
    return classifier.classify(generate_feature(name))

def func(s,classifier):
    n = s[1]
    ss=[]
    if n[len(n)-1]=='S' or (n[len(n)-2:]=='JJ' and (s[0][len(s[0])-2:]=='« '.decode('windows-1256') or s[0][len(s[0])-2:]=='«‰'.decode('windows-1256') or s[0][len(s[0])-2:]=='Ì‰'.decode('windows-1256') or s[0][len(s[0])-2:]=='Ê‰'.decode('windows-1256'))):
        ss.append('P')
    else:
        ss.append('S')

    if (n=='NNPS' or n=='NNP'):
        ss.append(classification(s[0],classifier))
    else:
        if (n[:2]=='VB'):
            if (s[0][0]==' '.decode('windows-1256') or s[0][len(s[0])-1]==' '.decode('windows-1256')):
                ss.append('female')
            else:
                ss.append('male')
        else:
            t = s[0][len(s[0])-1]
            
            if (t =='…'.decode('windows-1256') or t==' '.decode('windows-1256')):
                ss.append('female')
            else:
                ss.append('male')
    
    return ''.join([x+"|" for x in ss])

def generate_feature(word):
    return {'last_letter':word[-1]}

#create StanfordCoreNLP OBJ
local_corenlp_path = r'E:/nlp/stanford-corenlp-full-2017-06-09/'
nlp = StanfordCoreNLP(local_corenlp_path, lang='ar', quiet=False,port=9005)

corpw=codecs.open('finalData.txt','w','UTF-8')

female = codecs.open("ar-female.txt",'r','UTF-8')
male = codecs.open("ar-male.txt",'r','UTF-8')
lFemale = female.readlines()
lMale = male.readlines()

Features_female = [(generate_feature(name.strip()),'female') for name in lFemale ]
Features_male = [(generate_feature(name.strip()),'male') for name in lMale ]
random.shuffle(Features_female)
random.shuffle(Features_male)

numFemale = (5*len(Features_female))/100
numMale = (5*len(Features_male))/100


test_set = Features_female[:numFemale]+Features_male[:numMale]
random.shuffle(test_set)

train_set = Features_female[numFemale:]+Features_male[numMale:]
random.shuffle(train_set)

for i in range(0,5):
    classifier = nltk.NaiveBayesClassifier.train(train_set)


#open Corpora
corp=codecs.open('01.txt','r')
##corpw=codecs.open('w.txt','w','UTF-8')

corps=corp.readlines()
corps = [row.strip()  for row in corps]

ls=[]

#tag words
for l in corps:
    ps=nlp.pos_tag(l)
    if ps[0][0]==u'\ufeff': #ZERO WIDTH NO-BREAK SPACE
        ps=ps[1:]
    dp=nlp.dependency_parse(l)
    dp2=[]
    if len(dp)==len(ps):
        i = dp[0][2]
        for ind,w in enumerate(dp):
            if ind+1==i:
                dp2.append(w)
                dp2.append(("NONE",i,i))
            else:
                dp2.append(w)
    else:
        dp2=dp
    dp2 = dp2[1:]
    
    for ind,w in enumerate(ps) :
        stem = ArListem.light_stem(w[0])
        pre = ArListem.get_prefix()
        suf = ArListem.get_suffix()
        ls.append(w[0]+"|"+w[1]+"|"+dp2[ind][0]+"|"+str(dp2[ind][1]-1)+"|"+func([w[0],w[1]],classifier)+"p="+pre+"|s="+suf+"\n")
    ls.append(". PUNC\n")

corpw.writelines(ls)

corp.close()
corpw.close()
