import codecs,nltk,re,random
from stanfordcorenlp import StanfordCoreNLP
from tashaphyne.stemming import ArabicLightStemmer
ArListem = ArabicLightStemmer()

def generate_featureS(sentence):
    depgender=sentence[3][2]
    return {"depdender":depgender}

def generate_featureJ(sentence):
    depgender=sentence[3][2]
    depn=sentence[3][1]
    depsuf=sentence[3][4]
    return {"depgender":depgender,"depnum":depn,"depsuf":depsuf[2:]}

def generate_featureO(sentence):
    n=sentence[4]
    gender=sentence[5]
    return {"num":n,"gender":gender}

def generate_featureR(sentence):
    n=sentence[4]
    gender=sentence[5]
    return {"num":n,"gender":gender}

corp=codecs.open('finalData.txt','r','utf-8')

lines = corp.readlines()
corps = [(row.strip()).split('|')  for row in lines]

ls=[]
l=[]

for ind,r in enumerate(corps):
    if r[0] == '. PUNC':
        ls.append(l)
        l=[]
        continue
    l.append(r)

kk=[]
dep=[]
for s in ls:
    for w in s:
        index=w[3]
        w[3] = [s[int(index)][1]]+s[int(index)][4:]
        kk.append(w)
##        dep.append(w[2])

n1=0
n2=0
n3=0
n4=0
fs=[]
for w in kk:
    if w[2][len(w[2])-4:]=='subj':
##        n2=n2+1
        fs.append((generate_featureS(w),w[5]))
        continue
    if w[1][len(w[1])-2:]=='JJ':
        ##        n1=n1+1
        if(w[4]=='S'):
            fs.append((generate_featureJ(w),(w[5],w[4],'')))
            continue
        fs.append((generate_featureJ(w),(w[5],w[4],w[7][2:])))
        continue
    if w[2][len(w[2])-3:]=='obj' and w[3][0][:2]=='VB' and w[4]=='P':
##        n3+=1
        fs.append((generate_featureO(w),(w[7][2:],w[5])))
        continue
    if (w[1][:2]=='NN' or w[1][:4]=='DTNN') and w[3][0]=='IN' and w[4]=='P':
##        n4+=1
        fs.append((generate_featureR(w),(w[7][2:])))
        continue

##print n1
##print n2
##print len(fs)
##print n3_s
##print n3_p
##print n4
random.shuffle(fs)
ntrain = (len(fs)*80)/100
train = fs[:ntrain]
test = fs[ntrain:]

classifier=nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(classifier,test))

##****************************************************************
##test
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
    
    return ss

def generate_feature(word):
    return {'last_letter':word[-1]}

local_corenlp_path = r'E:/nlp/stanford-corenlp-full-2017-06-09/'
nlp = StanfordCoreNLP(local_corenlp_path, lang='ar', quiet=False,port=9010)

##train female,male
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
    classifier2 = nltk.NaiveBayesClassifier.train(train_set)

##**********************
inp=codecs.open('input.txt','r')
inps=inp.readlines()
l = inps[5].strip()
print l.decode('utf-8')

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

##delete root
dp2 = dp2[1:]
l=[]
ls=[]  
for ind,w in enumerate(ps):
    stem = ArListem.light_stem(w[0])
    pre = ArListem.get_prefix()
    suf = ArListem.get_suffix()
    l=[]
    l.append(w[0])
    l.append(w[1])
    l.append(dp2[ind][0])
    l.append(dp2[ind][1]-1)
    l.append(func([w[0],w[1]],classifier2)[0])
    l.append(func([w[0],w[1]],classifier2)[1])
    l.append("p="+pre)
    l.append("s="+suf)
    ls.append(l)

kk=[]
for w in ls:
    index=w[3]
    w[3] = [ls[index][1]]+ls[index][4:]
    kk.append(w)

##test = classifier.classify(generate_featureO(kk[2]))
##print "correct : "+test[0]+" , "+test[1]
##print "original : "+kk[2][7]+" , "+kk[2][5]
##print kk[2][7][2:]==test[0] and kk[2][5]==test[1]

test = classifier.classify(generate_featureR(kk[1]))
print test[0]
print "original : "+kk[1][7]
print kk[1][7][2:]==test[0]

##test = classifier.classify(generate_featureS(kk[0]))
##print "correct : "+test
##print "original : "+kk[0][5]
##print kk[0][5]==test

##test = classifier.classify(generate_featureJ(kk[1]))
##print "correct : "+str(test)
##print "original : "+kk[1][5] +" , "+ kk[1][4]+" , " + kk[1][7][2:]
##print (kk[1][5]==test[0] and kk[1][4]==test[1] and kk[1][7][2:]==test[2])
##*****************************

corp.close()
inp.close()
