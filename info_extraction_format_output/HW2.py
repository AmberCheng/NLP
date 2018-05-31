import nltk
import re
from nltk.corpus import PlaintextCorpusReader

output1 = open('2B_Result.txt', 'w',encoding='utf-8')
output2 = open('3_Result.txt', 'w',encoding='utf-8')

mycorpus = PlaintextCorpusReader('./NSF_abstracts', '.*\.txt')
#part2-B
for item in mycorpus.fileids():
    if(mycorpus.raw(item)!=''):
        #file = re.findall(r"a+[0-9]+",mycorpus.raw(item))
        file = re.findall(r"File\s+[:]\s+(.+?)\s",mycorpus.raw(item))   
        nsf = re.findall(r"NSF Org\s+[:]\s+(.+?)\s",mycorpus.raw(item))
        #amount = re.findall(r"[$][0-9]+", mycorpus.raw(item))
        amount = re.findall(r"Total Amt.\s+[:]\s+(.+?)\s",mycorpus.raw(item))
        abs = re.findall(r"Abstract\s+[:]\s+(.+?)$\n", mycorpus.raw(item),re.S)
        abs=' '.join(re.split(' +|\n+',abs[0])).strip()
        abs=' '.join(abs.split())
        abs=re.sub('\.\s*\W+','.',abs)
        #print(file[0]+"  "+nsf[0]+" "+amount[0]+"  "+abs)
        output1.write(file[0]+"    "+nsf[0]+" "+amount[0]+"  "+abs+'\n')   
output1.close()
#part3
for item in mycorpus.fileids():
    if(mycorpus.raw(item)!=''):
        i=0
        #print("---------------------------------------")
        #print("Abstract_ID | Sentence_No | Sentence")
        #print("---------------------------------------")
        output2.write("---------------------------------------\n")
        output2.write("Abstract_ID | Sentence_No | Sentence\n")
        output2.write("---------------------------------------\n")
        file = re.findall(r"File\s+[:]\s+(.+?)\s",mycorpus.raw(item))
        abs = re.findall(r"Abstract\s+[:]\s+(.+?)$\n", mycorpus.raw(item),re.S)
        abs=' '.join(re.split(' +|\n+',abs[0])).strip()
        abs=' '.join(abs.split())
        abs=re.sub('\.\s*\W+','.',abs)
        pat = re.compile(r"""([A-Z].*?[\.!?])(?<!Dr\.)(?<!Mrs\.)(?<!Mrs\.)(?<!Jr\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!No\.)(?<![A-Z]\.)""",re.M)
        sents=pat.findall(abs)
        for sen in sents:
            #print(file[0]+"|"+str(i+1)+"|"+sents[i])
            output2.write(file[0]+"|"+str(i+1)+"|"+sents[i]+"\n")
            i=i+1
        #print("Number of sentences : "+str(i))
        output2.write("Number of sentences : "+str(i)+"\n")
output2.close()
        
        
