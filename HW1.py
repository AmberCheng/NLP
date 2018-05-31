import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import re
from nltk.collocations import *

mycorpus = PlaintextCorpusReader('.', '.*\.txt')
'''----------------Part1-------------------------'''
part1string = mycorpus.raw('state_union_part1.txt')
part1tokens = nltk.word_tokenize(part1string)
part1words = [w.lower() for w in part1tokens]
alphapart1words = [w for w in part1words if w.isalpha()]

stopwords = nltk.corpus.stopwords.words('english')
fstop = open('Smart.English.stop','r',encoding='utf -8')
mystoptext = fstop.read()
fstop.close()
mystopwords = nltk.word_tokenize(mystoptext)
stoppedpart1words = [w for w in alphapart1words if w not in stopwords]
mystoppedpart1words = [w for w in stoppedpart1words if w not in mystopwords]

wnl = nltk.WordNetLemmatizer()
part1Lemma = [wnl.lemmatize(t) for t in mystoppedpart1words]

part1dist = FreqDist(part1Lemma)
part1topkeys = part1dist.most_common(50)
output1 = open('Part1_top50.txt', 'w',encoding='utf-8')
print("/////-----------------Part1 Top50 Words by Frequencies-------------------/////")
for pair in part1topkeys:
    print(pair)
    output1.write(pair.__str__()+"\n")   
output1.close()

bigram_measures = nltk.collocations.BigramAssocMeasures()
part1finder = BigramCollocationFinder.from_words(alphapart1words)
part1finder.apply_word_filter(lambda w: w in stopwords)
scored = part1finder.score_ngrams(bigram_measures.raw_freq)
print("/////-----------------Part1 Top50 Bigrams by Frequencies-------------------/////")
output2 = open('Part1_top50_Big_Fre.txt', 'w',encoding='utf-8')
for bscore in scored[:50]:
    print(bscore)
    output2.write(bscore.__str__()+"\n")
output2.close()

print("/////------------Part1 Top50 Bigrams by Mutual Information scores----------/////")
output3 = open('Part1_top50_Big_PMI.txt', 'w',encoding='utf-8')
part1finder.apply_freq_filter(5)
scored = part1finder.score_ngrams(bigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)
    output3.write(bscore.__str__()+"\n")
output3.close()

'''----------------Part2-------------------------'''
part2string = mycorpus.raw('state_union_part2.txt')
part2tokens = nltk.word_tokenize(part2string)
part2words = [w.lower() for w in part2tokens]
alphapart2words = [w for w in part2words if w.isalpha()]

stoppedpart2words = [w for w in alphapart2words if w not in stopwords]
mystoppedpart2words = [w for w in stoppedpart2words if w not in mystopwords]

wnl = nltk.WordNetLemmatizer()
part2Lemma = [wnl.lemmatize(t) for t in mystoppedpart2words]

part2dist = FreqDist(part2Lemma)
part2topkeys = part2dist.most_common(50)
output4 = open('Part2_top50.txt', 'w',encoding='utf-8')
print("/////-----------------Part2 Top50 Words by Frequencies-------------------/////")
for pair in part2topkeys:
    print(pair)
    output4.write(pair.__str__()+"\n")   
output4.close()

#bigram_measures = nltk.collocations.BigramAssocMeasures()
part2finder = BigramCollocationFinder.from_words(alphapart2words)
part2finder.apply_word_filter(lambda w: w in stopwords)
scored = part2finder.score_ngrams(bigram_measures.raw_freq)
print("/////-----------------Part2 Top50 Bigrams by Frequencies-------------------/////")
output5 = open('Part2_top50_Big_Fre.txt', 'w',encoding='utf-8')
for bscore in scored[:50]:
    print(bscore)
    output5.write(bscore.__str__()+"\n")
output5.close()

print("/////------------Part2 Top50 Bigrams by Mutual Information scores----------/////")
output6 = open('Part2_top50_Big_PMI.txt', 'w',encoding='utf-8')
part2finder.apply_freq_filter(5)
scored = part2finder.score_ngrams(bigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)
    output6.write(bscore.__str__()+"\n")
output6.close()


