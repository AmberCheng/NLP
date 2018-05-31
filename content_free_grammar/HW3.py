import nltk
from nltk import *

output = open('Result_CFG.txt', 'w',encoding='utf-8')
output1 = open('Result_CFG_my3.txt', 'w',encoding='utf-8')
output2 = open('Result_PCFG.txt', 'w',encoding='utf-8')

my_sents=["We had a nice party yesterday",
          "She came to visit me two days ago",
          "You may go now",
          "Their kids are not always naive"]

grammar = nltk.CFG.fromstring("""
  S -> NP VP | VP
  NP -> PRP | CD NNS| Det ADJP NN | NN | PRPS NNS
  ADJP -> JJ
  PRP -> "We"|"She"|"me"|"You"
  Det -> "a"
  JJ -> "nice" | "naive"
  NN -> "party"|"yesterday"
  CD -> "two"
  NNS -> "days"|"kids" 
  VP -> VBD NP NP| VBD S | TO VP | VB NP ADVP | MD VP|VB ADVP|VBP RB ADVP ADJP
  ADVP -> NP RB|RB
  VBD -> "had"|"came"
  TO -> "to"
  VB -> "visit"|"go"
  MD -> "may"
  VBP -> "are"
  RB -> "ago"|"now"|"not"|"always"
  PRPS -> "Their"
  """)
rd_parser = nltk.RecursiveDescentParser(grammar)
print("======================CFG==========================\n")
for sent in my_sents:
    trees=rd_parser.parse(sent.split())
    output.write("--------------------------CFG------------------------------\n")
    for tree in trees:
        print("\n--------------------------------------------------------\n")
        print(tree)
        output.write(str(tree)+"\n")
output.close()       
    
my_sents2=["two days to go party now",
          "We may go now",
          "Their days are not always nice"]
print("======================CFG_my3==========================\n")
for sent in my_sents2:
    trees=rd_parser.parse(sent.split())
    output1.write("--------------------------CFG_my3------------------------------\n")
    for tree in trees:
        print("\n--------------------------------------------------------\n")
        print(tree)
        output1.write(str(tree)+"\n")
output1.close()       

grammar = nltk.PCFG.fromstring("""
  S -> NP VP [0.8]
  S -> VP [0.2]
  NP -> PRP [0.5]
  NP -> CD NNS [0.125]
  NP -> Det ADJP NN [0.125]
  NP -> NN [0.125]
  NP -> PRPS NNS [0.125]
  ADJP -> JJ [1.0]
  PRP -> "We" [0.25]
  PRP -> "She" [0.25]
  PRP -> "me" [0.25]
  PRP -> "You" [0.25]
  Det -> "a" [1.0]
  JJ -> "nice" [0.5]
  JJ ->"naive" [0.5]
  NN -> "party" [0.5]
  NN -> "yesterday" [0.5]
  CD -> "two" [1.0]
  NNS -> "days" [0.5]
  NNS -> "kids" [0.5]
  VP -> VBD NP NP[0.143]
  VP -> VBD S [0.143]
  VP -> TO VP [0.143]
  VP -> VB NP ADVP [0.143]
  VP -> MD VP[0.143]
  VP -> VB ADVP[0.143]
  VP -> VBP RB ADVP ADJP[0.143]
  ADVP -> NP RB [0.33]
  ADVP -> RB [0.67]
  VBD -> "had"[0.5]
  VBD -> "came"[0.5]
  TO -> "to" [1.0]
  VB -> "visit"[0.5]
  VB -> "go" [0.5]
  MD -> "may" [1.0]
  VBP -> "are" [1.0]
  RB -> "ago"[0.25]
  RB -> "now" [0.25]
  RB -> "not" [0.25]
  RB -> "always" [0.25]
  PRPS -> "Their" [1.0]
  """)
viterbi_parser = nltk.ViterbiParser(grammar)
print("======================PCFG==========================\n")
for sent in my_sents:
    trees2=viterbi_parser.parse(sent.split())
    output2.write("--------------------------PCFG------------------------------\n")
    for tree in trees2:
        print("\n--------------------------------------------------------\n")
        print(tree)
        output2.write(str(tree)+"\n")
output2.close() 
