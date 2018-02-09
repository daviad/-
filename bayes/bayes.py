#https://github.com/wywywy01/machinelearninginaction-master
from numpy import *

def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],
	               ['maybe','not','take','him','to','dog','park','stupid'],
	               ['my','dalmatian','is','so','cute','i','love','him'],
	               ['stop','posting','stupid','worthless','garbage'],
	               ['mr','licks','ate','my','steak','how','to','stop','him'],
	               ['quit','buying','worthless','dog','food','stupid']
	              ]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec

def creatVocabList(dateSet):
	vocabSet = set([])
	for document in dateSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			# print (word)
			# print vocabList.index(word)
			returnVec[vocabList.index(word)] = 1
		else: print ("the word:%s is not in my vocabulary!" % word)
	return returnVec

def bagOfWords2VecMN(vacabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vacabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTranDocs = len(trainMatrix)
	print "numTranDocs:%s" %numTranDocs
	numWords = len(trainMatrix[0])
	print "numWords:%s" %numWords
	pAbusive = sum(trainCategory)/float(numTranDocs)
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0
	p0Vect = 0.0; p1Vect = 0.0
	for i in range(numTranDocs):
		if trainCategory[i] == 1:
			print "trainMatrix[i]:%s" %trainMatrix[i]
			p1Num += trainMatrix[i]
			print "p1num:%s" %p1Num
			p1Denom += sum(trainMatrix[i]) 
			print "p1Denom:%s" %p1Denom
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i]) 
		if(p1Denom):
			p1Vect = log(p1Num/p1Denom)
		if(p0Denom):
			p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else: return 0 

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []; classList = []; fullText = [];
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' %i).read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' %i).read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(0)
	vacabList = creatVocabList(docList)
	trainingSet = range(50)
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(),vocabList,docIndex[docIndex])
		trainClasses.append(classList[docIndex])
	p0v,p1v,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0v,p1V,pSpam) != classList[docIndex]:	
			errorCount += 1
	print 'the error rate is :', float(errorCount)/len(testSet)

def testingNB():
	listOpost,listClass = loadDataSet()
	listOpost,listClass = loadDataSet()
	myVocabList = creatVocabList(listOpost)
	print 'myVocabList:%s'  %myVocabList
	trainMat = []
	for ponstinDoc in listOpost:
		trainMat.append(setOfWords2Vec(myVocabList,ponstinDoc))

	print "trainMat:%s" %trainMat
	p0v,p1v,pAb = trainNB0(array(trainMat),array(listClass))
	# print setOfWords2Vec(myVocabList,listOpost[0])
	print "p0v:%s" %p0v 
	print "p1v:%s" %p1v
	print "pAb:%s" %pAb

	testEntry = ['love','my','dalmatian']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print 'thisDoc:',thisDoc
	print testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb)

	testEntry = ['stupid','garbage']
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb)


# testingNB()
# print textParse('AVC, sdf, eset cc123')
spamTest()
