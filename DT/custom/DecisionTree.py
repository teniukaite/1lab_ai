import numpy
import pandas
import random

#resources https://github.com/SebastianMantey/Decision-Tree-from-Scratch

def trainTestSplit(dataFrame, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize * len(dataFrame))
    indices = dataFrame.index.tolist()
    testIndices = random.sample(population = indices, k = testSize)
    dataFrameTest = dataFrame.loc[testIndices]
    dataFrameTrain = dataFrame.drop(testIndices)
    return dataFrameTrain, dataFrameTest


def checkPurity(data):
    return len(numpy.unique(data[:, -1])) == 1 # tikrina ar duomenys yra unikalūs


def classifyData(data):
    uniqueClasses, uniqueClassesCounts = numpy.unique(data[:, -1], return_counts=True) # paimam unikalius irasus ir ju kieki
    return uniqueClasses[uniqueClassesCounts.argmax()] # grazinam irasa su didziausiu kiekiu


def getPotentialSplits(data):
    potentialSplits = {}
    _, columns = data.shape
    columnsIndices = list(range(columns - 1))
    for column in columnsIndices: # ciklas per visus stulpelius
        values = data[:, column] # paimam stulpelio reiksmes
        uniqueValues = numpy.unique(values) # paimam unikalias reiksmes
        if len(uniqueValues) == 1: # jei unikaliu reiksmiu yra tik viena, tai toks stulpelis nera skirtas splitinti
            potentialSplits[column] = uniqueValues # todel jo nera ir sudedam i dictionary
        else: # jei yra daugiau nei viena unikali reiksme, tai galim splitinti
            potentialSplits[column] = [] # pridedam stulpeli i dictionary i ji priskiriam tuscia masyva
            for i in range(len(uniqueValues)): # ciklas per visus stulpelio unikalias reiksmes
                if i != 0: # jei ne pirma iteracija
                    currentValue = uniqueValues[i] # paimam dabartine reiksme
                    previousValue = uniqueValues[i - 1] # paimam ankstesne reiksme
                    potentialSplits[column].append((currentValue + previousValue) / 2) # ir sudedam vidurki i dictionary
    return potentialSplits # grazinam masyva


def splitData(data, splitColumn, splitValue): # splitina duomenis pagal stulpeli ir reiksme
    splitColumnValues = data[:, splitColumn] # paimam stulpelio reiksmes
    return data[splitColumnValues <= splitValue], data[splitColumnValues > splitValue] # grazina masyvus, pirmas yra zemiau ribos kitas yra virs ribos


def calculateEntropy(data): # skaiciuoja entropija (vidinis)
    _, uniqueClassesCounts = numpy.unique(data[:, -1], return_counts=True) # paimam unikalias klases ir ju kieki
    probabilities = uniqueClassesCounts / uniqueClassesCounts.sum() # paimam tik kieki ir skaiciuojam tikimybes
    return sum(probabilities * -numpy.log2(probabilities)) # grazinam entropija


def calculateOverallEntropy(dataBelow, dataAbove): # skaiciuoja bendra entropija (H)
    # print("dataBelow: ", dataBelow)
    # print("dataAbove: ", dataAbove)
    pDataBelow = len(dataBelow) / (len(dataBelow) + len(dataAbove)) # paimam duomenu kieki ir skaiciuojam dalis
    pDataAbove = len(dataAbove) / (len(dataBelow) + len(dataAbove)) # paimam duomenu kieki ir skaiciuojam dalis
    return pDataBelow * calculateEntropy(dataBelow) + pDataAbove * calculateEntropy(dataAbove) # grazinam bendra entropija


def determineBestSplit(data, potentialSplits): # randa geriausia splita
    overallEntropy = 9999 # priskiriam didziausia galima entropija
    bestSplitColumn = 0 # priskiriam stulpeli
    bestSplitValue = 0  # priskiriam reiksme
    for splitColumn in potentialSplits: # ciklas per visus stulpelius
        for splitValue in potentialSplits[splitColumn]: # ciklas per visus stulpelio reiksmes
            dataBelow, dataAbove = splitData(data, splitColumn, splitValue) # splitinam duomenis
            currentOverallEntropy = calculateOverallEntropy(dataBelow, dataAbove) # skaiciuojam entropija
            print("currentOverallEntropy: ", currentOverallEntropy)
            if currentOverallEntropy <= overallEntropy: # jei entropija mazesne uz esama
                overallEntropy = currentOverallEntropy # priskiriam nauja entropija
                bestSplitColumn = splitColumn   # priskiriam nauja stulpeli
                bestSplitValue = splitValue # priskiriam nauja reiksme
    return bestSplitColumn, bestSplitValue # grazinam stulpeli ir reiksme


def buildDecisionTree(data, columns, currentDepth=0, minSampleLeaf=2, maxDepth=1000): # sukuria decision tree
    if checkPurity(data) or len(data) < minSampleLeaf or currentDepth == maxDepth: # jei duomenys yra unikalus arba maziau nei minSampleLeaf arba pasiektas maxDepth
        return classifyData(data) # unikalus irasas su max kiekiu
    else:
        currentDepth += 1 # pridedam viena iteracija
        potentialSplits = getPotentialSplits(data) # paimam galimus splitus
        splitColumn, splitValue = determineBestSplit(data, potentialSplits) # randa geriausia splita
        dataBelow, dataAbove = splitData(data, splitColumn, splitValue) # splitina duomenis
        if len(dataBelow) == 0 or len(dataAbove) == 0: # jei yra tusciu
            return classifyData(data) # grazinam unikalaus iraso su max kiekiu
        else:
            question = str(columns[splitColumn]) + " <= " + str(splitValue) # sukuriam klausima
            decisionSubTree = {question: []} # sukuriam sub decision tree
            yesAnswer = buildDecisionTree(dataBelow, columns, currentDepth, minSampleLeaf, maxDepth) # sukuriam yes atsakyma melynas medis
            noAnswer = buildDecisionTree(dataAbove, columns, currentDepth, minSampleLeaf, maxDepth) # sukuriam no atsakyma raudonas medis
            if yesAnswer == noAnswer: # jei yes atsakymas lygus no atsakymui
                decisionSubTree = yesAnswer # priskiriam yes atsakyma
            else:
                decisionSubTree[question].append(yesAnswer) # pridedam yes atsakyma i decision tree
                decisionSubTree[question].append(noAnswer) # pridedam no atsakyma i decision tree
            return decisionSubTree  # grazinam subdecision tree


def classifySample(sample, decisionTree):
    if not isinstance(decisionTree, dict): # jei decision tree nera dictionary (key value)
        return decisionTree # grazinam decision tree
    question = list(decisionTree.keys())[0] # paimam pirma klausima
    attribute, value = question.split(" <= ") # paimam stulpeli ir reiksme
    if sample[attribute] <= float(value): # jei stulpelio reiksme mazesne uz riba
        answer = decisionTree[question][0] # paimam yes atsakyma
    else:
        answer = decisionTree[question][1] # paimam no atsakyma
    return classifySample(sample, answer) # grazinam atsakyma


def decisionTreePredictions(dataFrame, decisionTree):
    predictions = dataFrame.apply(classifySample, axis=1, args=(decisionTree,))
    return predictions


def calculateAccuracy(predictedResults, category):
    resultCorrect = predictedResults == category
    return resultCorrect.mean()
