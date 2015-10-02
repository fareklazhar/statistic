#
# The basis of this file is from the book 'Code file for the book Programmer's Guide to Data Mining' http://guidetodatamining.com
# that was written by Ron Zacharski. The sources were adapted for the research on the format analysis.
#


#
#  Nearest Neighbor Classifier 
#
#
#  Code file for the book Programmer's Guide to Data Mining
#  http://guidetodatamining.com
#
#  Ron Zacharski
#


##   I am trying to make the classifier more general purpose
##   by reading the data from a file.
##   Each line of the file contains tab separated fields.
##   The first line of the file describes how those fields (columns) should
##   be interpreted. The descriptors in the fields of the first line are:
##
##        comment   -  this field should be interpreted as a comment
##        class     -  this field describes the class of the field
##        num       -  this field describes an integer attribute that should 
##                     be included in the computation.
##
##        more to be described as needed
## 
##
##    So, for example, if our file describes athletes and is of the form:
##    Shavonte Zellous   basketball  70  155
##    The first line might be:
##    comment   class  num   num
##
##    Meaning the first column (name of the player) should be considered a comment; 
##    the next column represents the class of the entry (the sport); 
##    and the next 2 represent attributes to use in the calculations.
##
##    The classifer reads this file into the list called data.
##    The format of each entry in that list is a tuple
##  
##    (class, normalized attribute-list, comment-list)
##
##    so, for example
##
##   [('basketball', [1.28, 1.71], ['Brittainey Raven']),
##    ('basketball', [0.89, 1.47], ['Shavonte Zellous']),
##    ('gymnastics', [-1.68, -0.75], ['Shawn Johnson']),
##    ('gymnastics', [-2.27, -1.2], ['Ksenia Semenova']),
##    ('track', [0.09, -0.06], ['Blake Russell'])]
##

from math import sqrt
from scipy import linalg, mat, dot

TRAINING_SET = 'formatsTrainingSet.txt'
FLOAT_SIZE = 2

class Classifier:


    def __init__(self, filename):

        self.medianAndDeviation = []
        
        # reading the data in from the file
        f = open(filename)
        lines = f.readlines()
        f.close()
        self.format = lines[0].strip().split('\t')
        self.data = []
        for line in lines[1:]:
            fields = line.strip().split('\t')
            ignore = []
            vector = []
            for i in range(len(fields)):
                if self.format[i] == 'num':
                    vector.append(float(fields[i]))
                elif self.format[i] == 'comment':
                    ignore.append(fields[i])
                elif self.format[i] == 'class':
                    classification = fields[i]
            self.data.append((classification, vector, ignore))
        self.rawData = list(self.data)
        # get length of instance vector
        self.vlen = len(self.data[0][1])
        # now normalize the data
        for i in range(self.vlen):
            self.normalizeColumn(i)
        

        
    
    ##################################################
    ###
    ###  CODE TO COMPUTE THE MODIFIED STANDARD SCORE

    def getMedian(self, alist):
        """return median of alist"""
        if alist == []:
            return []
        blist = sorted(alist)
        length = len(alist)
        if length % 2 == 1:
            # length of list is odd so return middle element
            return blist[int(((length + 1) / 2) -  1)]
        else:
            # length of list is even so compute midpoint
            v1 = blist[int(length / 2)]
            v2 =blist[(int(length / 2) - 1)]
            return (v1 + v2) / 2.0
        

    def getAbsoluteStandardDeviation(self, alist, median):
        """given alist and median return absolute standard deviation"""
        sum = 0
        for item in alist:
            sum += abs(item - median)
        return sum / len(alist)


    def normalizeColumn(self, columnNumber):
       """given a column number, normalize that column in self.data"""
       # first extract values to list
       col = [v[1][columnNumber] for v in self.data]
       median = self.getMedian(col)
       asd = self.getAbsoluteStandardDeviation(col, median)
       if asd == 0.0:
           asd = 0.000001
       print("Median: %f   ASD = %f" % (median, asd))
       self.medianAndDeviation.append((median, asd))
       for v in self.data:
           v[1][columnNumber] = (v[1][columnNumber] - median) / asd
       print 'normalized column: ', self.data


    def normalizeVector(self, v):
        """We have stored the median and asd for each column.
        We now use them to normalize vector v"""
        vector = list(v)
        for i in range(len(vector)):
            (median, asd) = self.medianAndDeviation[i]
            vector[i] = (vector[i] - median) / asd
        print("normalized vector: ", vector)
        return vector

    
    ###
    ### END NORMALIZATION
    ##################################################



    def manhattan(self, vector1, vector2):
        """Computes the Manhattan distance."""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))


    def nearestNeighbor(self, itemVector):
        """return nearest neighbor to itemVector"""
        return min([ (self.manhattan(itemVector, item[1]), item)
                     for item in self.data])
    
    def classify(self, itemVector):
        """Return class we think item Vector is in"""
        return(self.nearestNeighbor(self.normalizeVector(itemVector))[1][0])
 
    def calculateCosineSimilarity(self, trainingVector, testVector):
        """Return cosine similarity between two vectors. We use cosine similarity because test data may be sparse"""
        #sqrt(lambda x: pow(x, 2), trainingVector)
        trainingVector = map(int, trainingVector[2:])
        testVector = map(int, testVector[2:])
        print 'trainingVector', trainingVector
        print 'testVector', testVector

        #a = mat(trainingVector)
        #b = mat(testVector)

        #print 'a', a, 'b', b
        #c = dot(a,b.T)/linalg.norm(a)/linalg.norm(b)

        #return c
        import math
        "compute cosine similarity of v1 to v2: (v1 dot v1)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(trainingVector)):
            x = float(trainingVector[i]); y = float(testVector[i])
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)

def unitTest():
    classifier = Classifier(TRAINING_SET)
    br = ('Basketball', [72, 162], ['Brittainey Raven'])
    nl = ('Gymnastics', [61, 76], ['Viktoria Komova'])
    cl = ("Basketball", [74, 190], ['Crystal Langhorne'])
    # first check normalize function
    brNorm = classifier.normalizeVector(br[1])
    nlNorm = classifier.normalizeVector(nl[1])
    clNorm = classifier.normalizeVector(cl[1])
    assert(brNorm == classifier.data[1][1])
    assert(nlNorm == classifier.data[-1][1])
    print('normalizeVector fn OK')
    # check distance
    assert (round(classifier.manhattan(clNorm, classifier.data[1][1]), 5) == 1.16823)
    assert(classifier.manhattan(brNorm, classifier.data[1][1]) == 0)
    assert(classifier.manhattan(nlNorm, classifier.data[-1][1]) == 0)
    print('Manhattan distance fn OK')
    # Brittainey Raven's nearest neighbor should be herself
    result = classifier.nearestNeighbor(brNorm)
    assert(result[1][2]== br[2])
    # Nastia Liukin's nearest neighbor should be herself
    result = classifier.nearestNeighbor(nlNorm)
    assert(result[1][2]== nl[2])
    # Crystal Langhorne's nearest neighbor is Jennifer Lacy"
    assert(classifier.nearestNeighbor(clNorm)[1][2][0] == "Jennifer Lacy")
    print("Nearest Neighbor fn OK")
    # Check if classify correctly identifies sports
    assert(classifier.classify(br[1]) == 'Basketball')
    assert(classifier.classify(cl[1]) == 'Basketball')
    assert(classifier.classify(nl[1]) == 'Gymnastics')
    print('Classify fn OK')

def test(training_filename, test_filename):
    """Test the classifier on a test set of data"""
    classifier = Classifier(training_filename)
    #print 'classifier: ', classifier
    f = open(test_filename)
    lines = f.readlines()
    f.close()
    numCorrect = 0.0
    for line in lines:
        data = line.strip().split('\t')
        vector = []
        classInColumn = -1
        #print("%s  " % (data))
        for i in range(len(classifier.format)):
              if classifier.format[i] == 'num':
                  vector.append(float(data[i]))
              elif classifier.format[i] == 'class':
                  classInColumn = i
        theClass= classifier.classify(vector)
        prefix = '-'
        if theClass == data[classInColumn]:
            # it is correct
            numCorrect += 1
            prefix = '+'
        print("%s  %12s  %s" % (prefix, theClass, line))
    print("%4.2f%% correct" % (numCorrect * 100/ len(lines)))
        
def visualize(training_filename, par1, par2):
    """Test the classifier on a test set of data"""
    #classifier = Classifier(training_filename)
    f = open(training_filename)
    lines = f.readlines()
    f.close()
    numCorrect = 0.0
    for line in lines:
        data = line.strip().split('\t')
#        print data[1], ' par1: ', data[par1], 'par2: ', data[par2]
        print '%s' % data[1] + ';' + data[par1] + ';' + data[par2]

def visualizeNormalized(training_filename, par1, par2):
    """Test the classifier on a test set of data"""
    classifier = Classifier(training_filename)
    f = open(training_filename)
    lines = f.readlines()
    f.close()
    i = -1
    for line in lines:
        data = line.strip().split('\t')
        ##print data
        ##print data[1], ' par1: ', data[par1], 'par2: ', data[par2]
        #print 'classifier.data: ', classifier.data[i][0], classifier.data[i][1]
        ##print '%s' % data[1] + ';' + data[par1] + ';' + data[par2]
#        print 'normalized %s' % classifier.data[i][0] + ';' + classifier.data[i][par1] + ';' + classifier.data[i][par2]
        print '%s' % classifier.data[i][0] + ';' \
              + str(round(classifier.data[i][1][par1 - 2], FLOAT_SIZE)).replace('.', ',') + ';' \
              + str(round(classifier.data[i][1][par2 - 2], FLOAT_SIZE)).replace('.', ',')
        i += 1

def findNearestNeighbourByCosineSimilarity(training_filename, test_filename):
    """Test the classifier on a test set of data"""
    classifier = Classifier(training_filename)
    f = open(training_filename)
    lines = f.readlines()
    f.close()
    f = open(test_filename)
    testLines = f.readlines()
    f.close()
    i = -1
    cosines = []
    expertData = []
    for line in lines:
        data = line.strip().split('\t')
        print data, i
        if i > -1:
            #cosine = classifier.calculateCosineSimilarity(data, lines.pop(i))
            cosine = classifier.calculateCosineSimilarity(data, testLines[0].strip().split('\t'))
            cosines.append(cosine)
            expertData.append(data)
        ##print data[1], ' par1: ', data[par1], 'par2: ', data[par2]
        #print 'classifier.data: ', classifier.data[i][0], classifier.data[i][1]
        ##print '%s' % data[1] + ';' + data[par1] + ';' + data[par2]
#        print 'normalized %s' % classifier.data[i][0] + ';' + classifier.data[i][par1] + ';' + classifier.data[i][par2]
        #print '%s' % classifier.data[i][0] + ';' \
        #      + str(round(classifier.data[i][1][par1 - 2], FLOAT_SIZE)).replace('.', ',') + ';' \
        #      + str(round(classifier.data[i][1][par2 - 2], FLOAT_SIZE)).replace('.', ',')
        i += 1

    idx = 0
    maxCosine = -1.0
    bestMatchIdx = 0
    print 'cosines', cosines
    print 'expertData', expertData
    for cosine in cosines:
        if cosine > maxCosine:
            maxCosine = cosine
            bestMatchIdx = idx
        idx +=1

    print 'best match index', bestMatchIdx, 'maxCosine', maxCosine
    print 'best match: ', expertData[bestMatchIdx]

##
##  Here are examples of how the classifier is used for format visualisation.
##
###test('formatsTrainingSet.txt', 'formatsTestSet.txt')#
#visualize('formatsTrainingSet.txt', 4, 6)# sw/versions
#visualizeNormalized('formatsTrainingSet.txt', 4, 6)# sw/versions
#visualizeNormalized('formatsTrainingSet.txt', 4, 5)# sw/vendors
#visualizeNormalized('formatsTrainingSet.txt', 4, 3)# sw/os
####visualizeNormalized('formatsTrainingSet.txt', 4, 2)# sw/popular

##
##  Here are examples of how the classifier is used for risk visualisation.
##
#findNearestNeighbourByCosineSimilarity('riskFactorsTrainingSet.txt', 'riskFactorsTestSet.txt')
#visualizeNormalized('riskFactorsTrainingSetVisualise.txt', 4, 6)# expert 3/expert 5
visualizeNormalized('riskFactorsTrainingSetVisualise.txt', 10, 12)# expert 9/institutional expert

##
##  Here are examples of how the classifier is used on different data sets
##  in the book.
##test('formatsTrainingSet-Sample.txt', 'formatsTestSet-Sample.txt')#
#  test('athletesTrainingSet.txt', 'athletesTestSet.txt')
#  test("irisTrainingSet.data", "irisTestSet.data")
#  test("mpgTrainingSet.txt", "mpgTestSet.txt")
#unitTest()
    
