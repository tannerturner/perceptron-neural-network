import math
import operator

__author__ = 'Tanner Turner, A11838573'


def loadData(filename):
    listOfVectors = []

    with open(filename, encoding='utf-8') as f:
        for line in f:

            vec = []

            for num in line.split():
                n = int(num)
                vec.append(n)

            listOfVectors.append(tuple(vec))

    return tuple(listOfVectors)


def perceptron(vectors, w, negClass=0):
    for v in vectors:
        sgnX = (-1 if v[-1] == negClass else 1)
        x = [i*sgnX for i in v]
        del x[-1]

        if inProd(w, x) <= 0:
            w = [a+b for a, b in zip(w, x)]

    return w


def votedPerceptron(vectors, wc):
    w = wc[0]
    c = wc[1]
    ws = []
    cs = []

    for v in vectors:
        sgnX = (-1 if v[-1] == 0 else 1)
        x = [i*sgnX for i in v]
        del x[-1]

        if inProd(w, x) <= 0:
            ws.append(w)
            cs.append(c)
            w = [a+b for a, b in zip(ws[-1], x)]
            c = 1
        else:
            ++c

    ws.append(w)
    cs.append(c)
    return [(a, b) for a, b in zip(ws, cs)]


def inProd(x, y):
    return sum(a*b for a, b in zip(x, y))


def sign(num, special=True, negClass=0, posClass=6):
    if special:
        return negClass if num <= 0 else posClass
    else:
        return -1 if num <= 0 else 1


def classifyReg(w, vectors):
    labels = []

    for v in vectors:
        label = sign(inProd(w, v))
        t = (label, v[-1])
        labels.append(t)

    return labels


def classifyVoted(pairs, vectors):
    labels = []

    for v in vectors:

        sum = 0

        for p in pairs:
            w = p[0]
            c = p[1]
            sum += c*sign(inProd(w, v), False)

        label = sign(sum, True)
        t = (label, v[-1])
        labels.append(t)

    return labels


def classifyAvg(pairs, vectors):
    labels = []

    for v in vectors:

        sumVec = [0]*784

        for p in pairs:
            w = p[0]
            c = p[1]
            newW = (c*i for i in w)
            sumVec = [(a+b) for a, b in zip(sumVec, newW)]

        label = sign(inProd(sumVec, v))
        t = (label, v[-1])
        labels.append(t)

    return labels


def classifyMulti(cs, vectors):
    labels = []

    for v in vectors:
        label = "Don't know"
        for i in range(10):
            if sign(inProd(cs[i], v), True, i, -1) == i:
                if label != "Don't know":
                    label = "Don't know"
                    break
                else:
                    label = i
        t = (label, v[-1])
        labels.append(t)

    return labels


def getErr(labels):
    return sum(0 if a == b else 1 for a, b in labels) / len(labels)


def main():
    w0 = [0]*784


    trainDataA = loadData("hw4atrain.txt")
    testDataA = loadData("hw4atest.txt")

    regW1 = perceptron(trainDataA, w0)
    regW2 = perceptron(trainDataA, regW1)
    regW3 = perceptron(trainDataA, regW2)

    regLabelsTr1 = classifyReg(regW1, trainDataA)
    regLabelsTr2 = classifyReg(regW2, trainDataA)
    regLabelsTr3 = classifyReg(regW3, trainDataA)

    regLabelsTs1 = classifyReg(regW1, testDataA)
    regLabelsTs2 = classifyReg(regW2, testDataA)
    regLabelsTs3 = classifyReg(regW3, testDataA)

    trRegError1 = getErr(regLabelsTr1)
    trRegError2 = getErr(regLabelsTr2)
    trRegError3 = getErr(regLabelsTr3)

    tsRegError1 = getErr(regLabelsTs1)
    tsRegError2 = getErr(regLabelsTs2)
    tsRegError3 = getErr(regLabelsTs3)

    print("Training error, regular perceptron, one pass: "+str(round(trRegError1*100, 3))+"%")
    print("Training error, regular perceptron, two passes: "+str(round(trRegError2*100, 3))+"%")
    print("Training error, regular perceptron, three passes: "+str(round(trRegError3*100, 3))+"%")
    print("Test error, regular perceptron, one pass: "+str(round(tsRegError1*100, 3))+"%")
    print("Test error, regular perceptron, two passes: "+str(round(tsRegError2*100, 3))+"%")
    print("Test error, regular perceptron, three passes: "+str(round(tsRegError3*100, 3))+"%")
    print()

    votedPairs1 = votedPerceptron(trainDataA, (w0, 1))

    tmp1 = votedPerceptron(trainDataA, votedPairs1[-1])
    del votedPairs1[-1]
    votedPairs2 = votedPairs1 + tmp1

    tmp2 = votedPerceptron(trainDataA, votedPairs2[-1])
    del votedPairs2[-1]
    votedPairs3 = votedPairs2 + tmp2

    votedLabelsTr1 = classifyVoted(votedPairs1, trainDataA)
    votedLabelsTr2 = classifyVoted(votedPairs2, trainDataA)
    votedLabelsTr3 = classifyVoted(votedPairs3, trainDataA)

    votedLabelsTs1 = classifyVoted(votedPairs1, testDataA)
    votedLabelsTs2 = classifyVoted(votedPairs2, testDataA)
    votedLabelsTs3 = classifyVoted(votedPairs3, testDataA)

    trVotError1 = getErr(votedLabelsTr1)
    trVotError2 = getErr(votedLabelsTr2)
    trVotError3 = getErr(votedLabelsTr3)

    tsVotError1 = getErr(votedLabelsTs1)
    tsVotError2 = getErr(votedLabelsTs2)
    tsVotError3 = getErr(votedLabelsTs3)

    print("Training error, voted perceptron, one pass: "+str(round(trVotError1*100, 3))+"%")
    print("Training error, voted perceptron, two passes: "+str(round(trVotError2*100, 3))+"%")
    print("Training error, voted perceptron, three passes: "+str(round(trVotError3*100, 3))+"%")
    print("Test error, voted perceptron, one pass: "+str(round(tsVotError1*100, 3))+"%")
    print("Test error, voted perceptron, two passes: "+str(round(tsVotError2*100, 3))+"%")
    print("Test error, voted perceptron, three passes: "+str(round(tsVotError3*100, 3))+"%")
    print()

    avgLabelsTr1 = classifyAvg(votedPairs1, trainDataA)
    avgLabelsTr2 = classifyAvg(votedPairs2, trainDataA)
    avgLabelsTr3 = classifyAvg(votedPairs3, trainDataA)

    avgLabelsTs1 = classifyAvg(votedPairs1, testDataA)
    avgLabelsTs2 = classifyAvg(votedPairs2, testDataA)
    avgLabelsTs3 = classifyAvg(votedPairs3, testDataA)

    trAvgError1 = getErr(avgLabelsTr1)
    trAvgError2 = getErr(avgLabelsTr2)
    trAvgError3 = getErr(avgLabelsTr3)

    tsAvgError1 = getErr(avgLabelsTs1)
    tsAvgError2 = getErr(avgLabelsTs2)
    tsAvgError3 = getErr(avgLabelsTs3)

    print("Training error, avg perceptron, one pass: "+str(round(trAvgError1*100, 3))+"%")
    print("Training error, avg perceptron, two passes: "+str(round(trAvgError2*100, 3))+"%")
    print("Training error, avg perceptron, three passes: "+str(round(trAvgError3*100, 3))+"%")
    print("Test error, avg perceptron, one pass: "+str(round(tsAvgError1*100, 3))+"%")
    print("Test error, avg perceptron, two passes: "+str(round(tsAvgError2*100, 3))+"%")
    print("Test error, avg perceptron, three passes: "+str(round(tsAvgError3*100, 3))+"%")
    print()


    trainDataB = loadData("hw4btrain.txt")
    testDataB = loadData("hw4btest.txt")

    c0 = perceptron(trainDataB, w0, 0)
    c1 = perceptron(trainDataB, w0, 1)
    c2 = perceptron(trainDataB, w0, 2)
    c3 = perceptron(trainDataB, w0, 3)
    c4 = perceptron(trainDataB, w0, 4)
    c5 = perceptron(trainDataB, w0, 5)
    c6 = perceptron(trainDataB, w0, 6)
    c7 = perceptron(trainDataB, w0, 7)
    c8 = perceptron(trainDataB, w0, 8)
    c9 = perceptron(trainDataB, w0, 9)
    cs = (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)

    multiLabels = classifyMulti(cs, testDataB)
    #print(str(multiLabels))

    ns = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for v in testDataB:
        ns[v[-1]] += 1

    print(str(ns))

    ms = {}
    for i in range(10):
        for j in range(10):
            ms[(i, j)] = 0

    for j in range(10):
        ms[("Don't know", j)] = 0

    for p in multiLabels:
        ms[p] += 1

    print(str(ms))

main()