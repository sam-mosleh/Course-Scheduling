from typing import List, Any
import multiprocessing as mp
import pandas as pd
from pandas import *
from random import *
import threading
import time
# Defining infinity number
maxInfinity = 1000000
stageSize = 200


# returns random element of a list x
def randE(x, f_lastMore = 0):
    if not f_lastMore:
        ri = randint(0, len(x) - 1)
    else:
        ri = randint(3*len(x)/4, len(x)-1)
    return x[ri]


class Instructor:
    def __init__(self, f_name):
        self.instructorName = f_name
        self.courseList = []
        self.freeTimes = [0 for i in range(5*4)]


class Course:
    def __init__(self, f_name, f_containing=0, f_capacity=maxInfinity, f_timesInWeek=2):
        self.courseName = f_name
        self.capacity = f_capacity
        self.timesInWeek = f_timesInWeek
        self.containing = f_containing
        self.presentors = []


class Classroom:
    def __init__(self, f_name, f_capacity=maxInfinity):
        self.className = f_name
        self.capacity = f_capacity


class Chromosome:
    def __init__(self):
        # Size of a chromosome is 5(Days)*4(Times)*Number of classes
        # Schedule contains ID of instructor and course
        self.scheduleSize = 5*4*len(classrooms)
        self.schedule = [(-1, -1) for i in range(self.scheduleSize)]
        self.__myScore = 0

    def scoring(self):
        return self.__myScore

    def randomInitialize(self):
        # Fill it randomly

        # End if cannot fill it
        if 2*len(courses) > self.scheduleSize:
            return

        for randCours in range(len(courses)):
            for repT in range(courses[randCours].timesInWeek):
                # ri = randint(0, len(courses[randCours].presentors) - 1)
                # randInst = courses[randCours].presentors[ri]
                randInst = randE(courses[randCours].presentors)
                rs = randint(0, self.scheduleSize-1)
                while self.schedule[rs][0] != -1:
                    rs = randint(0, self.scheduleSize - 1)
                self.schedule[rs] = (randInst, randCours)
                # print(rs, randInst, randCours)
        self.fitnessCalculation()


    def mutate(self, f_swapProb=0.01, f_instProb=0.003, f_coursProb=0.003, f_addProb=0.003, f_inverseProb=0.001):
        randprob = random()
        if randprob <= f_swapProb:
            self.mutateBySwap()

        randprob = random()
        if randprob <= f_instProb:
            self.mutateByChangingInstructor()

        randprob = random()
        if randprob <= f_coursProb:
            self.mutateByCourse()

        randprob = random()
        if randprob <= f_addProb:
            self.mutateByAdding()

        randprob = random()
        if randprob <= f_inverseProb:
            self.mutateByInversing()

    # Change one instructor's Course
    def mutateByCourse(self):
        randSch = randrange(self.scheduleSize)
        # Is randSch valid?
        if self.schedule[randSch][0] != -1:
            randCourse = randE(instructorsList[self.schedule[randSch][0]].courseList)
            self.schedule[randSch] = self.schedule[randSch][0], randCourse

    # Change one course's Instructor
    def mutateByChangingInstructor(self):
        randSch = randrange(self.scheduleSize)
        # Is randSch valid?
        if self.schedule[randSch][0] != -1:
            randInst = randE(courses[self.schedule[randSch][1]].presentors)
            self.schedule[randSch] = randInst, self.schedule[randSch][1]

    # Add instructor to one class and time
    def mutateByAdding(self):
        randSch = randrange(self.scheduleSize)
        if self.schedule[randSch][0] == -1:
            randInst = randint(0, len(instructorsList)-1)
            randCours = randE(instructorsList[randInst].courseList)
            self.schedule[randSch] = randInst, randCours

    # Remove one class and time
    # def mutateByRemoving(self):
    #     for i in range(self.mutationSize):
    #         randSch = randint(0, self.scheduleSize - 1)
    #         if self.schedule[randSch][0] != -1:
    #             self.schedule[randSch] = -1, -1

    # Swap 2 blocks in chromosome doing it mutationSize times
    def mutateBySwap(self):
        randSch1 = randrange(self.scheduleSize)
        randSch2 = randrange(self.scheduleSize)
        self.schedule[randSch1], self.schedule[randSch2] = self.schedule[randSch2], self.schedule[randSch1]

    # Inverse an interval
    def mutateByInversing(self):
        randSch1 = randrange(self.scheduleSize)
        randSch2 = randrange(self.scheduleSize)
        if randSch1 > randSch2:
            randSch1, randSch2 = randSch2, randSch1
        indexNum = 0
        while randSch1 + indexNum < randSch2 - indexNum:
            self.schedule[randSch1 + indexNum], self.schedule[randSch2 - indexNum] = self.schedule[randSch2 - indexNum], self.schedule[randSch1 + indexNum]
            indexNum += 1

    # Calculate score of the chromosome
    def fitnessCalculation(self):
        score = 0
        # Located with enough seats (Course cont and class cap)
        for i in range(self.scheduleSize):
            if self.schedule[i][0] != -1 and courses[self.schedule[i][1]].containing <= classrooms[classDayTime(i)[0]].capacity:
                score += 1

        # Professor is not busy
        for i in range(self.scheduleSize):
            if self.schedule[i][0]!=-1:
                _busy = 0
                _not_busy = 1
                indSch = i % 20
                flagOstad = _not_busy
                while indSch<self.scheduleSize:
                    if indSch != i and self.schedule[i][0] == self.schedule[indSch][0]:
                        flagOstad = _busy
                    indSch += 20
                if flagOstad == _not_busy:
                    score += 1


        # Course is not teaching in another class
        for i in range(self.scheduleSize):
            if self.schedule[i][0] != -1:
                _not_taught = 0
                _taught = 1
                indSch = i % 20
                flagDars = _not_taught
                while indSch < self.scheduleSize:
                    if indSch != i and self.schedule[i][1] == self.schedule[indSch][1]:
                        flagDars = _taught
                    indSch += 20
                if flagDars == _not_taught:
                    score += 1

        # Course has been taught more than enough
        courseStats = {}
        for i, j in self.schedule:
            if i != -1:
                if j not in courseStats:
                    courseStats[j] = [i, ]
                else:
                    courseStats[j].append(i)
        for i in courseStats:
            if len(courseStats[i]) <= courses[i].timesInWeek:
                score += 1
            else:
                score -= len(courseStats[i]) - courses[i].timesInWeek
            if len(list(set(courseStats[i]))) == 1:
                score += 1
            else:
                score -= len(list(set(courseStats[i]))) - 1

        # Instructors are available on that time
        for i in range(self.scheduleSize):
            if self.schedule[i][0] != -1:
                if instructorsList[self.schedule[i][0]].freeTimes[i % 20]:
                    score += 1
                else:
                    score -= 1

        self.__myScore = score

def classDayTime(f_n):
    classN = f_n//20
    f_n = f_n%20
    dayN = f_n//4
    timeN = f_n%4
    return classN,dayN,timeN


def readFromExcel():
    global allDays, classrooms, instructorsList, courses
    # Opening Excels
    profskillRead = pd.read_excel('Prof_Skill.xlsx', sheet_name='Sheet1')
    amoozeshRead = pd.read_excel('Amoozesh.xlsx', sheet_name='Sheet1')
    daysRead = pd.read_excel('Proffosor_FreeTime.xlsx')

    # Name of days
    for d in daysRead.index:
        allDays.append(d)
    # Import Classes
    classrooms = [Classroom(i) for i in amoozeshRead.columns]

    # Import Courses
    courses = [Course(i) for i in profskillRead.columns]

    # Import Instructors
    for i in profskillRead.index:
        # Name of Instructor
        newIns = Instructor(i)

        # Time of Instructor
        prTimeRead = pd.read_excel('Proffosor_FreeTime.xlsx', sheet_name=i)
        for j in range(len(prTimeRead.columns)):
            for k in range(len(prTimeRead.index)):
                if prTimeRead[prTimeRead.columns[j]][prTimeRead.index[k]]:
                    newIns.freeTimes[4*k+j] = 1

        # Courses of Instructor
        for j in range(len(profskillRead.columns)):
            if profskillRead[profskillRead.columns[j]][i]:
                newIns.courseList.append(j)
                courses[j].presentors.append(len(instructorsList))
        instructorsList.append(newIns)


def initChromosomes(f_numberOfNodes, f_chromoList):
    for i in range(f_numberOfNodes):
        newChro = Chromosome()
        newChro.randomInitialize()
        newChro.fitnessCalculation()
        f_chromoList.append(newChro)


# Offsprings of Crossover of Two Chromosomes
def crossover(chrom1: Chromosome, chrom2: Chromosome, f_crossoverProb, f_crossoverPointNumber=8):
    firstChromo = Chromosome()
    secondChromo = Chromosome()
    if f_crossoverProb < randrange(100):
        crossoverPoints = {}
        while len(crossoverPoints) < f_crossoverPointNumber:
            randPoint = randrange(stageSize)
            if randPoint not in crossoverPoints:
                crossoverPoints[randPoint] = 1
        firstChoice = randint(0, 1)
        for i in range(chrom1.scheduleSize):
            if firstChoice:
                firstChromo.schedule[i] = chrom1.schedule[i]
                secondChromo.schedule[i] = chrom2.schedule[i]
            else:
                firstChromo.schedule[i] = chrom2.schedule[i]
                secondChromo.schedule[i] = chrom1.schedule[i]
            if i in crossoverPoints:
                firstChoice = (not firstChoice)
        firstChromo.fitnessCalculation()
        secondChromo.fitnessCalculation()
    else:
        firstChromo = chrom1
        secondChromo = chrom2
    return firstChromo, secondChromo


def printCourses():
    tmpCourses = {}
    allc = 0
    for i, j in chromosomeList[-1].schedule:
        if j != -1:
            if i == -1:
                print(i, j)
            if j not in tmpCourses:
                tmpCourses[j] = [i, ]
                allc += 1
            else:
                tmpCourses[j].append(i)
    print('diffrent courses=', allc)
    for i in tmpCourses:
        print(i, tmpCourses[i])


def writeToExcel():
    tim = ['8 - 10', '10 - 12', '14 - 16', '16 - 18']
    time_table_dict = {'Class': [], 'Day': [], 'Time': [], 'Course': [], 'Professor': []}
    # print([i for i in allDays])
    for i in range(chromosomeList[-1].scheduleSize):
        tmpSch = chromosomeList[-1].schedule[i]
        if tmpSch[0] != -1:
            # print(i,classDayTime(i))
            time_table_dict['Class'].append(classrooms[classDayTime(i)[0]].className)
            time_table_dict['Day'].append(allDays[classDayTime(i)[1]])
            time_table_dict['Time'].append(tim[classDayTime(i)[2]])
            time_table_dict['Course'].append(courses[tmpSch[1]].courseName)
            time_table_dict['Professor'].append(instructorsList[tmpSch[0]].instructorName)

    time_table = pd.DataFrame(time_table_dict)
    writer = ExcelWriter('Time_Table.xlsx')
    time_table.to_excel(writer, 'Sheet1', index=False)
    writer.save()


# Gets sorted chromList and select by rank of chromosome
def selectRandomByRank(f_chromList: List[Chromosome]):
    uniformRandomSelect = randrange(stageSize*(stageSize + 1)/2)  # Sum of all ranks
    # find the selection by binary search ==> [right,left]
    rightBound = stageSize
    leftBound = 1
    mid = 0
    while rightBound > leftBound:
        mid = (rightBound + leftBound) // 2
        nowSum = (mid * (mid + 1)) // 2
        if nowSum > uniformRandomSelect:
            rightBound = mid
        elif nowSum < uniformRandomSelect:
            leftBound = mid + 1
        else:
            break
    return f_chromList[mid]


def selectRandomByRWS(f_chromList: List[Chromosome]):
    tmpSum = 0
    for i in f_chromList:
        tmpSum += i.scoring()
    uniformRandomSelect = randrange(tmpSum)  # Sum of all ranks
    resultInd = 0
    while uniformRandomSelect > 0:
        uniformRandomSelect -= f_chromList[resultInd].scoring()
        resultInd += 1
    return f_chromList[resultInd-1]


# X with for main pop with negativity 0 and search pop with negativity 1
def crossoverByCorrolate(f_chromeA, f_chromeB, f_negativity):
    dist = 0
    #print(f_chromeA.scheduleSize, f_chromeB.scheduleSize)
    for i in range(f_chromeA.scheduleSize):
        #print(i)
        if f_chromeA.schedule[i] != f_chromeB.schedule[i]:
            dist += 1
    s = dist/stageSize
    if f_negativity:
        if s > 0.8:
            return crossover(f_chromeA, f_chromeB, 20)
        elif s < 0.2:
            return crossover(f_chromeA, f_chromeB, 100)
        else:
            return crossover(f_chromeA, f_chromeB, 100 - 100 * s)
    else:
        if s > 0.8:
            return crossover(f_chromeA, f_chromeB, 100)
        elif s < 0.2:
            return crossover(f_chromeA, f_chromeB, 20)
        else:
            return crossover(f_chromeA, f_chromeB, 100 * s)


def makeGeneration(f_nowGeneration: List[Chromosome], f_ExploitOrExplore):
    nextGeneration = [] # type: List[Chromosome]
    for popIteration in range(stageSize//4):
        #firstParentChromo = selectRandomByRank(f_nowGeneration)
        #secondParentChromo = selectRandomByRank(f_nowGeneration)
        firstParentChromo = selectRandomByRWS(f_nowGeneration)
        secondParentChromo = selectRandomByRWS(f_nowGeneration)
        firstOffspring, secondOffspring = crossoverByCorrolate(firstParentChromo, secondParentChromo, f_ExploitOrExplore)
        if f_ExploitOrExplore == 0:
            firstOffspring.mutate()
            secondOffspring.mutate()
        else:
            firstOffspring.mutate(0.05, 0.015, 0.015, 0.015, 0.005)
            secondOffspring.mutate(0.05, 0.015, 0.015, 0.015, 0.005)
        firstOffspring.fitnessCalculation()
        secondOffspring.fitnessCalculation()
        nextGeneration.append(firstOffspring)
        nextGeneration.append(secondOffspring)
    return nextGeneration


def hundredGen(f_gen: List[Chromosome], f_ExploitOrExplore):
    tmpGen = f_gen[:]
    for i in range(100):
        tmpGen = makeGeneration(tmpGen, f_ExploitOrExplore)
    return tmpGen


def addBinarySearch(f_list, f_elem):
    leftBound = 0
    rightBound = len(f_list) - 1
    mid = 0
    while rightBound > leftBound:
        mid = (leftBound + rightBound) // 2
        if f_list[mid].scoring() > f_elem.scoring():
            rightBound = mid - 1
        elif f_list[mid].scoring() < f_elem.scoring():
            leftBound = mid + 1
        else:
            break
    f_list.insert(mid, f_elem)


# Dual Population Genetic Algorithm
def dualPopProcess(f_numberOfIterations: int, f_numberOfElitism: int):
    lastGenerationPop = []  # type: List[Chromosome]
    newGenerationPop = []  # type: List[Chromosome]
    mainPop = []  # type: List[Chromosome]
    searchPop = []  # type: List[Chromosome]
    initChromosomes(stageSize, lastGenerationPop)
    lastGenerationPop.sort(key=lambda x: x.scoring())

    lastGenerationPop2 = []  # type: List[Chromosome]
    newGenerationPop2 = []  # type: List[Chromosome]
    mainPop2 = []  # type: List[Chromosome]
    searchPop2 = []  # type: List[Chromosome]
    initChromosomes(stageSize, lastGenerationPop2)
    lastGenerationPop2.sort(key=lambda x: x.scoring())

    for iterNum in range(f_numberOfIterations):
        myPool = mp.Pool()
        processList = list()
        processList.append(myPool.apply_async(hundredGen, (lastGenerationPop, 0)))  # Do main population
        processList.append(myPool.apply_async(hundredGen, (lastGenerationPop, 1)))  # Do searching pop
        processList.append(myPool.apply_async(hundredGen, (lastGenerationPop2, 0)))
        processList.append(myPool.apply_async(hundredGen, (lastGenerationPop2, 1)))
        myPool.close()
        myPool.join()

        mainPop = processList[0].get()[:]
        searchPop = processList[1].get()[:]
        mainPop2 = processList[2].get()[:]
        searchPop2 = processList[3].get()[:]

        newGenerationPop.extend(mainPop)
        newGenerationPop.extend(searchPop)
        newGenerationPop.sort(key=lambda x: x.scoring())

        newGenerationPop2.extend(mainPop2)
        newGenerationPop2.extend(searchPop2)
        newGenerationPop2.sort(key=lambda x: x.scoring())

        # Elitism
        newGenerationPop = newGenerationPop[f_numberOfElitism:]
        for bigLast in lastGenerationPop[(-1)*f_numberOfElitism:]:
            addBinarySearch(newGenerationPop, bigLast)
        lastGenerationPop = newGenerationPop[:]
        newGenerationPop.clear()

        newGenerationPop2 = newGenerationPop2[f_numberOfElitism:]
        for bigLast in lastGenerationPop2[(-1)*f_numberOfElitism:]:
            addBinarySearch(newGenerationPop2, bigLast)
        lastGenerationPop2 = newGenerationPop2[:]
        newGenerationPop2.clear()

    return lastGenerationPop, lastGenerationPop2


# Initialize Global variables

allDays = []
classrooms = []  # type: List[Classroom]
courses = []  # type: List[Course]
instructorsList = []  # type: List[Instructor]
chromosomeList = []  # type: List[Chromosome]
if __name__ == "__main__":
    # Start reading
    readFromExcel()
    # Initialize the Stage list
    #initChromosomes(stageSize)
    # Start timing and processing
    beforeStarting = time.time()
    #multiProcess(40, 4)
    # Run new processes on 2 Threads! then make it 4
    myTmp = dualPopProcess(5, 5)
    print(time.time() - beforeStarting)
    # Print data
    #printCourses()
    print("Stage Size:", stageSize, " and answer sizes==>", len(myTmp[0]), " ", len(myTmp[1]))
    print("And the Pop is:")
    for i in range(stageSize):
        print(myTmp[0][i].scoring(), myTmp[1][i].scoring())
    #for i in range(stageSize):
    #    print(chromosomeList[i].scoring())
    # Write data
    #writeToExcel()
