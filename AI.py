from typing import List, Any

import pandas as pd
from pandas import *
from random import *

# Defining infinity number
maxInfinity = 1000000
stageSize = 400

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
    def __init__(self, f_mutationProb=50, f_mutateSize=20):
        # Size of a chromosome is 5(Days)*4(Times)*Number of classes
        # Schedule contains ID of instructor and course
        self.scheduleSize = 5*4*len(classrooms)
        self.schedule = [(-1, -1) for i in range(self.scheduleSize)]
        self.mutationProbability = f_mutationProb
        self.mutationSize = f_mutateSize
        self.__myScore = 0

    def scoring(self):
        return self.__myScore;

    def randomInitialize(self):
        # Fill it randomly

        # End if cannot fill it
        if 2*len(courses) > self.scheduleSize:
            return

        for randCours in range(len(courses)):
            for repT in range(courses[randCours].timesInWeek):
                ri = randint(0, len(courses[randCours].presentors) - 1)
                randInst = courses[randCours].presentors[ri]
                rs = randint(0, self.scheduleSize-1)
                while self.schedule[rs][0] != -1:
                    rs = randint(0, self.scheduleSize - 1)
                self.schedule[rs] = (randInst, randCours)
                #print(rs, randInst, randCours)
        self.fitnessCalculation()

    def mutate(self):
        if randint(0,100)>self.mutationProbability:
            return
        # Swap 2 blocks in chromosome doing it mutationSize times
        for i in range(self.mutationSize):
            randSch1 = randint(0, self.scheduleSize - 1)
            randSch2 = randint(0, self.scheduleSize - 1)
            while randSch1==randSch2:
                randSch1 = randint(0, self.scheduleSize - 1)
                randSch2 = randint(0, self.scheduleSize - 1)
            self.schedule[randSch1], self.schedule[randSch2] = self.schedule[randSch2], self.schedule[randSch1]
        self.fitnessCalculation()

    def fitnessCalculation(self):
        score = 0
        # Located with enough seats (Course cont and class cap)
        for i in range(self.scheduleSize):
            if self.schedule[i][0]!=-1 and courses[self.schedule[i][1]].containing <= classrooms[classDayTime(i)[0]].capacity:
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


def initChromosomes(f_numberOfNodes):
    for i in range(f_numberOfNodes):
        newChro = Chromosome()
        newChro.randomInitialize()
        chromosomeList.append(newChro)


# Crossover (Child) of two Chromosomes
def crossover(chrom1: Chromosome, chrom2: Chromosome, f_crossoverPointNumber=10):
    chromoSize = chrom1.scheduleSize
    newChromo=Chromosome()
    crossoverPoints = {}
    while len(crossoverPoints) < f_crossoverPointNumber:
        randPoint = randint(0, chromoSize-1)
        if randPoint not in crossoverPoints:
            crossoverPoints[randPoint] = 1
    firstChoice = randint(0, 1)
    for i in range(chromoSize):
        if firstChoice:
            newChromo.schedule[i] = chrom1.schedule[i]
        else:
            newChromo.schedule[i] = chrom2.schedule[i]
        if i in crossoverPoints:
            firstChoice = (not firstChoice)
    newChromo.fitnessCalculation()
    return newChromo


# Selection algorithm and Iteration
def chromosomSelector(f_selectedNodes=30):
    global chromosomeList
    selectionList = []
    chromosomeList.sort(key=lambda x: x.scoring())
    for i in range(f_selectedNodes):
        randChro1 = randint(0, stageSize-1)
        randChro2 = randint(0, stageSize-1)
        #print(randChro1, randChro2, len(chromosomeList))
        newChro = crossover(chromosomeList[randChro1], chromosomeList[randChro2])
        newChro.mutate()
        selectionList.append(newChro)
    tmpIndex = 0
    tmpJndx = 0
    while tmpJndx < len(selectionList):
        if selectionList[tmpJndx].scoring() > chromosomeList[tmpIndex].scoring():
            chromosomeList[tmpIndex] = selectionList[0]
            tmpIndex += 1
        tmpJndx += 1

# Initialize Global variables
allDays = []
classrooms = []  # type: List[Classroom]
courses = []  # type: List[Course]
instructorsList = []  # type: List[Instructor]
chromosomeList = []  # type: List[Chromosome]

# Start reading
readFromExcel()

initChromosomes(stageSize)
# for i in range(100):
#     chromosomSelector()
#print([i.scoring() for i in chromosomeList])
for i in range(400):
    chromosomSelector()
print(chromosomeList[-1].scoring())
for i in range(400):
    chromosomSelector()
print(chromosomeList[-1].scoring())
for i in range(400):
    chromosomSelector()
print(chromosomeList[-1].scoring())
for i in range(400):
    chromosomSelector()
print(chromosomeList[-1].scoring())
for i in range(400):
    chromosomSelector()
print(chromosomeList[-1].scoring())
#print([i.scoring() for i in chromosomeList])
# print(len(chromosomeList))
# print(chromosomeList[0].schedule)
# print(chromosomeList[1].schedule)
# chrNC = crossover(chromosomeList[0],chromosomeList[1])
# print(chrNC.schedule)
# chromosomeList[0].fitnessCalculation()
# chromosomeList[1].fitnessCalculation()
# chrNC.fitnessCalculation()
# print(chromosomeList[0].scoring())
# print(chromosomeList[1].scoring())
# print(chrNC.scoring())

tim = ['8 - 10', '10 - 12', '14 - 16', '16 - 18']
time_table_dict = {'Class': [], 'Day': [], 'Time': [], 'Course': [], 'Professor': []}
#print([i for i in allDays])
for i in range(chromosomeList[-1].scheduleSize):
    tmpSch = chromosomeList[-1].schedule[i]
    if tmpSch[0] != -1:
        #print(i,classDayTime(i))
        time_table_dict['Class'].append(classrooms[classDayTime(i)[0]].className)
        time_table_dict['Day'].append(allDays[classDayTime(i)[1]])
        time_table_dict['Time'].append(tim[classDayTime(i)[2]])
        time_table_dict['Course'].append(courses[tmpSch[1]].courseName)
        time_table_dict['Professor'].append(instructorsList[tmpSch[0]].instructorName)

time_table = pd.DataFrame(time_table_dict)
writer = ExcelWriter('Time_Table.xlsx')
time_table.to_excel(writer, 'Sheet1', index=False)
writer.save()