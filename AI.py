from typing import List, Any

import pandas as pd
from pandas import *
from random import *

# Defining infinity number
maxInfinity = 1000000


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
        self.schedule = [(0, 0) for i in range(self.scheduleSize)]
        self.setFlag=[0 for i in range(self.scheduleSize)]

    def randomInitialize(self):
        # Fill it randomly

        # End if cannot fill it
        if 2*len(courses)>self.scheduleSize :
            return

        for randCours in range(len(courses)):
            for repT in range(courses[randCours].timesInWeek):
                randInst = randint(0, len(courses[randCours].presentors) - 1)
                rs = randint(0, self.scheduleSize-1)
                while self.setFlag[rs]:
                    rs = randint(0, self.scheduleSize - 1)
                self.setFlag[rs] = 1
                self.schedule[rs] = (randInst, randCours)



def readFromExcel():
    global allDays, classrooms, instructorsList, courses
    # Opening Excels
    profskillRead = pd.read_excel('Prof_Skill.xlsx', sheet_name='Sheet1')
    amoozeshRead = pd.read_excel('Amoozesh.xlsx', sheet_name='Sheet1')
    daysRead = pd.read_excel('Proffosor_FreeTime.xlsx')

    # Name of days
    for d in daysRead:
        allDays.append(d)

    # Import Classes
    classrooms=[Classroom(i) for i in amoozeshRead.columns]

    # Import Courses
    courses=[Course(i) for i in profskillRead.columns]
    print
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


# Initialize Global variables
allDays = []
classrooms = []  # type: List[Classroom]
courses = []  # type: List[Course]
instructorsList = []  # type: List[Instructor]
chromosomeList = []  # type: List[Chromosome]

# Start reading
readFromExcel()


