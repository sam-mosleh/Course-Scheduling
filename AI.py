import pandas
print(pandas.__version__)
#get Instructors
#Instructors num
#Dic[I name]->ID
#get Courses
#Dic[C name]->ID
#get Classrooms
#Dic[Class name]->ID

# Defining infinity number
maxInfinity = 1000000


class Instructor:
    def __init__(self, f_name):
        self.instructorName = f_name
        self.courseList = []


class Course:
    def __init__(self, f_name, f_capacity=maxInfinity):
        self.courseName = f_name
        self.capacity = f_capacity


class Classroom:
    def __init__(self, f_name, f_capacity=maxInfinity):
        self.className = f_name
        self.capacity = f_capacity

