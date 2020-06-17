# Creating a class "Employee"
class Employee:
    # Creating members to calculate Average salary and keep track of count of employees
    numofEmp = 0
    total_salary = 0

    # Constructor for Initializing the name, family salary and department
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.numofEmp += 1
        Employee.total_salary += salary

    # Function for calculating the average salary of all employees
    def get_avg_sal(self):
        average_sal = self.total_salary / self.numofEmp
        print("The average salary is: " + str(average_sal))
        return

    # Function to display employee Information
    def display(self):
        print("Name:" + self.name, "Family:" + self.family, "Salary:" + str(self.salary),
              "Department:" + self.department)

# Creating a class FulltimeEmployee
class FulltimeEmployee(Employee):

    # Inheriting the properties of Employee during initialization
    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)

# e1 and e2 are the instances created for Employee class
print("-" * 50)
e1 = Employee("Shravyala", "Keesari", 7000, "Software")
e1.display()
print("-" * 50)
e2 = Employee("Prajwala", "Keesari", 8000, "QA-Testing")
e2.display()

# e3 and e4 are the instances created for FulltimeEmployee class
print("-" * 50)
e3 = FulltimeEmployee("Ramesh", "Kothapally", 9000, "Development")
e3.display()
print("-" * 50)
e4 = FulltimeEmployee("Sweety", "Reddy", 8500, "Design")
e4.display()
print("-" * 50)

print("The number of the employees is : ", Employee.numofEmp)
Employee.get_avg_sal(Employee)
print("-" * 50)