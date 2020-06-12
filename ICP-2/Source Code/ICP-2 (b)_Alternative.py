# Return every other char of a given string starting with first using a function named “string_alternative
def string_alternative(input):
    return input[::2]


word = input("Enter the word:")
output = string_alternative(word)
print(output)
