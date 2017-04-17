__author__="Anjali Gopal REddy"
import sys
def anatomical_preprocessing(inputFile):
	print "Function for anatomical preprocessing"
def func_preprocessing(inputFile):
	print "Function for functional preprocessing"

def main(arg):
	print "MENU\n"
	print "1. Anatomical Preprocessing\n"
	print "2. Functional Preprocessing\n"
	print "3. Motion Correction\n"
	choice = raw_input("Choose any of the above options: ")
	inputfile = "Provide file here"
	if choice=="1":
		anatomical_preprocessing(inputfile)
	elif choice=="2":
		func_preprocessing(inputfile)
	else:
		print "none"
if __name__=='__main__':
	main(sys.argv[1])
	
