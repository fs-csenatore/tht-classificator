import pickle

with open('distortion.pkl', 'rb') as file:
    magic_tuple = pickle.load(file)

print("Camera matrix : \n")
print(magic_tuple[1])
print("dist : \n")
print(magic_tuple[2])
print("rvecs : \n")
print(magic_tuple[3])
print("tvecs : \n")
print(magic_tuple[4])