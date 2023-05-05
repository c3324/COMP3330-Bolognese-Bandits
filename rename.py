import os
folders = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
parent_folder = 'A2/Q1/datasets/seg_test/seg_test'

for curr_folder in folders:

	folder = parent_folder + "/" + curr_folder
	# Iterate
	for file in os.listdir(folder):
		# Checking if the file is present in the list

		oldName = os.path.join(folder, file)
		print(oldName)

		n = os.path.splitext(file)[0]

		b = curr_folder + "." + n + ".jpg"
		
		newName = os.path.join(parent_folder + "/renamed_files", b)
		print(newName)

		# Rename the file
		os.rename(oldName, newName)
	#print(newName)

#res = os.listdir(folder)
#print(res)