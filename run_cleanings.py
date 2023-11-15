import subprocess

# Replace 'script1.py' and 'script2.py' with the actual filenames of the scripts you want to run
script1 = 'grades_full_dirty/full_split.py'
script2 = 'grades_split_dirty/broad_cleaning.py'

# Run the first script
process1 = subprocess.Popen(['python', script1])

# Run the second script
process2 = subprocess.Popen(['python', script2])

# Wait for both processes to complete
process1.wait()
process2.wait()