import os

delete = input('Do you want to delete all csv files, this can not be undone? ')

if delete[0].lower() == 'y':
    confirm = input('To confirm type DELETE ')
    if confirm == 'DELETE':
        # Get the current working directory
        current_directory = os.getcwd()

        # List all files in the current directory
        files = os.listdir(current_directory)

        # Loop through the files and remove .csv files
        for file in files:
            if file.endswith(".xlsx"):
                file_path = os.path.join(current_directory, file)
                os.remove(file_path)
        print(f"Removed file: {file_path}")
else:
    print('Program aborted, files saved')
