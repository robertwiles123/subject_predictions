Grade Prediction Project
This project aims to predict grades using machine learning models. It involves cleaning a CSV file, training several machine learning models, and making predictions based on those models.

Usage Instructions
Prepare the CSV File:

For a combined science class, ensure the CSV file has the following columns and correct headings:
",FFT20,PP,Mock 1,Mock 2,Mock 3,SEN bool"
For triple science, ensure the CSV file has the following columns and correct headings:
",FFT20,PP,year 10 bio grade,year 10 chem grade,year 10 phys grade,year 11 paper 1 bio grade,year 11 paper 1 chem grade,year 11 paper 1 phys grade,year 11 paper 2 bio grade,year 11 paper 2 chem grade,year 11 paper 2 phys grade,SEN bool"
The CSV file should be cleaned, without any missing data.
Place the CSV file in csv_clean folder.
Run the Prediction Script:

Execute the "predictions.py" script.
Provide the name of the CSV file when prompted.
Enter the desired name for the output CSV file when prompted.
Usage Example
bash
Copy code
$ python predictions.py
Enter the name of the CSV file: grades.csv
Enter the name of the output file: predicted_grades.csv
The script will load the specified CSV file, perform predictions using the trained machine learning models, and save the predicted grades in a new CSV file named "predicted_grades.csv".

Restrictions
This code is intended for personal use only. The author does not grant permission for anyone else to use or contribute to this code without explicit authorization.

For any inquiries or issues related to this code, please contact the author directly.

Disclaimer
This project is provided as-is and without warranty. The author is not responsible for any inaccuracies or errors in the predictions made by the machine learning models. Use the predicted grades at your own discretion.




