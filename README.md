Machine Learning Model Training and Testing
This repository contains code for training and testing machine learning models to predict two types of grades: Science Combined Classes and Science Triple. The code is designed to clean the input data, train the models, generate predictions, and evaluate their performance.

Files and Folders
1. clean_csv.py
This file is responsible for cleaning the input data for both Science Triple and Science Combined Classes. It preprocesses the data to ensure it is suitable for training the machine learning models.

2. model_predictions.py
Running this script generates a Pandas DataFrame containing predicted grades from multiple models. The code implements various machine learning models to train and generate predictions.

3. grades_packages folder
This folder contains code modules used in various other packages related to grades prediction. It may include helper functions, preprocessing utilities, and custom model architectures.

4. model_scores.txt
This text file provides a record of the scores achieved by different models during evaluation, along with brief thoughts or observations on their performance. It serves as a reference for comparing the effectiveness of different models.

5. TODO.txt
This file outlines the remaining tasks or improvements that need to be addressed in the codebase. It serves as a reminder for future development and enhancements.

Usage
To train and test the machine learning models, follow these steps:

Ensure that the necessary dependencies and libraries are installed. Refer to the requirements.txt file for the list of required packages.

Execute the clean_csv.py script to clean the input data for Science Triple and Science Combined Classes.

Run the model_predictions.py script to make predictions based on trained models.

Evaluate the performance of the models using the generated predictions and compare the scores in the model_scores.txt file.

Contributing
This is a personal project and therefore is not taking controbuters at the moment.

License
All rights reserved. This code is proprietary and confidential. No part of this code may be reproduced, distributed, or transmitted in any form or by any means without the prior written permission of the author or organization. 

