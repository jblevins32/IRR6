## IRR Lab 6

This code generates a classification model for images of traffic signs. 

### How to run
- Create and activate conda environment `conda create env -n <name> -r requirements.txt & conda activate <name>`
- Run test on my model `python3 model_grader.py --data_path ./data/2024F_Gimgs --model_path ./saved_model_best.pkl`
    - Replace the data path as needed

### File Structure
- `data`: git ignored data folder
- `example_knn.py`: Unused, provided knn script
- `generate_requirement.py`: Generate requirements.txt file for environment installation
- `learn_signs.py`: Script for training model
- `model_grader.py`: Grading script
- `README.md`: YOU ARE HERE
- `requirements.txt`: You generated this, good job
- `saved_model_best.pkl`: My best saved model
- `take_picture.sh`: I have no idea what this is...

### How it works
`learn_signs.py` loads the data which includes splitting it into training and validation sets and preprocessing all data. Preprocessing includes:
- Convert images to HSV
- Mask RGB colors from the HSV images
- Greyscale the images
- Choose the color mask with the most true values (i.e. the masking gave more of that color than any other)
- Crop the greyscale image to the chosen color mask
- Check if the ratio of true values in the chosen color mask is too small, and if so, try to find the sign with find contours
    - Crop again to new pixel area if true
- Resize the image
Training is run on an svm, knn, or rf with the processed training set. Finally, the validation dataset is used to predict and thus validate the trained model.