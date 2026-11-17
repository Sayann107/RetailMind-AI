RetailMind AI
RetailMind AI is an incremental end-to-end AI project developed by adding a new machine learning or deep learning capability each day.

The project currently covers customer churn prediction using classical machine learning and deep learning, followed by a computer vision module for fashion image classification.

Project Overview
The project is developed incrementally:

Day 1: Customer Churn Baseline
Day 2: Model Comparison
Day 3: Hyperparameter Tuning and Cross-Validation
Day 4: Feature Engineering and Ensemble Learning
Day 5: Artificial Neural Network
Day 6: CNN Fashion Image Classification
Day 1: Customer Churn Baseline
Objective
Build the first machine learning pipeline for predicting whether a telecom customer is likely to churn.

Dataset
Telco Customer Churn dataset.

The target variable is:

No = Customer does not churn
Yes = Customer churns
Workflow
Load the dataset using Pandas.
Inspect the data.
Handle missing or invalid values.
Remove unnecessary identifiers.
Separate features and target.
Preprocess numerical and categorical features.
Split the data into training and testing sets.
Train the baseline classifier.
Evaluate its performance.
Save the trained pipeline using Joblib.
This forms the baseline for later RetailMind experiments.

Day 2: Multi-Model Comparison
Objective
Compare multiple classification algorithms and determine which models are promising for churn prediction.

Models
Three models were evaluated:

Decision Tree
Random Forest
Support Vector Machine (SVM)
Results
Model	Accuracy	Precision	Recall	ROC-AUC
Decision Tree	0.7289	0.4896	0.5053	0.6573
SVM	0.7913	0.6418	0.4840	0.7905
Random Forest	0.7779	0.6034	0.4759	0.8162
The models show different trade-offs between accuracy, precision, recall and ROC-AUC.

Day 3: Hyperparameter Tuning and Validation
Objective
Improve the competing models using hyperparameter tuning and cross-validation.

Techniques
Day 3 introduces:

GridSearchCV
Cross-validation
Hyperparameter optimisation
Bias and variance analysis
Learning curves
Tuned Models
SVM
Best configuration included:

C = 0.1
Linear kernel
Balanced class weights
Gamma = scale
Results:

CV Training Recall: 0.8355
CV Validation Recall: 0.8355
Test Accuracy: 0.6941
Test Precision: 0.4575
Test Recall: 0.8209
Test F1: 0.5876
Test ROC-AUC: 0.8093
Random Forest
Results:

CV Training Recall: 0.8237
CV Validation Recall: 0.8054
Test Accuracy: 0.7395
Test Recall: 0.7861
Test F1: 0.6157
Test ROC-AUC: 0.8391
Decision Tree
Results:

CV Training Recall: 0.8249
CV Validation Recall: 0.7933
Test Accuracy: 0.7544
Test Recall: 0.7513
Test F1: 0.6189
Test ROC-AUC: 0.8375
Day 3 Champion
SVM was selected as the recall-focused champion.

Its validation recall and training recall were both 0.8355, indicating a small train-validation gap.

The learning curve also stabilised around a validation recall of approximately 0.8355.

Day 4: Feature Engineering and Ensemble Learning
Objective
Improve churn prediction using engineered features, dimensionality reduction and ensemble learning.

Feature Engineering
Additional domain-based features include:

Tenure groups
Average monthly charges
Charge differences
Month-to-month contract flag
Electronic payment flag
Fibre optic service flag
Automatic payment flag
Number of subscribed services
Security/support flags
Senior customer relationship indicators
Techniques
Day 4 explores:

Feature engineering
PCA / feature reduction
Random Forest
XGBoost
Comparison against the Day 3 champion
Results
Model	CV Validation Recall	Test Accuracy	Test Recall	Test F1	Test ROC-AUC
Day 3 Champion	0.8355	0.6941	0.8209	0.5876	0.8093
Random Forest + Features + PCA	0.7385	0.7601	0.7353	0.6194	0.8343
XGBoost + Features + PCA	0.7244	0.7502	0.7166	0.6036	0.8330
Conclusion
The feature-engineered ensemble models achieved higher ROC-AUC and accuracy in some comparisons, but the Day 3 champion retained the strongest recall.

Because detecting churn customers is an important objective, the Day 3 champion remained a strong candidate rather than automatically replacing it with the more complex models.

Day 5: Artificial Neural Network
Objective
Introduce deep learning and determine whether an Artificial Neural Network provides an improvement over the classical machine learning champion.

Architecture
The ANN uses:

Normalised inputs
Dense hidden layers
ReLU activation
Dropout
Sigmoid output
Binary cross-entropy
Adam optimiser
Early stopping
The Sigmoid output provides a probability of customer churn.

Early Stopping
Validation loss is monitored during training.

Training stops when validation performance stops improving, helping reduce overfitting.

ANN vs Classical Machine Learning
After running the Day 5 training script, record the results below.

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Classical Champion	0.6941	0.4575	0.8209	0.5876	0.8093
ANN	ADD_RESULT	ADD_RESULT	ADD_RESULT	ADD_RESULT	ADD_RESULT
Conclusion
The ANN is not automatically considered better because it uses deep learning.

The ANN should only replace the classical champion if its evaluation demonstrates a meaningful benefit.

Day 6: CNN Fashion Image Classification
Objective
Add the first computer vision capability to RetailMind AI using a Convolutional Neural Network.

The CNN classifies fashion images automatically.

Dataset
Fashion-MNIST is used as the Day 6 prototype because an external fashion-product image dataset was not provided.

Fashion-MNIST contains 10 classes:

T-shirt / top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
CNN Pipeline
The image classification process is:

Fashion-MNIST Image → Normalisation → Conv2D → Batch Normalisation → Max Pooling → Conv2D → Batch Normalisation → Max Pooling → Flatten → Dense Layer → Softmax Output → Fashion Category

Evaluation
The CNN is evaluated using:

Training accuracy
Validation accuracy
Test accuracy
Training loss
Validation loss
Confusion matrix
Classification report
The confusion matrix helps identify fashion categories that the CNN commonly confuses.

Day 6 Results
Test Accuracy: ADD_DAY6_ACCURACY

Replace ADD_DAY6_ACCURACY with the actual value generated after running the Day 6 CNN.

Generated Files
The Day 6 script generates:

models/fashion_mnist_cnn.keras
models/day6_fashion_mnist_results.csv
models/day6_fashion_mnist_confusion_matrix.csv
models/day6_fashion_mnist_history.csv
Project Structure
RetailMind/
│
├── data/
│
├── models/
│   ├── churn models
│   ├── model comparison results
│   ├── ANN model and results
│   ├── fashion_mnist_cnn.keras
│   └── Day 6 result files
│
├── src/
│   ├── train.py
│   ├── compare_models.py
│   ├── tune_models.py
│   ├── features.py
│   ├── finalize_model.py
│   ├── predict.py
│   ├── train_ann.py
│   └── train_fashion_mnist_cnn.py
│
├── README.md
├── requirements.txt
└── .gitignore
Technologies Used
Python
Pandas
Scikit-learn
Joblib
XGBoost
TensorFlow
Keras
Machine Learning Techniques
The project currently demonstrates:

Data preprocessing
Train-test splitting
Categorical encoding
Feature scaling
Logistic Regression
Decision Trees
Random Forest
Support Vector Machines
GridSearchCV
Cross-validation
Learning curves
Hyperparameter tuning
Feature engineering
PCA
Bagging
Boosting
XGBoost
Artificial Neural Networks
Convolutional Neural Networks
Early stopping
Image classification
Confusion matrices
Running the Project
Install dependencies:

python -m pip install -r requirements.txt
Example classical ML training:

python src/train.py
Day 3 tuning:

python src/tune_models.py
Day 5 ANN:

python src/train_ann.py
Day 6 CNN:

python src/train_fashion_mnist_cnn.py
Current RetailMind AI Progress
Day	Module	Status
Day 1	Churn Baseline	Completed
Day 2	Model Comparison	Completed
Day 3	Hyperparameter Tuning	Completed
Day 4	Feature Engineering + Ensembles	Completed
Day 5	Artificial Neural Network	Completed / Update Results
Day 6	CNN Fashion Classification	Completed / Update Accuracy
Key Learning
RetailMind AI demonstrates an incremental machine learning development workflow:

Baseline → Compare → Tune → Validate → Engineer Features → Ensemble → Deep Learning → Computer Vision

Each project increment is integrated into the same codebase so that new techniques can be evaluated against previous approaches rather than being treated as isolated exercises.
Day 7: Transfer Learning and Object Detection
Objective
Upgrade the RetailMind AI computer-vision module using transfer learning and object detection.

Transfer Learning
The Day 6 CNN was trained from scratch.

Day 7 uses MobileNetV2 pretrained on ImageNet as the image-feature extractor.

The original MobileNetV2 layers are initially frozen, and a new classification head is trained for the fashion-product categories.

The top layers are then fine-tuned using a lower learning rate.

Transfer-Learning Pipeline
Product image → Image augmentation → MobileNetV2 preprocessing → Pretrained MobileNetV2 → Global average pooling → Dropout → Softmax output → Product category

Transfer-Learning Result
Validation accuracy: ADD_DAY7_TRANSFER_ACCURACY

Replace this placeholder using:

models/day7_transfer_results.csv

Object Detection
A pretrained Ultralytics YOLO model is used to locate objects in catalog images.

Unlike classification, object detection returns:

Object class
Confidence score
Bounding-box coordinates
Annotated image
Retail Applications
Object detection can support:

Automated catalog quality checks
Detection of multiple products in one image
Product and shelf analytics
Verifying whether product images contain expected objects
Generated Files
models/fashion_transfer_model.keras
models/day7_class_names.txt
models/day7_transfer_results.csv
models/day7_transfer_history.csv
models/day7_detection_results.csv
outputs/day7_detection/
Running Day 7
Train transfer-learning classifier:

python src/train_transfer_learning.py

Predict one product category:

python src/predict_transfer_learning.py "path/to/image.jpg"

Run YOLO with the automatic sample:

python src/detect_products.py

Run YOLO on a local image:

python src/detect_products.py "path/to/image.jpg"
Day 8: Grad-CAM Explainability
Objective
Make the RetailMind AI product classifier explainable by identifying the image regions that contribute to each classification.

Approach
Day 8 reuses the Day 7 MobileNetV2 transfer-learning classifier.

No new classification model is trained.

Grad-CAM computes gradients for the predicted class and combines them with the model's convolutional feature maps to create a visual heatmap.

Pipeline
Product Image → Day 7 MobileNetV2 Classifier → Category Prediction → Grad-CAM → Heatmap → Overlay on Original Image → Explainable Product Prediction

Output
For each selected product image, the system produces:

Predicted category
Prediction confidence
Grad-CAM heatmap
Heatmap overlay on the original product image
Example Grad-CAM outputs are stored in:

outputs/day8_gradcam/

Running Day 8
python src/explain_product.py "path/to/product.jpg"

Conclusion
Day 8 completes Phase 2 by adding visual explainability to the RetailMind product-classification module.

Grad-CAM highlights image regions that contribute strongly to the model prediction, helping inspect and understand model behaviour.
Day 9: LSTM Demand Forecasting
Objective
Forecast future store and product demand using an LSTM neural network.

Accurate demand predictions can support inventory planning and help reduce stockouts and overstock.

Dataset
The Store Item Demand Forecasting dataset is used.

The dataset contains:

Date
Store ID
Item ID
Daily sales
The current prototype forecasts demand for Store 1 and Item 1.

Preprocessing
The Day 9 pipeline:

Loads the sales data with Pandas.
Selects one store and item.
Sorts and resamples sales into a daily time series.
Scales sales using training-data minimum and maximum.
Creates 30-day look-back sequences.
Reserves the most recent 30 days for testing.
LSTM Architecture
Input sequence of 30 daily sales values
LSTM layer with 32 units
Dropout
LSTM layer with 16 units
Dropout
Dense layer
Single regression output
Evaluation
The model is evaluated using:

MAE: Mean Absolute Error
RMSE: Root Mean Squared Error
Actual versus predicted demand graph
Lower MAE and RMSE values indicate better forecasting performance.

Results
MAE: ADD_DAY9_MAE

RMSE: ADD_DAY9_RMSE

Replace the placeholders using:

models/day9_results.csv

Forecast
The saved LSTM produces a recursive seven-day demand forecast.

The forecast is stored in:

models/day9_forecast.csv

Generated Files
models/day9_lstm.keras
models/day9_scaling.csv
models/day9_results.csv
models/day9_test_predictions.csv
models/day9_last_window.csv
models/day9_forecast.csv
outputs/day9_actual_vs_predicted.svg
Running Day 9
Train and evaluate the LSTM:

python src/train_lstm_demand.py

Generate the next seven-day forecast:

python src/forecast_demand.py
Day 10: Multi-Horizon Forecasting with Attention
Objective
Upgrade the RetailMind demand forecasting module from single-step LSTM forecasting to multi-horizon forecasting with attention.

Dataset
Day 10 reuses the Store Item Demand data from Day 9.

The current prototype forecasts Store 1 and Item 1.

Sequence Setup
The model uses:

30 historical days as input
7 future days as output
Unlike Day 9, the model predicts all seven future horizons together.

Architecture
The Day 10 model contains:

LSTM sequence encoder
Temporal attention
Dense forecast decoder
Seven-step regression output
Attention
The attention mechanism assigns weights to the historical sequence.

Higher attention weights indicate historical positions that contribute more strongly to the model's representation used for the forecast.

The visualization is stored in:

outputs/day10_attention.svg

Evaluation
The model reports MAE and RMSE separately for:

Day +1
Day +2
Day +3
Day +4
Day +5
Day +6
Day +7
An overall MAE and RMSE are also calculated.

Results are stored in:

models/day10_results.csv

Forecast
The final multi-horizon demand forecast is stored in:

models/day10_forecast.csv

Running Day 10
Train the model:

python src/train_day10_seq2seq.py

Generate the multi-horizon forecast:

python src/forecast_day10.py

Generate the attention visualization:

python src/visualize_day10_attention.py
Day 11: Synthetic Product Image Augmentation
Objective
Improve representation of under-represented fashion-product categories using synthetic image generation.

Dataset
Day 11 reuses the Fashion Product Images dataset used by the RetailMind computer-vision module.

Product metadata is used to identify a comparatively under-represented category.

Approach
A Variational Autoencoder (VAE) is trained using real images from the selected category.

The VAE contains:

Convolutional encoder
Latent representation
Variational sampling
Convolutional decoder
The decoder generates new synthetic product images from latent vectors.

Quality Review
Synthetic images are manually inspected before they are used as augmentation data.

Poor or unrealistic outputs are removed rather than automatically including every generated image.

Augmentation
Approved synthetic images are added to the selected category's real training images.

The classifier can then be retrained using the augmented dataset to compare performance before and after synthetic augmentation.

Generated Files
models/day11_category.txt
models/day11_vae_encoder.keras
models/day11_vae_decoder.keras
models/day11_results.csv
outputs/day11_synthetic/
Running Day 11
Prepare the under-represented category:

python src/prepare_day11_category.py

Train the VAE:

python src/train_day11_vae.py

Generate synthetic product images:

python src/generate_day11_images.py

After manually reviewing generated images:

python src/evaluate_day11_augmentation.py

Conclusion
Day 11 extends RetailMind AI with generative modelling.

Synthetic augmentation is intended to improve representation of under-represented product categories. Generated samples are reviewed before being incorporated into classifier training because synthetic data quality directly affects downstream classification performance.
