# Smart Attendance System Using Face Recognition

## 1. Abstract

Smart Attendance is a local academic prototype that demonstrates student attendance management with computer vision and deep learning. An administrator stores a student's university ID and profile, captures several face images, and trains a small convolutional neural network (CNN). During a camera check-in, OpenCV detects the largest face and the trained CNN predicts a registered university ID. A confidence threshold is applied before the record is written. SQLite saves the student, date, time, status and check-in method. A database uniqueness constraint prevents a second attendance record for the same student on the same day. An ID-based check-in, attendance dashboard, date-filtered reports and CSV export provide a practical fallback and review path. The project exposes training and validation accuracy, held-out test accuracy and loss, a confusion matrix, a classification report and learning curves. It is designed for learning and supervised demonstration; it does not claim production-grade identity verification.

## 2. Introduction

Attendance is commonly recorded using paper registers or manual spreadsheets. Those approaches can take class time, require later data entry and make it harder to prepare consistent reports. Face recognition is a useful learning problem because it combines image capture, preprocessing, machine learning and application development. This project turns that workflow into a local web dashboard using a webcam, OpenCV, a CNN classifier and SQLite.

## 3. Problem Statement

Manual attendance takes time and may contain omissions, duplicate entries or transcription errors. The goal is to demonstrate a simple system that enrolls students with institutional IDs, records one check-in per day, and makes attendance history and rates easy to review. The face classifier is a supporting academic feature; uncertain predictions require operator review.

## 4. Objectives

1. Register and manage student profiles with unique university IDs.
2. Capture, detect, crop and store face images locally.
3. Train a beginner-sized CNN classifier with a stratified train/test split.
4. Track training, validation and test metrics and display evaluation plots.
5. Recognize a registered student from a webcam frame and record attendance.
6. Support ID-based check-in as a transparent alternative.
7. Prevent duplicate records for one student on one date.
8. Display attendance history, per-student rates and a downloadable CSV.
9. Explain machine learning, ANN, CNN, evaluation and system limitations.

## 5. Existing System

Paper registers and basic spreadsheets are inexpensive and familiar. However, records can be difficult to search, reports require manual work, and duplicate or inconsistent entries can occur. Some commercial biometric systems require dedicated devices or hosted services and may be unsuitable for a small academic demonstration.

## 6. Proposed System

The proposed local web app has an administrator dashboard, a student directory, browser-based camera capture, model training and evaluation, camera and ID attendance modes, and reports. Flask serves the UI and application routes. OpenCV detects faces and prepares image crops. TensorFlow/Keras trains the CNN. SQLite stores admin, student and attendance data. Captured images and model files remain in local folders.

## 7. System Requirements

### Hardware

- 64-bit Windows computer with a webcam.
- Recommended: 8 GB RAM and enough free disk space for the Python environment, photos and TensorFlow.
- CPU is sufficient for this small model; a GPU is optional and not required.

### Software

- Windows 10/11, 64-bit; Python 3.11 recommended (the pinned dependency range supports Python 3.10–3.12).
- Flask, OpenCV, TensorFlow/Keras, NumPy, Pandas, scikit-learn, Matplotlib, SQLite and a modern browser.
- Browser camera permission on the local address `http://127.0.0.1:5000`.

## 8. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application and training code |
| Flask | Local web server, routes, sessions and templates |
| OpenCV | Image decoding, webcam frames and Haar face detection |
| TensorFlow/Keras | CNN definition, training, saving and inference |
| NumPy | Image arrays and model inputs |
| Pandas | Attendance report preparation and CSV output |
| scikit-learn | Stratified train/test split and classification metrics |
| Matplotlib | Accuracy, loss and confusion-matrix figures |
| SQLite | Student, admin and attendance records |
| HTML, CSS, JavaScript | Responsive dashboard and browser camera access |

## 9. Methodology

1. **Registration:** An administrator enters a university ID, name, program, academic year and optional email.
2. **Capture:** The browser requests camera permission. Each image is posted to the local server; OpenCV detects exactly one face, crops it, resizes it to 64 × 64 pixels and saves a JPEG in a folder keyed by the internal student record ID. The model labels map back to university IDs.
3. **Preparation:** The training script loads the images, scales pixel values to 0–1, assigns a numeric class to each university ID and checks the minimum image counts.
4. **Split:** `train_test_split` makes a reproducible stratified training/test split so each class is represented in both sets. Keras reserves a validation portion of the training set while it fits.
5. **Training:** The CNN updates its weights over a small number of epochs using categorical cross-entropy and Adam.
6. **Evaluation:** The held-out test images produce test accuracy, test loss, confusion matrix and classification report. Accuracy and loss curves are saved with Matplotlib.
7. **Recognition:** OpenCV locates a face in a camera frame. The saved CNN predicts a class. Low-confidence predictions are not automatically recorded.
8. **Attendance:** The predicted class maps to an active student record. SQLite writes date, time, present status and method. The unique `(student_id, attendance_date)` key prevents duplicates. An administrator can remove a mistaken record from the history and re-check a student in.
9. **Reporting:** The dashboard groups attendance by date and student, calculates weekday-based rates and exports the selected records as CSV.

## 10. System Architecture

```text
Administrator browser
  ├── Student forms / browser camera / attendance filters
  └── HTML + CSS + JavaScript
          │ HTTP and local camera frames
          ▼
      Flask application
       ├── Sign-in and CSRF protection
       ├── Student and attendance routes ─────── SQLite database
       ├── OpenCV face detection                 ├── admins
       └── TensorFlow/Keras CNN                  ├── students
              ├── dataset/student_<internal-id>/ └── attendance
              └── model/weights + metrics + plots
```

## 11. Data Flow

```text
Student ID and profile → students table
Camera image → OpenCV decode → Haar face crop → local student dataset
Student datasets → normalized arrays → stratified train/test split
Training images → CNN fit + validation tracking → saved Keras model
Test images → predicted ID → accuracy / loss / confusion matrix / report
Live image → OpenCV crop → trained CNN → confidence check → student lookup
Student lookup → unique daily attendance insert → history / rates / CSV
```

## 12. CNN Architecture Explanation

The teaching model accepts a 64 × 64 RGB face crop. Three convolution and pooling blocks learn increasingly useful local visual features. A flatten layer turns the final feature maps into a vector. A 128-unit dense layer combines features; dropout helps limit overfitting; the softmax output produces one probability per enrolled class.

| Layer | Output / function |
|---|---|
| Input | 64 × 64 × 3 normalized pixel values |
| Conv2D, 16 filters, ReLU, same padding | Local color and edge features |
| MaxPooling2D, 2 × 2 | Downsample feature maps |
| Conv2D, 32 filters, ReLU, same padding | More complex patterns |
| MaxPooling2D, 2 × 2 | Reduce spatial size |
| Conv2D, 64 filters, ReLU, same padding | Higher-level image features |
| MaxPooling2D, 2 × 2 | Reduce spatial size |
| Flatten | Convert feature maps to a vector |
| Dense, 128, ReLU | Combine learned features |
| Dropout, 0.35 | Randomly regularize training activations |
| Dense, number of students, softmax | Probabilities for student IDs |

## 13. ANN Explanation

An artificial neural network consists of layers of units. Each unit takes numerical inputs, applies learned weights and a bias, uses an activation function and passes a result to the next layer. Training changes weights to reduce a loss function. The final dense layers in this CNN are conventional ANN layers. A CNN is a specialized ANN: its convolution layers share small filters over image locations, which makes it effective for spatial patterns.

## 14. Machine Learning Explanation

Machine learning is a way to fit a predictive model from examples rather than hand-writing every rule. Here, captured images are examples and each student's university ID is the class label. The model learns associations between image patterns and labels. A correct test score measures performance on held-out samples, but it does not guarantee accuracy on new cameras, lighting, expressions or people.

## 15. Train/Test Split Explanation

`train_test_split` separates the dataset into a training portion for fitting the model and a test portion for evaluation after training. The script sets a fixed random seed for repeatability and stratifies by student class. This avoids placing all examples of a student on only one side. A validation portion is taken from training data during fitting to monitor training progress. Similar images captured in one session can still make a small test score optimistic; stronger experiments should split by capture session or person rather than by near-duplicate frame.

## 16. Database Design

| Table | Important columns | Purpose / rules |
|---|---|---|
| `admins` | `admin_id`, `username`, `password_hash` | Stores login identity; password is a one-way Werkzeug hash. |
| `students` | `student_id`, `university_id`, `name`, `program`, `year`, `email`, `status` | `university_id` is unique; inactive students are retained instead of deleted. |
| `attendance` | `attendance_id`, `student_id`, `attendance_date`, `attendance_time`, `status`, `method` | Foreign key points to a student; unique `(student_id, attendance_date)` prevents duplicate daily attendance. |

SQLite is suitable for a single-computer teaching demo because it runs without a separate database server. It is not configured for concurrent multi-campus production workloads.

## 17. Results

No model score is hard-coded or fabricated. On a first install there are no face photos, so the model is not trained. After valid student images are captured, the **Face model** page displays training accuracy, validation accuracy, test accuracy, test loss, accuracy and loss curves, confusion matrix and per-class precision, recall and F1. Record the observed values and date from the actual demonstration in the final submission. Results depend on capture quality, student count, sample count and the train/test split. Synthetic student records do not include sample faces.

Suggested demonstration record:

| Measure | Observed value |
|---|---|
| Students with face samples | Fill from the model page |
| Images used | Fill from the model page |
| Training accuracy | Fill from the model page |
| Validation accuracy | Fill from the model page |
| Held-out test accuracy | Fill from the model page |
| Test loss | Fill from the model page |

## 18. Advantages

- Simple local install with SQLite and no separate database service.
- Student directory, daily attendance and export in one dashboard.
- Duplicate attendance prevented by a database constraint.
- Demonstrates the full image-to-model-to-record workflow.
- Reports both training progress and held-out metrics.
- ID check-in makes the app usable before face-model setup and supports manual verification.

## 19. Limitations

- The CNN is a small teaching classifier trained only on images captured in this project; it is not a pretrained or production-grade face-recognition system.
- Face recognition can be affected by lighting, pose, occlusion, camera variation and small datasets.
- A confidence score is not proof of identity; presentation attacks and spoofing are not detected.
- A random image split may put very similar captures in both train and test sets and overstate generalization.
- There is no liveness detection, multi-camera support, class timetable, role-based access or institution-wide deployment.
- SQLite is intended for one local demo, not high-volume concurrent use.
- The sample check-ins are synthetic and should be cleared/replaced before presenting real results.
- Users must obtain appropriate consent and follow university policy for face-data collection and retention.

## 20. Future Scope

- Test with session-level splits and larger, consented, varied datasets.
- Evaluate a face-embedding model and nearest-neighbor matching rather than training a small classifier for each class.
- Add liveness checks, audit history, consent and retention controls, and an explicit manual review queue.
- Add class schedules, courses, semesters, absence thresholds and approved leave.
- Add stronger account management, rate limiting, deployment hardening and backups.
- Introduce PostgreSQL or MySQL and role-based access for multiple authorized staff.
- Add PDF summaries and institutional single sign-on where policy permits.

## 21. Conclusion

The project provides a complete local demonstration of student registration, camera image capture, CNN training/evaluation, attendance recording and reporting. It connects machine-learning concepts to a usable dashboard and guards against duplicate daily check-ins at the database layer. The face model is deliberately presented as an academic prototype: its actual performance must be measured with real, appropriately collected samples, and an operator should verify results.

## 22. References

1. TensorFlow, “Install TensorFlow with pip,” <https://www.tensorflow.org/install/pip>.
2. TensorFlow, “Keras Sequential model,” <https://www.tensorflow.org/guide/keras/sequential_model>.
3. OpenCV, “OpenCV: Cascade Classifier,” <https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html>.
4. scikit-learn, “train_test_split,” <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>.
5. scikit-learn, “classification_report,” <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>.
6. Python Software Foundation, “sqlite3 — DB-API 2.0 interface for SQLite databases,” <https://docs.python.org/3/library/sqlite3.html>.
7. Flask, “Quickstart,” <https://flask.palletsprojects.com/en/stable/quickstart/>.
8. NumPy, “NumPy documentation,” <https://numpy.org/doc/stable/>.
9. Pandas, “Pandas documentation,” <https://pandas.pydata.org/docs/>.
10. Matplotlib, “Matplotlib documentation,” <https://matplotlib.org/stable/>.

## Viva Preparation: 30 Questions and Answers

1. **What is the purpose of this project?** — To demonstrate student attendance management using university IDs, a camera, a CNN classifier and a local database.
2. **What is Python?** — A high-level programming language used here for the web application, image processing and model training.
3. **What is Flask?** — A lightweight Python web framework that serves pages and handles application requests.
4. **What is machine learning?** — A method for fitting a model from examples so it can make predictions on new inputs.
5. **What is deep learning?** — A branch of machine learning that uses neural networks with multiple learned layers.
6. **What is an ANN?** — A network of connected computational units that learn weights from examples.
7. **What is a CNN?** — A neural network that uses convolution filters to learn spatial patterns in images.
8. **How are ANN and CNN related?** — A CNN is a type of ANN; it adds convolution layers that are suited to image data.
9. **Why use a CNN for face images?** — It can learn local edges and visual patterns while reusing filters across image positions.
10. **What does OpenCV do here?** — It decodes camera images, detects a face region, crops and resizes it for the model.
11. **What is face detection?** — Finding where a face appears in an image; this project uses a Haar cascade detector.
12. **What is face recognition?** — Predicting which enrolled identity a detected face most resembles according to the trained model.
13. **What is a dataset?** — A collection of input examples and labels used to train or evaluate a model.
14. **What is a class label in this system?** — A university ID assigned to each student's face images during training.
15. **What is preprocessing?** — Preparing raw images, for example by cropping, resizing and scaling pixel values.
16. **What does `train_test_split` do?** — Separates examples into sets used to fit and evaluate a model.
17. **Why keep a test dataset separate?** — To estimate performance on examples the model did not use for fitting.
18. **What is stratification?** — Preserving class proportions across the training and test sets.
19. **What is validation data?** — Training-held-out examples used to monitor a model while it learns.
20. **What is an epoch?** — One pass through the training examples during model fitting.
21. **What is accuracy?** — The fraction of evaluated predictions that match their true labels.
22. **What is loss?** — A numerical measure of prediction error that the optimizer tries to reduce.
23. **What is a confusion matrix?** — A table comparing true classes with predicted classes, showing correct and incorrect predictions.
24. **What are precision and recall?** — Precision measures correctness among predicted matches; recall measures how many true examples were found.
25. **What is dropout?** — A training technique that temporarily drops units to help reduce overfitting.
26. **Why is SQLite used?** — It stores the demo data locally in one file without a separate database server.
27. **How are duplicate check-ins prevented?** — A unique database constraint permits only one `(student_id, attendance_date)` row.
28. **What happens if model confidence is low?** — The app asks for another capture and does not automatically record that face match.
29. **What is the most important limitation?** — The small model can misidentify people and has no liveness detection; results must be verified.
30. **How could the project be improved?** — Use larger session-separated datasets, stronger embeddings and liveness checks, with consent and audit controls.
