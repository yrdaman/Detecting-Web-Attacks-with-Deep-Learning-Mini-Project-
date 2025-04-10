A mini-project that utilizes deep learning techniques to detect and classify various types of web-based attacks such as SQL Injection, XSS, and DDoS. The system is built using Django for the frontend/backend, with machine learning models like AutoEncoders, SVM, Naive Bayes, and LSTM handling the detection logic.

📌 Project Objective
To build an intelligent, end-to-end system that automatically detects web attacks by analyzing structured and unstructured web request data using deep learning models. This system provides an adaptive and scalable alternative to traditional rule-based intrusion detection systems.

🚀 Features
✅ Upload and preprocess web traffic datasets.
✅ Train and compare multiple ML/DL algorithms:

Support Vector Machine (SVM)

Naive Bayes

AutoEncoder-based anomaly detection

LSTM-based sequential threat detection*

✅ Visualize model performance (accuracy, precision, recall, F1 score).
✅ OTP-based secure login system.
✅ Real-time prediction and classification of test data.

🛠️ Tech Stack
Backend: Django, Python
Frontend: HTML, CSS (Django templates)
Machine Learning: Scikit-learn, Keras, TensorFlow
Data Processing: Pandas, NumPy
Visualization: Matplotlib
Database: MySQL
Email Service: SMTP (Gmail)

📂 Project Structure
WebApp/
├── static/                 # Uploaded CSVs and plots
├── templates/              # HTML templates (Login, Register, Predict, etc.)
├── views.py                # Main application logic
├── urls.py                 # URL routes
├── models.py               # (optional) Django models
├── settings.py             # Django configuration
└── ...

📦 Requirements
Python 3.7+
Django 3.x+
MySQL Server
pip packages:
pandas, numpy, scikit-learn, tensorflow, keras, matplotlib, pymysql

Install all requirements
pip install -r requirements.txt

How to Run
git clone https://github.com/your-username/web-attack-detection.git
cd web-attack-detection
Configure your MySQL database in settings.py.
Run migrations (if using Django models):

python manage.py makemigrations
python manage.py runserver

🧠 Model Training Flow
Upload dataset → Preprocessing (label encoding + one-hot encoding) → Train-test split

Train ML models (SVM, Naive Bayes)

Train deep learning models (AutoEncoder, LSTM)

Evaluate and visualize metrics

Predict attack status from user-uploaded test data

📈 Sample Output
Algorithm 	Accuracy	Precision	Recall	F1 Score
SVM	        92.5%	    90.1%	    89.7%	  89.9%
Naive Bayes	87.3%	    85.2%	    84.5%	  84.8%
AutoEncoder	94.1%	    91.5%	    92.2%	  91.8%
LSTM	      95.2%	    93.0%	    92.7%	  92.8%


🔒 Security Notes
The OTP system is functional via Gmail SMTP.
Be sure to use environment variables to store sensitive email credentials.
SQL queries should be parameterized to avoid SQL injection.

👨‍💻 Author
Naveen Kumar (Mini Project)
Guided by: [S. Rajender]

📃 License
This project is open-source for academic and educational use. Feel free to fork, use, and build upon it. ⭐
