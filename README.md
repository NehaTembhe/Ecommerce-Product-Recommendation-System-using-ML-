# 🛒 E-Commerce Product Recommendation System Using Machine Learning
## Overview

This project implements an E-commerce product recommendation system leveraging machine learning techniques within a Flask-based web application. The system aims to enhance user experience by providing personalized product suggestions based on user interactions, product features, and collaborative filtering. It employs a multi-faceted approach, incorporating rating-based recommendations, content-based filtering (utilizing TF-IDF and dimensionality reduction with SVD, accelerated by a FAISS index), collaborative filtering (via user-item interactions and SVD), and hybrid strategies. The backend is developed using Python and the Flask framework, interacting with a MySQL database and utilizing libraries such as Pandas, NumPy, and scikit-learn for data handling and machine learning tasks.

## Key Features

* 🎯**Personalized Product Recommendations:** Delivers tailored suggestions based on user behavior (likes, clicks, searches) and product attributes.
* ⭐**Rating-Based Recommendations:** Showcases top-rated products to all users.
* 🔍**Content-Based Filtering:** Recommends products similar to those a user has shown interest in, analyzing product names, categories, and tags using TF-IDF and SVD for efficient similarity calculations via a FAISS index.
* 🤝**Collaborative Filtering:** Suggests products that users with similar preferences have interacted with, employing user-item interaction data and matrix factorization techniques (Truncated SVD).
* 🔀**Hybrid Recommendation Model:** Integrates content-based and collaborative filtering to leverage the strengths of both approaches and mitigate their limitations (e.g., e.g., the cold-start problem).
* 📊**User Activity Tracking:** Records user interactions (likes, clicks, searches) to refine future recommendations.
* 💖**Wishlist Functionality:** Enables users to add and manage a list of their favorite products.
* 🛒**Shopping Cart Functionality:** Allows users to add and manage items in a shopping cart.
* 🔐**User Authentication:** Implements signup and sign-in functionality to personalize the user experience.
* 🌐**Web-Based User Interface:** Provides an accessible platform for users to interact with the system (likely built with HTML, CSS, and JavaScript within the `templates` and `static` directories).

## Technologies Used

* Python (version 3.7+)
* Flask (for the web application framework)
* Flask-SQLAlchemy (for Object Relational Mapping and interaction with the MySQL database)
* Flask-Migrate (for managing database schema migrations)
* Pandas (for efficient data manipulation and analysis)
* NumPy (for numerical computations)
* Scikit-learn (for various machine learning algorithms including TF-IDF, TruncatedSVD, and cosine similarity)
* FAISS (for fast and efficient similarity search in high-dimensional feature spaces)
* PyMySQL (MySQL database connector for Python)
* HTML, CSS, JavaScript (for the frontend user interface, located in the `static` and `templates` folders)

## Folder Structure

    ├── Python Project                            # Root directory of the project
        ├── .idea/                                # IntelliJ IDEA project configuration files
        ├── .venv/                                # Python virtual environment with installed dependencies
        ├── __pycache__/                          # Python bytecode cache directory
        ├── migrations/                           # Flask-Migrate database migration scripts
        ├── models/                               # Stored machine learning assets and data
        │   ├── faiss_index.idx                   # FAISS index for similarity search
        │   ├── Final_Product_Dataset_with_Tags   # Excel dataset 
        │   ├── svd_model.pkl                     # Trained SVD model for collaborative filtering
        │   ├── tfidf_matrix.pkl                  # Full TF-IDF matrix
        │   ├── tfidf_matrix_reduced.pkl          # Dimensionality-reduced TF-IDF matrix
        │   └── tfidf_vectorizer.pkl              # Trained TF-IDF vectorizer
        ├── static/                               # Static files like Video, images
        │   └── images/                        
        │       ├── v2.mvp
        │       ├── demo video.mvp                #Project Demo Video
        │       └── ...                           # Additional product or UI-related images
        ├── templates/                            # HTML templates for rendering Flask pages
        │   ├── addtocart.html                    # Add to cart page
        │   ├── buynow.html                       # Buy now flow
        │   ├── index.html                        # Homepage or landing page
        │   ├── main.html                         # Main app page or dashboard where user can browse more products
        │   ├── product_detail.html               # Product detail and description view
        │   ├── signin.html                       # User sign-in page
        │   ├── signup.html                       # User registration page
        │   └── wishlist.html                     # Wishlist management page
        ├── amazon_analysis.ipynb                 # Jupyter Notebook for data loading, cleaning, EDA, applying algorithms, and evaluating results and accuracy
        ├── app.py                                # Main Flask application with routes and logic
        ├── requirements.txt                      # Required Python packages for the project
        └── README.md                             # Project overview, setup instructions, and usage guide


## Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone (https://github.com/NehaTembhe/Ecommerce-Product-Recommendation-System-using-ML-.git)
    cd <your_project_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS and Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Database Setup:**

    * Ensure you have a MySQL server running and accessible.
    * Create a database named `ecomm` if it doesn't already exist.
    * Verify and update the database connection URI in the `app.py` file if your MySQL server configuration differs:

        ```python
        app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456789@localhost:3307/ecomm"
        ```

        *(Replace `root`, `123456789`, and `3307` with your MySQL username, password, and port if necessary.)*

5.  **Run Database Migrations:**

    Initialize Flask-Migrate and apply the migrations to create the database tables:

    ```bash
    flask db init
    flask db migrate -m "Initial migration"
    flask db upgrade
    ```

6.  **Precompute Machine Learning Models:**

    Execute the `app.py` script. This will load the training data (`models/Final_Product_Dataset_with_Tags.csv`), precompute the TF-IDF matrix, perform dimensionality reduction using SVD, and build the FAISS index for efficient content-based recommendations. Ensure the training data file is correctly placed in the `models/` directory.

    ```bash
    python app.py
    ```

    *(Note: This step might take some time depending on the size of your dataset and the computational resources available.)*

## Usage

1.  **Run the Flask application:**

    ```bash
    python app.py
    ```

2.  **Access the web application:** Open your web browser and navigate to `http://127.0.0.1:5000/`.

3.  **Explore the features:**

    * 🔐**Signup and Sign-in:** Create a new user account or log in to an existing one to receive personalized recommendations.
    * 🏠**Homepage:** View a selection of trending products and personalized recommendations based on your past interactions (likes,Clicks,searches). If you are a new user, you might initially see only trending products.
    * 🔎**Search:** Use the search bar to look for specific products. The system will display relevant products and may also incorporate content-based recommendations based on your search query.
    * 🛍️**Product Interaction:** Clicking on products you are interested in helps the system learn your preferences for future recommendations.
    * 💖**Wishlist:** Add products to your personal wishlist by clicking the like button associated with a product. Access your saved items through the "Wishlist" link in the navigation.
    * 🛒**(Potentially) Shopping Cart:** Look for options to add products to a shopping cart for potential purchase.


## Evaluation

The recommendation system's performance is evaluated using standard metrics for recommendation systems, including accuracy, precision, recall, F1-score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), as detailed in the project reports. The implemented hybrid approach aims to leverage the strengths of both content-based and collaborative filtering to provide more effective and relevant recommendations.

## Project Demo Video
<h3>Watch Demo Video Here 
Link (https://drive.google.com/file/d/131l_vHf6t1L7Y1DXsM__fbKAeatVWy_d/view?usp=drive_link)



## If you like this project, please give it a 🌟.
## Thank you 😊.
