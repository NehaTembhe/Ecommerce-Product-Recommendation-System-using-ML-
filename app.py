from flask import Flask, request, render_template, flash, redirect, url_for,  session,jsonify
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os
import logging
from flask_migrate import Migrate
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.decomposition import TruncatedSVD
import numpy as np
from datetime import datetime
from sqlalchemy import desc
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import faiss


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load files
base_dir = os.path.abspath(os.path.dirname(__file__))

# Use relative path for training data to make it portable
train_data_path = os.path.join(base_dir, "models/Final_Product_Dataset_with_Tags.csv")

try:
    train_data = pd.read_csv(train_data_path)
    app.logger.debug(f"Train data loaded successfully with {train_data.shape[0]} rows and {train_data.shape[1]} columns.")
except OSError as e:
    app.logger.error(f"Error loading training data: {e}")
    flash(f"Error loading training data: {e}", 'error')
    train_data = pd.DataFrame()  # Set to an empty DataFrame to avoid further errors


# Debugging paths
app.logger.debug(f"BASE_DIR: {base_dir}")
app.logger.debug(f"Training Data Path: {train_data_path}")

# Database configuration
app.secret_key = "your_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456789@localhost:3307/ecomm"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Replace with your actual database URL
DATABASE_URL = "mysql+pymysql://root:123456789@localhost:3307/ecomm"

# Create the engine
engine = create_engine(DATABASE_URL)

# SQLAlchemy session
Session = sessionmaker(bind=engine)
db_session = Session()  # Renamed to avoid conflicts


# Initialize Flask-Migrate
migrate = Migrate(app, db)
product_id = train_data.loc[0, 'Product_ID']   # Ensure this matches your dataset
train_data['Ratings'] = pd.to_numeric(train_data['Ratings'], errors='coerce')

class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)  # Store the original password


# Define your model class for the 'signin' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    product_id = db.Column(db.String(50), nullable=True) # Ensure this matches the dataset
    action_type = db.Column(db.String(50))  # e.g., "click", "purchase", "search"
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp)
    search_keyword = db.Column(db.String(50))

class Wishlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("signup.id"), nullable=False)  # Ensure this matches the Signup model
    product_id = db.Column(db.String(50), nullable=False)  # Ensure this matches the dataset
    added_on = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Wishlist user_id={self.user_id}, product_id={self.product_id}>"

class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("signup.id"), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    added_on = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Cart user_id={self.user_id}, product_id={self.product_id}, quantity={self.quantity}>"



#Funtions
def truncate_filter(value, length=28):
    """Truncate a string to a specified length."""
    if len(value) > length:
        return value[:length] + "..."
    return value


# Register the custom filter
app.jinja_env.filters['truncate'] = truncate_filter


# User activity tracker
user_activity_data = []


def insert_user_activity(user_id, product_id=None, action_type="search", search_keyword=None):
    """Insert user activity into the database, correctly handling searches and clicks."""

    try:
        user_activity = UserActivity(
            user_id=user_id,
            product_id=str(product_id).strip() if product_id else None,
            action_type=action_type,  # Ensure 'click' is properly stored
            timestamp=datetime.now(),
            search_keyword=search_keyword.strip() if isinstance(search_keyword, str) else None
        )

        db.session.add(user_activity)
        db.session.commit()
        print(f"INSERTED: user_id={user_id}, product_id={repr(product_id)}, action_type={action_type}")
    except Exception as e:
        db.session.rollback()
        print(f"Database error: {e}")

def get_top_rated_products(n=28):
    """ Returns top N products with a rating of 5 and specific columns from the first 2000 rows """
    # Limit the data to first 2000 rows
    limited_train_data = train_data.head(1000000)

    # Convert product_rating to float (if not already)
    limited_train_data['Ratings'] = limited_train_data['Ratings'].astype(float)

    # Filter products with rating of 5.0
    top_rated_products = limited_train_data[limited_train_data['Ratings'] == 5.0]

    if top_rated_products.empty:
        return pd.DataFrame(columns=['Product_Name','Image_URL','Ratings', 'Selling_Price'])

    # Selecting only the required columns
    selected_columns = ['Product_ID', 'Product_Name', 'Image_URL', 'Ratings', 'Selling_Price']
    top_rated_products = top_rated_products[selected_columns]

    return top_rated_products.sample(n=min(n, top_rated_products.shape[0])).reset_index(drop=True)

def preprocess_and_store_tfidf(train_data, svd_components=100):
    """Precompute and save TF-IDF matrix and vectorizer with dimensionality reduction."""
    train_data['combined_text'] = (
        train_data['Product_Name'].astype(str).fillna('') + " " +
        train_data['Category_Name'].astype(str).fillna('') + " " +
        train_data['Tags'].astype(str).fillna('')
    ).str.lower().str.strip()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(train_data['combined_text'])

    # Apply TruncatedSVD to reduce dimensionality
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

    # Save vectorizer, svd model, and reduced matrix
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    svd_path = os.path.join(base_dir, "models", "svd_model.pkl")
    matrix_path = os.path.join(base_dir, "models", "tfidf_matrix_reduced.pkl")

    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    pickle.dump(tfidf, open(vectorizer_path, "wb"))
    pickle.dump(svd, open(svd_path, "wb"))
    pickle.dump(tfidf_matrix_reduced, open(matrix_path, "wb"))

    return tfidf_matrix_reduced

def build_faiss_index(tfidf_matrix_reduced):
    """Create and store FAISS index for fast similarity search."""
    tfidf_matrix_reduced = tfidf_matrix_reduced.astype(np.float32)
    index = faiss.IndexFlatL2(tfidf_matrix_reduced.shape[1])
    index.add(tfidf_matrix_reduced)

    faiss_index_path = os.path.join(base_dir, "models/faiss_index.idx")
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    faiss.write_index(index, faiss_index_path)

    return index

def content_based_recommendations(train_data, item_name, top_n=20):
    """Generate fast content-based recommendations using FAISS."""
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    svd_path = os.path.join(base_dir, "models", "svd_model.pkl")  # Ensure this is defined
    faiss_index_path = os.path.join(base_dir, "models", "faiss_index.idx")

    # Load precomputed vectorizer and SVD model
    tfidf = pickle.load(open(vectorizer_path, "rb"))
    svd = pickle.load(open(svd_path, "rb"))
    index = faiss.read_index(faiss_index_path)

    # Prepare query vector
    item_name = item_name.lower().strip()
    query_vector = tfidf.transform([item_name]).toarray().astype(np.float32)
    query_vector_reduced = svd.transform(query_vector)

    # Perform fast search
    distances, indices = index.search(query_vector_reduced, top_n)

    # Get recommended products
    recommended_products = train_data.iloc[indices[0]].copy()
    recommended_products['Ratings'] = recommended_products['Ratings'].astype(float)

    return recommended_products[['Product_ID', 'Product_Name', 'Image_URL', 'Ratings', 'Selling_Price']]


def collaborative_recommendations(train_data, user_item_matrix, product_id, top_n=20):

    try:
        product_idx = train_data[train_data['Product_ID'] == product_id].index[0]
        user_item_matrix_sparse = csr_matrix(user_item_matrix)
        svd = TruncatedSVD(n_components=50, random_state=42)
        latent_matrix = svd.fit_transform(user_item_matrix_sparse)
        similarity_scores = cosine_similarity(latent_matrix[product_idx].reshape(1, -1), latent_matrix).flatten()
        similar_indices = np.argsort(-similarity_scores)[1:top_n + 1]
        recommended_products = train_data.iloc[similar_indices]
        return recommended_products[['Product_Name', 'Image_URL', 'Ratings', 'Selling_Price', 'Original_Price']]
    except IndexError:
        print(f"[ERROR] Product ID '{product_id}' not found in the dataset.")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Collaborative filtering failed: {e}")
        return pd.DataFrame()

def hybrid_recommendations(train_data, user_item_matrix, item_name,product_id, top_n=20):
    content_recs = content_based_recommendations(train_data, item_name, top_n//2)
    collab_recs = collaborative_recommendations(train_data, user_item_matrix, product_id, top_n//2)
    merged_recs = pd.concat([content_recs, collab_recs]).drop_duplicates()
    return merged_recs.head(top_n)

CATEGORY_TREE = {
    "Beauty & Health": ["Beauty and Personal Care", "Skincare", "Haircare", "Makeup", "Health", "Baby Care"],
    "Fashion": ["Clothing", "Footwear", "Jewelry", "Accessories", "Men", "Women", "Jeans", "Denim", "Boots", "Leather"],
    "Home & Furniture": ["Furniture", "Home Decor", "Bedroom", "Living Room", "Kitchen"],
    "Outdoor & Garden": ["Patio & Garden", "Household Supplies", "Tool & Hardware"],
    "Sports & Leisure": ["Sports & Fitness", "Toys & School Supplies", "Gifts & Registry"],
    "Industrial & Scientific": ["Industrial & Scientific", "Automotive", "Tool & Hardware"],
    "Electronics & Accessories": ["Electronics", "Mobiles & Accessories", "Smartphones", "Laptops", "Tablets", "Wearables"],
    "Office & Stationery": ["Pens & Stationery", "Books"],
    "Pet Supplies": ["Pet Supplies"]
}

def extract_relevant_subcategories(product_name, category, category_tree):
    """A more intelligent way to extract relevant subcategories."""
    relevant_subs = []
    product_name_lower = product_name.lower()
    if category in category_tree and category != 'Unknown':
        for sub in category_tree[category]:
            if sub.lower() in product_name_lower:
                relevant_subs.append(sub)
    # Add some general terms if not already found (only if category is known)
    if category != 'Unknown':
        if "Men" in product_name and "Men" not in relevant_subs:
            relevant_subs.insert(0, "Men") # Put gender earlier
        if "Women" in product_name and "Women" not in relevant_subs:
            relevant_subs.insert(0, "Women") # Put gender earlier
        if category in category_tree and "Clothing" in category_tree[category] and "Clothing" not in relevant_subs:
            relevant_subs.insert(0, "Clothing")
        if "Jeans" in product_name_lower and "Jeans" not in relevant_subs and category in category_tree and "Clothing" in category_tree[category]:
            relevant_subs.append("Jeans")
        if "Denim" in product_name_lower and "Denim" not in relevant_subs and "Jeans" in relevant_subs:
            relevant_subs.append("Denim")
        elif "Denim" in product_name_lower and "Denim" not in relevant_subs and category in category_tree and "Clothing" in category_tree[category]:
            relevant_subs.append("Denim")
        if category in category_tree and "Footwear" in category_tree[category] and "Footwear" not in relevant_subs:
            relevant_subs.insert(0, "Footwear")
        if "Boots" in product_name_lower and "Boots,shoes" not in relevant_subs and category in category_tree and "Footwear" in category_tree[category]:
            relevant_subs.append("Boots")
        # ... add more logic to extract based on keywords

        # Ensure we have at least one subcategory if the category is known
        if not relevant_subs and category in category_tree:
            relevant_subs.append(category_tree[category][0]) # Fallback to the first subcategory
    elif category == 'Unknown':
        return [] # Return an empty list if the category is unknown

    return list(dict.fromkeys(relevant_subs)) # Remove duplicates while preserving order

def build_user_item_matrix(train_data):
    """
    Build a memory-efficient sparse user-item matrix using Product_ID and Ratings.
    If no User_ID exists, we simulate one.
    """
    filtered = train_data.dropna(subset=["Product_ID", "Ratings"])
    product_ids = filtered["Product_ID"].astype(str)

    # Use dummy user if no user column exists
    user_ids = filtered["User_ID"] if "User_ID" in filtered.columns else pd.Series(["user"] * len(filtered))

    ratings = filtered["Ratings"].astype(float)

    # Encode IDs to index numbers
    product_index = pd.Series(product_ids.unique()).reset_index().set_index(0)['index']
    user_index = pd.Series(user_ids.unique()).reset_index().set_index(0)['index']

    row_indices = product_ids.map(product_index).values
    col_indices = user_ids.map(user_index).values

    matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(len(product_index), len(user_index)))

    return matrix, product_index


@app.route("/")
def index():
    """Main page to display trending products and personalized recommendations."""
    username = session.get("username")
    user_id = session.get("user_id")

    # Fetch trending products
    trending_subset = get_top_rated_products(28).to_dict(orient="records")

    recommendations = []  # Initialize list

    if user_id:
        # Load user-item matrix for hybrid model
        user_item_matrix, product_index = build_user_item_matrix(train_data)

        # Fetch liked products
        liked_products = UserActivity.query.filter_by(user_id=user_id, action_type="like").limit(20).all()
        liked_product_ids = [item.product_id for item in liked_products]

        for product_id in liked_product_ids:
            product_name_arr = train_data.loc[train_data["Product_ID"] == product_id, "Product_Name"].values
            if len(product_name_arr) > 0:
                hybrid_recs = hybrid_recommendations(train_data, user_item_matrix, product_name_arr[0], product_id, top_n=5)
                recommendations.extend(hybrid_recs.to_dict(orient="records"))

        # Fetch recent searches and generate smarter recommendations
        recent_searches = UserActivity.query.filter_by(user_id=user_id, action_type="search") \
                                            .order_by(desc(UserActivity.timestamp)).limit(3).all()
        for search in recent_searches:
            search_keyword = search.search_keyword
            search_matches = train_data[
                train_data["Product_Name"].str.contains(search_keyword, case=False, na=False)
            ]
            # Sort by Ratings descending and take top 5
            if not search_matches.empty:
                search_matches = search_matches.sort_values(by="Ratings", ascending=False).head(5)
                recommendations.extend(search_matches.to_dict(orient="records"))

    # Deduplicate recommendations by Product_ID
    recommendations_dict = {rec["Product_ID"]: rec for rec in recommendations}
    final_recommendations = list(recommendations_dict.values())[:20]  # Limit to 20

    return render_template(
        "index.html",
        username=username,
        trending_products=trending_subset,
        recommendations=final_recommendations,
    )


@app.route("/recommendations", methods=["POST", "GET"])
def recommendations():
    """Generate recommendations based on product clicks, searches, or liked products."""
    prod = None
    if request.method == "POST":
        prod = request.form.get("prod", "").strip()
        if not prod:
            return render_template(
                "main.html",
                message="Please enter a product name.",
                username=session.get("username"),
            )
        session["last_search"] = prod  # Store search term in session for pagination
    else:
        prod = request.args.get("prod", session.get("last_search", ""))

    if not prod:
        return render_template(
            "main.html",
            message="Please enter a product name.",
            username=session.get("username"),
        )

    # Log search activity if user is logged in
    user_id = session.get("user_id")
    if user_id:
        insert_user_activity(user_id, product_id=None, action_type="search", search_keyword=prod)

    # Pagination variables
    page = int(request.args.get("page", 1))
    per_page = 40  # Items per page

    # Fetch products based on search
    recommended_products = train_data[train_data["Product_Name"].str.contains(prod, case=False, na=False)]

    # Fetch liked and clicked products for recommendations
    liked_products_rec = []
    clicked_products_rec = []
    if user_id:
        liked_products = UserActivity.query.filter_by(user_id=user_id, action_type="like").all()
        clicked_products = UserActivity.query.filter_by(user_id=user_id, action_type="click").all()

        liked_product_ids = [item.product_id for item in liked_products]
        clicked_product_ids = [item.product_id for item in clicked_products]

        # Generate recommendations based on liked products
        for product_id in liked_product_ids:
            product_name = train_data.loc[train_data["Product_ID"] == product_id, "Product_Name"].values
            if len(product_name) > 0:
                liked_products_rec.extend(
                    content_based_recommendations(train_data, product_name[0], top_n=5).to_dict(orient="records")
                )

        # Generate recommendations based on clicked products
        for product_id in clicked_product_ids:
            product_name = train_data.loc[train_data["Product_ID"] == product_id, "Product_Name"].values
            if len(product_name) > 0:
                clicked_products_rec.extend(
                    content_based_recommendations(train_data, product_name[0], top_n=5).to_dict(orient="records")
                )

    # Merge and deduplicate recommendations
    all_recommendations = recommended_products.to_dict(orient="records") + liked_products_rec + clicked_products_rec
    recommendations_dict = {rec["Product_ID"]: rec for rec in all_recommendations}
    final_recommendations = list(recommendations_dict.values())[:20]  # Limit to 20 recommendations

    # Apply pagination
    total_pages = (len(final_recommendations) // per_page) + (1 if len(final_recommendations) % per_page > 0 else 0)
    paginated_results = final_recommendations[(page - 1) * per_page: page * per_page]

    return render_template(
        "main.html",
        content_based_rec=paginated_results,
        username=session.get("username"),
        page=page,
        total_pages=total_pages,
        prod=prod,  # Pass `prod` to template
    )


@app.route('/main')
def main():
    username = session.get('username')
    user_id = session.get('user_id')

    # Fetch trending products
    trending_subset = get_top_rated_products(28).to_dict(orient="records")

    recommendations = []  # Initialize recommendation list
    user_has_history = False  # Flag for checking user history

    if user_id:
        # Check user activity history
        user_actions = UserActivity.query.filter_by(user_id=user_id).count()
        user_has_history = user_actions > 0  # Set flag if any past actions exist

        # Load user-item matrix for hybrid model
        user_item_matrix, product_index = build_user_item_matrix(train_data)

        # Fetch liked products
        liked_products = UserActivity.query.filter_by(user_id=user_id, action_type="like").limit(20).all()
        liked_product_ids = [item.product_id for item in liked_products]

        for product_id in liked_product_ids:
            product_name_arr = train_data.loc[train_data["Product_ID"] == product_id, "Product_Name"].values
            if len(product_name_arr) > 0:
                hybrid_recs = hybrid_recommendations(train_data, user_item_matrix, product_name_arr[0], product_id,
                                                     top_n=5)
                recommendations.extend(hybrid_recs.to_dict(orient="records"))

        # Fetch recent searches and generate smarter recommendations
        recent_searches = UserActivity.query.filter_by(user_id=user_id, action_type="search") \
            .order_by(desc(UserActivity.timestamp)).limit(3).all()
        for search in recent_searches:
            search_keyword = search.search_keyword
            search_matches = train_data[
                train_data["Product_Name"].str.contains(search_keyword, case=False, na=False)
            ]
            # Sort by Ratings descending and take top 5
            if not search_matches.empty:
                search_matches = search_matches.sort_values(by="Ratings", ascending=False).head(5)
                recommendations.extend(search_matches.to_dict(orient="records"))

    # Deduplicate recommendations by Product_ID
    recommendations_dict = {rec["Product_ID"]: rec for rec in recommendations}
    final_recommendations = list(recommendations_dict.values())[:20]  # Limit to 20 recommendations

    return render_template(
        'main.html',
        username=username,
        content_based_rec=final_recommendations if user_has_history else None,  # Pass None if no history
        trending_products=trending_subset if not user_has_history else None,  # Show trending only if no history
        user_has_history=user_has_history  # Pass user history flag to the template
    )

@app.route("/index")
def indexredirect():
    return redirect(url_for('index'))


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if Signup.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'error')
            return redirect(url_for('signup'))
        if Signup.query.filter_by(email=email).first():
            flash('Email already registered. Please use another email.', 'error')
            return redirect(url_for('signup'))

        # Store the original password in the database
        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        existing_user = Signup.query.filter_by(username=username).first()
        if existing_user:
            if existing_user.password == password:  # Consider hashing for security
                session.clear()  # Clear existing session
                session['user_id'] = existing_user.id  # Store user ID in session
                session['username'] = existing_user.username  # Store username in session
                flash('User signed in successfully!', 'success')
                return redirect(url_for('index'))  # Redirect to index after successful sign-in
            else:
                flash('Invalid password. Please try again.', 'error')
        else:
            flash('Invalid username. Please try again.', 'error')

    return render_template('signin.html')

@app.route("/toggle_like", methods=["POST"])
def toggle_like():
    app.logger.debug("🔁 Received request to toggle like.")

    if "user_id" not in session:
        app.logger.warning("⚠️ User not logged in.")
        return jsonify({"success": False, "message": "User not logged in"}), 401

    try:
        data = request.get_json(force=True)
        app.logger.debug(f"📦 Received data: {data}")
    except Exception as e:
        app.logger.error(f"❌ Failed to parse JSON: {e}")
        return jsonify({"success": False, "message": "Invalid JSON format"}), 400

    user_id = session["user_id"]
    product_id = str(data.get("Product_ID", "")).strip()

    if not product_id or product_id.lower() in ["none", "null", "undefined", ""]:
        app.logger.warning("⚠️ Product ID is missing or invalid.")
        return jsonify({"success": False, "message": "Product ID is required for like action"}), 400

    try:
        wishlist_item = Wishlist.query.filter_by(user_id=user_id, product_id=product_id).first()

        if wishlist_item:
            db.session.delete(wishlist_item)

            # Remove activity if exists
            existing_like = UserActivity.query.filter_by(
                user_id=user_id, product_id=product_id, action_type="like"
            ).first()
            if existing_like:
                db.session.delete(existing_like)
            liked = False
        else:
            db.session.add(Wishlist(user_id=user_id, product_id=product_id))

            # Avoid duplicate like entry
            existing_like = UserActivity.query.filter_by(
                user_id=user_id, product_id=product_id, action_type="like"
            ).first()
            if not existing_like:
                db.session.add(UserActivity(user_id=user_id, product_id=product_id, action_type="like"))
            liked = True

        db.session.commit()

        # Update session liked products (optional)
        session['liked_products'] = [
            item.product_id for item in Wishlist.query.filter_by(user_id=user_id).all()
        ]

        app.logger.info(f"🎉 Product {'liked' if liked else 'unliked'} by user {user_id}.")
        return jsonify({
            "success": True,
            "liked": liked,
            "message": f"Product {'liked' if liked else 'unliked'} successfully."
        })

    except Exception as e:
        db.session.rollback()
        app.logger.exception(f"❌ Error processing like: {e}")
        return jsonify({"success": False, "message": "Internal server error."}), 500


@app.route("/wishlist")
def wishlist():
    if "user_id" not in session:
        return redirect(url_for("signin"))  # Redirect to sign-in if not logged in

    user_id = session.get("user_id")
    username = session.get("username")

    # Fetch wishlist items for the current user
    wishlist_items = Wishlist.query.filter_by(user_id=user_id).all()
    app.logger.debug(f"Fetched wishlist items for user {user_id}: {wishlist_items}")

    # Create a list to hold the product details
    wishlist_data = []
    for item in wishlist_items:
        product = train_data[train_data['Product_ID'] == item.product_id]
        if not product.empty:
            product_info = product.iloc[0]  # Get the first matching product
            wishlist_data.append({
                "product_id": item.product_id,
                "product_name": product_info['Product_Name'],
                "image": product_info.get('Image_URL', ''),  # Use empty string if no image URL
                "category": product_info.get('Category_Name', ''),
                "original_price": product_info.get('Original_Price', 0),
                "discounted_price": product_info.get('Selling_Price', 0),
                "discount": round(((product_info.get('Original_Price', 0) - product_info.get('Selling_Price', 0)) / max(product_info.get('Original_Price', 1),1)) * 100, 2),
            })

    app.logger.debug(f"Wishlist data prepared: {wishlist_data}")

    return render_template(
        "wishlist.html",
        username=username,
        wishlist_items=wishlist_data,
    )



@app.route("/add_to_wishlist", methods=["POST"])
def add_to_wishlist():
    if "user_id" not in session:
        return {"success": False, "message": "User  not logged in"}, 401

    user_id = session["user_id"]
    data = request.json
    product_id = data.get("Product_ID")

    if not product_id:
        return {"success": False, "message": "Product ID is required"}, 400

    try:
        existing_wishlist_item = Wishlist.query.filter_by(user_id=user_id, product_id=product_id).first()
        if existing_wishlist_item:
            return {"success": False, "message": "Product already in wishlist"}, 400

        db.session.add(Wishlist(user_id=user_id, product_id=product_id))
        db.session.commit()
        print(f"Product {product_id} added to wishlist for user {user_id}.")  # Debugging line
        return {"success": True, "message": "Added to wishlist"}
    except Exception as e:
        db.session.rollback()
        print(f"Error adding to wishlist: {e}")  # Debugging line
        return {"success": False, "message": "Database error"}, 500

@app.route("/api/wishlist")
def api_wishlist():
    if "user_id" not in session:
        return jsonify([])  # or an appropriate error structure

    user_id = session["user_id"]
    wishlist_items = Wishlist.query.filter_by(user_id=user_id).all()
    wishlist_data = []
    for item in wishlist_items:
        product = train_data[train_data['Product_ID'] == item.product_id]
        if not product.empty:
            product_info = product.iloc[0]
            wishlist_data.append({
                "product_id": item.product_id,
                "product_name": product_info['Product_Name'],
                "image": product_info.get('Image_URL', ''),
                "category": product_info.get('Category_Name', ''),
                "original_price": product_info.get('Original_Price', 0),
                "discounted_price": product_info.get('Selling_Price', 0),
                # Add a "description" field or remove it from the JS if not available
                "description": "",
                "discount": round(
                    ((product_info.get('Original_Price', 0) - product_info.get('Selling_Price', 0)) /
                     max(product_info.get('Original_Price', 1), 1)) * 100, 2)
            })
    return jsonify(wishlist_data)

@app.route("/remove_from_wishlist", methods=["POST"])
def remove_from_wishlist():
    # Ensure the user is logged in
    if "user_id" not in session:
        return jsonify({"success": False, "message": "User not logged in"}), 401

    user_id = session["user_id"]
    data = request.get_json(force=True)
    product_id = data.get("product_id")

    # Validate the request
    if not product_id:
        return jsonify({"success": False, "message": "Product ID is required"}), 400

    # Look for the wishlist item
    wishlist_item = Wishlist.query.filter_by(user_id=user_id, product_id=product_id).first()
    if not wishlist_item:
        return jsonify({"success": False, "message": "Item not found in wishlist"}), 404

    try:
        db.session.delete(wishlist_item)
        db.session.commit()
        return jsonify({"success": True, "message": "Item removed successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": "Internal server error"}), 500

@app.route("/product/<product_id>")
def product_detail(product_id):
    """View function to display details of a specific product."""
    try:
        username = session.get('username')
        user_id = session.get('user_id')

        # Get the selected product from train_data
        product_data = train_data[train_data['Product_ID'].astype(str) == str(product_id)]
        if product_data.empty:
            return "Product not found", 404

        product = product_data.iloc[0].to_dict()

        # Log the product click if user is logged in
        if user_id:
            insert_user_activity(user_id=user_id, product_id=product_id, action_type="click")

        # Get category and subcategories
        category = product.get("Category_Name", "Unknown")
        product_name = product.get("Product_Name", "")
        subcategories = extract_relevant_subcategories(product_name, category, CATEGORY_TREE)
        product["Subcategories"] = subcategories

        # Breadcrumb building
        breadcrumb = ["Home"]
        if category and category != "Unknown":
            breadcrumb.append(category)
        if subcategories:
            breadcrumb.append(subcategories[0])
        breadcrumb.append(product_name or "Unknown Product")

        # Format rating stars
        rating = int(float(product.get('Ratings', 0)))
        product['RatingStars'] = "★" * min(5, rating) + "☆" * max(0, 5 - rating)

        # ----------- PRICE & DISCOUNT LOGIC -----------
        try:
            selling_price = float(product.get("Selling_Price", 0))
        except:
            selling_price = 0.0

        try:
            original_price = float(product.get("Original_Price", 0))
        except:
            original_price = 0.0

        # If Original Price is not available or invalid, generate it
        if original_price <= selling_price or original_price == 0:
            # Generate 30% to 50% markup based on selling price (1.3 to 1.5 multiplier)
            markup_factor = 1.3 + (hash(product_id) % 21) / 100  # 1.30 to 1.50
            original_price = round(selling_price * markup_factor, 2)

        # Calculate the larger price (ensure it's valid and greater than zero)
        larger_price = max(selling_price, original_price)  # Ensure original price is greater than selling price
        if larger_price == 0:
            larger_price = 1  # Fallback value if both prices are zero

        # Calculate the discount percentage
        if larger_price > 0 and selling_price > 0:
            discount_percent = round(100 * (larger_price - selling_price) / larger_price)
        else:
            discount_percent = 0

        # Assign back to product dict for frontend rendering
        product["Original_Price"] = original_price
        product["Discount"] = discount_percent

        # ----------- LIKE CHECK -----------
        liked = False
        if user_id:
            liked_entry = UserActivity.query.filter_by(
                user_id=user_id,
                product_id=product_id,
                action_type="like"
            ).first()
            liked = bool(liked_entry)

        # ----------- SIMILAR PRODUCTS (FAISS-based) -----------
        try:
            # Use the better content-based method
            recommended_df = content_based_recommendations(train_data, product_name, top_n=40)

            # Exclude the current product itself
            recommended_df = recommended_df[recommended_df["Product_ID"] != product_id]

            # Add fallback if no good matches
            if recommended_df.empty:
                app.logger.warning(f"No content-based matches for product: {product_name}")
                recommended_df = train_data[train_data['Product_ID'] != product_id].sample(n=10, random_state=42)

            interested_products = recommended_df.to_dict(orient="records")
        except Exception as e:
            app.logger.error(f"Failed to fetch content-based recommendations: {e}")
            interested_products = []

        # Debugging log
        app.logger.debug(f"Number of interested products selected: {len(interested_products)}")

        # ----------- RENDER FINAL PAGE -----------
        return render_template(
            "product_detail.html",
            product=product,
            username=username,
            breadcrumb=" > ".join(breadcrumb),
            interested_products=interested_products,
            liked=liked,
            larger_price=larger_price  # Pass larger_price to the template
        )

    except Exception as e:
        import traceback
        app.logger.error(f"Error in product_detail route: {traceback.format_exc()}")
        return "Internal server error", 500




@app.route('/buy-now/', methods=['GET'])
def buy_now():
    # Get the product_id from the query string
    product_id = request.args.get('product_id')

    # Check if product_id is missing
    if not product_id:
        return "Product ID is required", 400

    # Ensure the user is logged in
    if "user_id" not in session:
        return redirect(url_for('signin'))  # ✅ Correct route name

    # Fetch product details from the `train_data` dataframe
    product_data = train_data[train_data['Product_ID'].astype(str) == str(product_id)]
    if product_data.empty:
        return "Product not found", 404

    # Get the product details
    product = product_data.iloc[0].to_dict()

    # ----------- PRICE & DISCOUNT LOGIC -----------
    try:
        selling_price = float(product.get("Selling_Price", 0))
    except:
        selling_price = 0.0

    try:
        original_price = float(product.get("Original_Price", 0))
    except:
        original_price = 0.0

    # Auto-generate original price if not present or invalid
    if not original_price or original_price <= selling_price:
        markup_factor = 1.3 + (hash(product_id) % 21) / 100  # 1.30 to 1.50
        original_price = round(selling_price * markup_factor, 2)

    # Calculate the larger price (used in discount calculation)
    try:
        larger_price = max(selling_price, original_price)
        if larger_price == 0:
            larger_price = 1  # Fallback
    except:
        larger_price = 1

    # Calculate discount percent
    if larger_price > 0 and selling_price > 0:
        discount_percent = round(100 * (larger_price - selling_price) / larger_price)
    else:
        discount_percent = 0

    # Update product dictionary for rendering
    product["Original_Price"] = original_price
    product["Discount"] = discount_percent
    product["Selling_Price"] = selling_price

    return render_template('buynow.html', product=product, larger_price=larger_price)


@app.route("/toggle_cart", methods=["POST"])
def toggle_cart():
    app.logger.debug("🔁 Received request to toggle cart.")

    if "user_id" not in session:
        app.logger.warning("⚠️ User not logged in.")
        return jsonify({"success": False, "message": "User not logged in"}), 401

    try:
        data = request.get_json(force=True)
        app.logger.debug(f"📦 Received data: {data}")
    except Exception as e:
        app.logger.error(f"❌ Failed to parse JSON: {e}")
        return jsonify({"success": False, "message": "Invalid JSON format"}), 400

    user_id = session["user_id"]
    product_id = str(data.get("Product_ID", "")).strip()

    if not product_id or product_id.lower() in ["none", "null", "undefined", ""]:
        app.logger.warning("⚠️ Product ID is missing or invalid.")
        return jsonify({"success": False, "message": "Product ID is required for cart action"}), 400

    try:
        cart_item = Cart.query.filter_by(user_id=user_id, product_id=product_id).first()

        if cart_item:
            db.session.delete(cart_item)
            in_cart = False
        else:
            db.session.add(Cart(user_id=user_id, product_id=product_id))
            in_cart = True

        db.session.commit()

        session['cart_products'] = [
            item.product_id for item in Cart.query.filter_by(user_id=user_id).all()
        ]

        app.logger.info(f"🛒 Product {'added to' if in_cart else 'removed from'} cart by user {user_id}.")
        return jsonify({
            "success": True,
            "in_cart": in_cart,
            "message": f"Product {'added to' if in_cart else 'removed from'} cart successfully."
        })

    except Exception as e:
        db.session.rollback()
        app.logger.exception(f"❌ Error processing cart action: {e}")
        return jsonify({"success": False, "message": "Internal server error."}), 500



@app.route("/cart")
def cart():
    if "user_id" not in session:
        return redirect(url_for("signin"))  # Redirect to sign-in if not logged in

    user_id = session.get("user_id")
    username = session.get("username")

    cart_items = Cart.query.filter_by(user_id=user_id).all()
    app.logger.debug(f"Fetched cart items for user {user_id}: {cart_items}")

    cart_data = []

    for item in cart_items:
        product = train_data[train_data['Product_ID'] == item.product_id]
        if not product.empty:
            product_info = product.iloc[0]
            cart_data.append({
                "product_id": item.product_id,
                "product_name": product_info['Product_Name'],
                "image": product_info.get('Image_URL', ''),
                "category": product_info.get('Category_Name', ''),
                "original_price": product_info.get('Original_Price', 0),
                "discounted_price": product_info.get('Selling_Price', 0),
                "quantity": getattr(item, 'quantity', 1),
                "discount": round(((product_info.get('Original_Price', 0) - product_info.get('Selling_Price', 0)) /
                                   max(product_info.get('Original_Price', 1), 1)) * 100, 2),
            })

    app.logger.debug(f"Cart data prepared: {cart_data}")

    return render_template(
        "addtocart.html",
        username=username,
        cart_items=cart_data,
    )



@app.route("/api/cart")
def api_cart():
    if "user_id" not in session:
        return jsonify([])

    user_id = session["user_id"]
    cart_items = Cart.query.filter_by(user_id=user_id).all()
    cart_data = []

    for item in cart_items:
        product = train_data[train_data['Product_ID'] == item.product_id]
        if not product.empty:
            product_info = product.iloc[0]
            cart_data.append({
                "product_id": item.product_id,
                "product_name": product_info['Product_Name'],
                "image": product_info.get('Image_URL', ''),
                "original_price": product_info.get('Original_Price', 0),
                "discounted_price": product_info.get('Selling_Price', 0),
                "quantity": item.quantity,
                "discount": round(((product_info.get('Original_Price', 0) - product_info.get('Selling_Price', 0)) /
                                  max(product_info.get('Original_Price', 1), 1)) * 100, 2),
            })
        else:
            app.logger.warning(f"Product not found for ID: {item.product_id}")

    return jsonify(cart_data)


@app.route("/remove_from_cart", methods=["POST"])
def remove_from_cart():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "User not logged in"}), 401

    user_id = session["user_id"]
    data = request.get_json(force=True)
    product_id = data.get("product_id")

    if not product_id:
        return jsonify({"success": False, "message": "Product ID is required"}), 400

    cart_item = Cart.query.filter_by(user_id=user_id, product_id=product_id).first()
    if not cart_item:
        return jsonify({"success": False, "message": "Item not found in cart"}), 404

    try:
        db.session.delete(cart_item)
        db.session.commit()
        return jsonify({"success": True, "message": "Item removed from cart successfully"})
    except Exception:
        db.session.rollback()
        return jsonify({"success": False, "message": "Internal server error"}), 500



@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


# Optional debug log
app.logger.debug(f"Received data: {train_data}")


if __name__ == '__main__':
    # Check if the necessary model files exist
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    svd_path = os.path.join(base_dir, "models", "svd_model.pkl")
    faiss_index_path = os.path.join(base_dir, "models/faiss_index.idx")

    if not os.path.exists(vectorizer_path) or not os.path.exists(svd_path) or not os.path.exists(faiss_index_path):
        app.logger.debug("Generating TF-IDF vectorizer and matrix...")
        tfidf_matrix = preprocess_and_store_tfidf(train_data)

        try:
            build_faiss_index(tfidf_matrix)
            app.logger.debug("FAISS index created successfully.")
        except Exception as e:
            app.logger.error(f"Error building FAISS index: {e}")

    app.run(debug=True)






