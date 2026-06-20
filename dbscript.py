import mysql.connector

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': ''
}

DB_NAME = 'plantleafconvnext_2025'

def create_database():
    """Drop and recreate database and tables"""
    try:
        # Connect to MySQL server
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Drop database if exists
        print(f"Dropping database '{DB_NAME}' if exists...")
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
        
        # Create database
        print(f"Creating database '{DB_NAME}'...")
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        
        # Close initial connection
        cursor.close()
        conn.close()
        
        # Connect to the new database
        db_config_with_db = DB_CONFIG.copy()
        db_config_with_db['database'] = DB_NAME
        conn = mysql.connector.connect(**db_config_with_db)
        cursor = conn.cursor()
        
        # Create users table
        print("Creating 'users' table...")
        cursor.execute("""
            CREATE TABLE users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(191) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create predictions table
        print("Creating 'predictions' table...")
        cursor.execute("""
            CREATE TABLE predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                image_filename VARCHAR(255) NOT NULL,
                disease_type VARCHAR(255) NOT NULL,
                accuracy FLOAT NOT NULL,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n✓ Database and tables created successfully!")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    create_database()
