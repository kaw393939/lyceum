/**
 * MongoDB initialization script for Goliath platform
 * 
 * This script runs on MongoDB container startup and sets up:
 * - Database users for each service with appropriate access
 * - Initial collections needed for the application
 * - Indexes for performance optimization
 * 
 * Usage:
 * This script is mounted into the MongoDB container and executed automatically on startup.
 * It uses environment variables from docker-compose.yml to configure users.
 */

// Print initialization beginning
print('Starting MongoDB initialization script...');

// Connect to admin database
db = db.getSiblingDB('admin');

// Auth setup
try {
    // Get user credentials from environment variables
    const PTOLEMY_USER = process.env.PTOLEMY_USER || "ptolemy_user";
    const PTOLEMY_PASSWORD = process.env.PTOLEMY_PASSWORD || "ptolemy_password";
    const GUTENBERG_USER = process.env.GUTENBERG_USER || "gutenberg_user";
    const GUTENBERG_PASSWORD = process.env.GUTENBERG_PASSWORD || "gutenberg_password";
    const SOCRATES_USER = process.env.SOCRATES_USER || "socrates_user";
    const SOCRATES_PASSWORD = process.env.SOCRATES_PASSWORD || "socrates_password";
    
    print(`Setting up service users: [${PTOLEMY_USER}, ${GUTENBERG_USER}, ${SOCRATES_USER}]`);
    
    // Create Ptolemy database and user
    db = db.getSiblingDB('ptolemy');
    
    if (!db.getUser(PTOLEMY_USER)) {
        db.createUser({
            user: PTOLEMY_USER,
            pwd: PTOLEMY_PASSWORD,
            roles: [
                { role: "readWrite", db: "ptolemy" },
                { role: "dbAdmin", db: "ptolemy" }
            ]
        });
        print(`Created Ptolemy user: ${PTOLEMY_USER}`);
    } else {
        print(`Ptolemy user already exists: ${PTOLEMY_USER}`);
    }
    
    // Create initial collections for Ptolemy
    db.createCollection("concepts");
    db.createCollection("domains");
    db.createCollection("relationships");
    db.createCollection("learning_paths");
    db.createCollection("embeddings");
    db.createCollection("activity");
    db.createCollection("cache");
    
    // Create indexes for common queries in Ptolemy
    db.concepts.createIndex({ "name": 1 }, { unique: true });
    db.concepts.createIndex({ "concept_type": 1 });
    db.concepts.createIndex({ "domain": 1 });
    db.concepts.createIndex({ "keywords": 1 });
    db.concepts.createIndex({ "created_at": 1 });
    db.relationships.createIndex({ "source": 1, "target": 1, "type": 1 });
    db.relationships.createIndex({ "source": 1 });
    db.relationships.createIndex({ "target": 1 });
    db.domains.createIndex({ "name": 1 }, { unique: true });
    db.learning_paths.createIndex({ "name": 1 });
    db.learning_paths.createIndex({ "user_id": 1 });
    
    print('Ptolemy collections and indexes created successfully');
    
    // Create Gutenberg database and user
    db = db.getSiblingDB('gutenberg');
    
    if (!db.getUser(GUTENBERG_USER)) {
        db.createUser({
            user: GUTENBERG_USER,
            pwd: GUTENBERG_PASSWORD,
            roles: [
                { role: "readWrite", db: "gutenberg" },
                { role: "dbAdmin", db: "gutenberg" }
            ]
        });
        print(`Created Gutenberg user: ${GUTENBERG_USER}`);
    } else {
        print(`Gutenberg user already exists: ${GUTENBERG_USER}`);
    }
    
    // Create initial collections for Gutenberg
    db.createCollection("templates");
    db.createCollection("content");
    db.createCollection("generations");
    db.createCollection("feedback");
    db.createCollection("media");
    db.createCollection("sessions");
    db.createCollection("cached_data");
    
    // Create indexes for common queries in Gutenberg
    db.templates.createIndex({ "name": 1 }, { unique: true });
    db.templates.createIndex({ "template_type": 1 });
    db.content.createIndex({ "content_id": 1 }, { unique: true });
    db.content.createIndex({ "concept_id": 1 });
    db.content.createIndex({ "content_type": 1 });
    db.content.createIndex({ "created_at": 1 });
    db.generations.createIndex({ "request_id": 1 }, { unique: true });
    db.generations.createIndex({ "status": 1 });
    db.generations.createIndex({ "timestamp": 1 });
    db.feedback.createIndex({ "content_id": 1 });
    db.feedback.createIndex({ "rating": 1 });
    
    print('Gutenberg collections and indexes created successfully');
    
    // Create Socrates database and user
    db = db.getSiblingDB('socrates');
    
    if (!db.getUser(SOCRATES_USER)) {
        db.createUser({
            user: SOCRATES_USER,
            pwd: SOCRATES_PASSWORD,
            roles: [
                { role: "readWrite", db: "socrates" },
                { role: "dbAdmin", db: "socrates" }
            ]
        });
        print(`Created Socrates user: ${SOCRATES_USER}`);
    } else {
        print(`Socrates user already exists: ${SOCRATES_USER}`);
    }
    
    // Create initial collections for Socrates
    db.createCollection("chat_histories");
    db.createCollection("user_sessions");
    db.createCollection("user_preferences");
    db.createCollection("analytics");
    
    // Create indexes for common queries in Socrates
    db.chat_histories.createIndex({ "user_id": 1 });
    db.chat_histories.createIndex({ "created_at": 1 });
    db.chat_histories.createIndex({ "updated_at": 1 });
    db.user_sessions.createIndex({ "user_id": 1 }, { unique: true });
    db.user_sessions.createIndex({ "last_active": 1 });
    
    print('Socrates collections and indexes created successfully');
    
    // Print completion
    print('MongoDB initialization completed successfully');
    
} catch (error) {
    print('Error during MongoDB initialization:');
    print(error);
    throw error;
}