class Config(object):
    DEBUG = False
    TESTING = False

    ALLOWED_EXTENSIONS = ["MP4", "AVI"]

    UPLOADS = "C:/Users/manue/Documents/Tesis/Code/api/backend/public/uploads/"
    COLLECTIONS = "C:/Users/manue/Documents/Tesis/Code/api/backend/public/collections/"

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True
    
class TestingConfig(Config):
    TESTING = True