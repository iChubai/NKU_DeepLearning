import os

# Create necessary directories
dirs = ['data', 'models', 'results']
for dir_name in dirs:
    os.makedirs(dir_name, exist_ok=True)

print("Project structure created successfully!") 