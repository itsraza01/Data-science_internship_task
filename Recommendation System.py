#Recommendation_System_Task_1
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate

# Load data from CSV file
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)

# Split data into training and testing sets
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

# Train the model
model = SVD()
model.fit(trainset)

# Test the model
predictions = model.test(testset)

# Get top recommendations for each user
top_n = {}
for uid, iid, true_r, est, _ in predictions:
    if uid not in top_n:
        top_n[uid] = []
    top_n[uid].append((iid, est))
for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n[uid] = user_ratings[:10]

# Print top recommendations for a user
print(top_n[1])
