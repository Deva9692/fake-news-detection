import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    text = input("Enter news: ")

    vec = vectorizer.transform([text])
    result = model.predict(vec)

    if result[0] == 1:
        print("Fake News ❌")
    else:
        print("Real News ✅")