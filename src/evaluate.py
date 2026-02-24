from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_test, y_test, le):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))