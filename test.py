# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on a sample
sample = X_test[0].reshape(1, 28, 28, 1)
prediction = np.argmax(model.predict(sample))
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {prediction}, Actual: {y_test[0]}")
plt.axis('off')
plt.show()
