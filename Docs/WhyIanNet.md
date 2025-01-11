# Why IanNet?

## Each layer gets its own optimizer

```csharp
public Net MakeTheNetwork()
{
	var net = new Net();
	var learningRate = 0.1f;
	
	var inputLayer = new Input2DLayer<Image>(new Shape2D(28, 28));
	inputLayer.SetPreprocess(Preprocess);
	
	var convLayer = new Conv2D(new Shape2D(3, 3));
	convLayer.SetInitializer(new HeUniform(9, scale: 1.0f / 741f));
	convLayer.SetActivation(new ReLU2D());
	
	var poolingLayer = new MaxPooling2D(new Shape2D(2, 2));
	var flattenLayer = new Flatten1D();
	
	var denseLayer1 = new Layer1D(100);
	denseLayer1.SetActivation(new ReLU1D());
	denseLayer1.SetInitializer(new HeUniform(100));
	denseLayer2.SetOptimizer(new SGD(learningRate));
	
	var denseLayer2 = new Layer1D(10);
	denseLayer2.SetActivation(new None1D());
	denseLayer2.SetInitializer(new HeUniform(10));
	denseLayer2.SetOptimizer(new Adam(learningRate));
	
	var softmaxLayer = new Softmax1D();
	
	int numberOfLabels = Enum.GetValues(typeof(Label)).Length;
	var outputLayer = new Output1DLayer<Label>(numberOfLabels);
	outputLayer.SetProcessing(new EnumProcessing1D<Label>());
	// The output layer uses categorical cross-entropy by default
	
	net.AddLayer(inputLayer);
	net.AddLayer(convLayer);
	net.AddLayer(poolingLayer);
	net.AddLayer(flattenLayer);
	net.AddLayer(denseLayer1);
	net.AddLayer(denseLayer2);
	net.AddLayer(softmaxLayer);
	net.AddLayer(outputLayer);
	
	return net;
}

public void Train(LabelledBatch<object, object> batch)
{
	var options = new TrainingOptions()
	{
		Epochs = 20,
		TrackCategoricalCrossEntropy = true,
	};
	
	Net.Train(batch, options);
}

```

```python
# Copied from ChatGPT

import tensorflow as tf
from tensorflow.keras import Model, layers

# Define a simple model with two convolutional layers
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)

# Initialize the model and optimizers
model = CustomModel()
optimizer_conv1 = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_conv2 = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Training step
@tf.function
def train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)

    # Compute gradients for each layer
    grads_conv1 = tape.gradient(loss, model.conv1.trainable_variables)
    grads_conv2 = tape.gradient(loss, model.conv2.trainable_variables)

    # Apply gradients using layer-specific optimizers
    optimizer_conv1.apply_gradients(zip(grads_conv1, model.conv1.trainable_variables))
    optimizer_conv2.apply_gradients(zip(grads_conv2, model.conv2.trainable_variables))

    return loss

# Dummy data for training
x_train = tf.random.normal((32, 28, 28, 1))  # Batch of 32 grayscale images (28x28)
y_train = tf.random.uniform((32,), maxval=10, dtype=tf.int32)  # Random labels (10 classes)

# Training loop
epochs = 5
for epoch in range(epochs):
    loss = train_step(x_train, y_train)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")
```