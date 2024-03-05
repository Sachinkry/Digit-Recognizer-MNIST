# 95% accuracy on test set
import torch
import torch.nn.functional as F
from prepare_data import prepare_data

# Load and prepare the data
Xtr, Ytr, Xval, Yval, Xte, Yte = prepare_data('mnist-data')

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

# hyperparameters
input_size = 28*28
output_size = 10
hidden_size = 128
batch_size = 32
max_steps = 200000
lr = 0.1

# Parameter initialization
g = torch.Generator().manual_seed(2147483647)  # for reproducibility
W1 = torch.randn((input_size, hidden_size), generator=g) * 0.1
b1 = torch.randn(hidden_size, generator=g) * 0.02
W2 = torch.randn((hidden_size, output_size), generator=g) * 0.01
b2 = torch.randn(output_size, generator=g) * 0

parameters = [W1, b1, W2, b2]
# no. of parameters in total
print(f'Total no. of parameters: ', sum(p.nelement() for p in parameters))  
for p in parameters:
    p.requires_grad = True

# Training the model
for i in range(max_steps):
    # Randomly sample a batch of data
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]  # Xb: batch of images, Yb: batch of labels
    
    # Forward pass
    z1 = Xb.view(-1, 784) @ W1 + b1
    a1 = torch.sigmoid(z1)
    z2 = a1 @ W2 + b2
    pred = softmax(z2)
    
    # Compute loss
    loss = F.cross_entropy(pred, Yb.long())
    
    # Backward pass
    for p in parameters:
        if p.grad is not None:
            p.grad = None
    loss.backward()
    
    # Update parameters
    for p in parameters:
        p.data -= lr * p.grad
    
    # Adjust learning rate after 100,000 steps
    if i == 100000:
        lr = 0.01
    
    # Log progress
    if i % 10000 == 0:
        print(f'Step {i}/{max_steps}: Loss = {loss.item():.4f}')


# evaluation
with torch.no_grad():
    # Forward pass for the test set
    z1_test = Xte.view(-1, 784) @ W1 + b1
    a1_test = torch.sigmoid(z1_test)
    z2_test = a1_test @ W2 + b2
    pred_test = softmax(z2_test)

    # Compute the loss on the test set
    test_loss = F.cross_entropy(pred_test, Yte.long())

    # Get the predicted labels by finding the index of the max log-probability
    pred_labels = torch.argmax(pred_test, dim=1)

    # Compare with true labels to find where predictions are correct
    correct_preds = (pred_labels == Yte.long()).float()  # Convert to float to allow mean calculation
    num_correct = correct_preds.sum().item()  # Total number of correct predictions
    num_samples = Yte.size(0)  # Total number of samples

    # Calculate the accuracy as percentage
    accuracy = 100.0 * num_correct / num_samples

print(f'Test Loss: {test_loss.item():.4f}')
print(f'Correct Predictions: {num_correct}/{num_samples}')
print(f'Accuracy: {accuracy:.2f}%')