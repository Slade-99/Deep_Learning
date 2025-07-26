import matplotlib.pyplot as plt

def plot_loss_curves(results: dict[str,list[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    
    accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]
    
    
    epochs = range(len(results["train_loss"]))
    
    
    plt.figure(figsize=(15,7))
    
    
    
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label = "train_loss")
    plt.plot(epochs,test_loss,label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    
    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy,label = "train_accuracy")
    plt.plot(epochs,test_accuracy,label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    
    
    plt.show()