import torch 
from torch import nn

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training & test loops
def train_test_loop(model: nn.Module, epochs, loss_fn, optimizer, train_loader, test_loader, best_path='best_model_checkpoint.pth'):
    print('Training Started')
    best_test_loss = 100
    train_arr = []
    test_arr = []
    clip_value = 1
    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        print('---------------------')
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            # Get data
            gui = data['gui'].to(device)
            tokens = data['token'].to(device)
            labels = data['label'].to(device)
            # 1. Forward pass
            logits = model(gui, tokens)
            # 2. Calculate the loss
            loss = loss_fn(logits, labels)
            train_loss += loss
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            #4. Loss backward
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            #5. Optimzer_step
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Step {i + 1}: Training loss : {train_loss / (i + 1)}")
        train_loss = train_loss / len(train_loader)
        train_arr.append(train_loss)
        # Testing step
        model.eval()
        with torch.inference_mode():
            test_loss = 0
            for i, data in enumerate(test_loader):
                gui = data['gui'].to(device)
                tokens = data['token'].to(device)
                labels = data['label'].to(device)
                # forward pass
                logits = model(gui, tokens)
                # calculate the loss
                test_loss += loss_fn(logits, labels)
            test_loss = test_loss / len(test_loader)
        test_arr.append(test_loss)
        if best_test_loss > test_loss:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': test_loss,
            }, best_path)
            best_test_loss = test_loss
        print(f"Epoch {epoch + 1} - Training loss = {train_loss} | Testing loss = {test_loss}")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return train_arr, test_arr
