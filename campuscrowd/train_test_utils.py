import torch
import torch.nn.functional as F

def train(model, train_loader, val_loader, static_edge_index, num_epochs, lr):
    loss_fn = torch.nn.MSELoss() 
    train_epoch_losses = []
    val_epoch_losses = []
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start training loop 
    for epoch in range(num_epochs):
        step = 0
        loss_list = []
        for encoder_inputs, labels in train_loader:
            y_hat = model(encoder_inputs, static_edge_index)         # Get model predictions
            loss = loss_fn(y_hat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if step % 100 == 0 :
                print("Epoch {} Step {} train MSE: {:.4f}".format(epoch, step, sum(loss_list)/len(loss_list)))
            step= step+ 1
            
        avg_train_loss = sum(loss_list) / len(loss_list)
        train_epoch_losses.append(avg_train_loss)
        print("Epoch {} Average Training MSE: {:.4f}".format(epoch, avg_train_loss))
    
        # Validation. 
        '''Note that we are just storing validation loss for now 
        instead of picking the model with the best validation loss for hyperparameter tuning. 
        I don't think hyperparameter tuning is necessary 
        since we're solving a new problem in this paper,
         but could be an incremental future direction.'''
        if len(val_loader) > 0: 
            model.eval()
            val_loss_list = []
            with torch.no_grad():
                for val_encoder_inputs, val_labels in val_loader:
                    val_y_hat = model(val_encoder_inputs, static_edge_index)
                    val_loss = loss_fn(val_y_hat, val_labels)
                    val_loss_list.append(val_loss.item())

            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            val_epoch_losses.append(avg_val_loss)
            print("Epoch {} Average Validation MSE: {:.4f}".format(epoch, avg_val_loss))
            model.train()
        
    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr': lr,
                        'train_epoch_losses': train_epoch_losses,
                        'val_epoch_losses': val_epoch_losses,
                      }
    return model, checkpoint_dict

def evaluate(model, test_loader, static_edge_index, checkpoint_dict=dict()): 
    loss_fn = torch.nn.MSELoss()
    torch.no_grad()
    model.eval()
    # Store for analysis
    total_loss = []
    total_mae = []
    
    for encoder_inputs, labels in test_loader:
        # Get model predictions
        y_hat = model(encoder_inputs, static_edge_index)
        # Mean squared error
        loss = loss_fn(y_hat, labels)
        mae = F.l1_loss(y_hat, labels)
        total_loss.append(loss.item())
        total_mae.append(mae.item())
        
    # update checkpoint_dict
    checkpoint_dict['test_mse'] = sum(total_loss)/len(total_loss)
    checkpoint_dict['test_mae'] = sum(total_mae)/len(total_mae)
    
    print("Test MSE: {:.4f}".format(checkpoint_dict['test_mse']))
    print("Test MAE: {:.4f}".format(checkpoint_dict['test_mae']))
    
    return model, checkpoint_dict

def save_or_update_checkpoint(checkpoint_dict, path):
    # save trained model
    torch.save(checkpoint_dict, path)
    print('Model/results saved to '+path)
    
def load_checkpoint(model, path): 
    checkpoint_dict = torch.load(path)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint_dict['lr'])
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    print(model)
    return model, optimizer, checkpoint_dict
