#%%
from src.pricemodel.models import *
import torch 

class price_predictor:
    def __init__(self, embedding_dim, hidden_dim, property_dim, community_length,
                 community_feature_dim, week_length, year_length):
        self.device = torch.device('mps' if torch.mps.is_available() 
                                   else 'cuda' if torch.cuda.is_available() 
                                   else 'cpu')
        self.model = embeddingmodel(embedding_dim, hidden_dim, property_dim, 
                                    community_length,community_feature_dim, 
                                    week_length, year_length).to(self.device)
        # Specify loss measure
        self.criterion = nn.MSELoss()
        # And Adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.community_length = community_length
    def eval(self):
        self.model.eval()

    def train(self, train_loader, val_loader, epochs):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                # Move each tensor in the batch to the device
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the batch
                community, community_features, year, week, property, targets = batch
                self.optimizer.zero_grad()
                print("Community Indices Min:", community.min())
                print("Community Indices Max:", community.max())
                print(self.community_length)
                predictions, _ = self.model(community, community_features, year,
                                            week, property, targets)

                loss = self.criterion(predictions.squeeze(), targets.squeeze())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                print(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Move each tensor in the batch to the device
                    batch = tuple(t.to(self.device) for t in batch)
                    # Unpack the batch
                    community, community_features, year, week, property, targets = batch
                    self.optimizer.zero_grad()
                    predictions, _ = self.model(community, community_features, year, week, property, targets)
                    val_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')

        return train_losses, val_losses



# %%
