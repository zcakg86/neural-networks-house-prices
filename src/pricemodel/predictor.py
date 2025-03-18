from src.pricemodel.models import *
class price_predictor:
    def __init__(self, dataset, embedding_dim, hidden_dim, property_dim):
        self.device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = embeddingmodel(dataset, embedding_dim, hidden_dim, property_dim).to(self.device)
        # Specify loss measure
        self.criterion = nn.MSELoss()
        # And Adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        
    def eval(self):
        self.model.eval()

    def train(self, train_loader, val_loader, epochs):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for sequences, spatial, property_feat, targets, sequence_lengths in train_loader:
                sequences = sequences.to(self.device)
                spatial = spatial.to(self.device)
                property_feat = property_feat.to(self.device)
                targets = targets.to(self.device)
                sequence_lengths = sequence_lengths.to(self.device)
                self.optimizer.zero_grad()
                predictions, _ = self.model()

                loss = self.criterion(predictions.squeeze(), targets.squeeze())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                print(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, spatial, property_feat, targets, sequence_lengths in val_loader:
                    sequences = sequences.to(self.device)
                    spatial = spatial.to(self.device)
                    property_feat = property_feat.to(self.device)
                    targets = targets.to(self.device)
                    sequence_lengths = sequence_lengths.to(self.device)
                    predictions, _ = self.model(sequences, spatial, property_feat, sequence_lengths)
                    loss = self.criterion(predictions.squeeze(), targets.squeeze())
                    val_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')

        return train_losses, val_losses


