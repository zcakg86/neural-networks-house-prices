from src.pricemodel.models import *
class price_predictor:
    def __init__(self, sequence_dim, spatial_dim, property_dim, hidden_dim=64):
        self.device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = initialmodel(
            sequence_dim,
            spatial_dim,
            property_dim,
            hidden_dim
        ).to(self.device)
        # Specify loss measure
        self.criterion = nn.MSELoss()
        # And Adam optimiser
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def eval(self):
        self.model.eval()

    def train(self, train_loader, val_loader, epochs):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for sequences, spatial, property_feat, targets in train_loader:
                sequences = sequences.to(self.device)
                spatial = spatial.to(self.device)
                property_feat = property_feat.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                predictions, _ = self.model(sequences, spatial, property_feat)
                loss = self.criterion(predictions.squeeze(), targets)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, spatial, property_feat, targets in val_loader:
                    sequences = sequences.to(self.device)
                    spatial = spatial.to(self.device)
                    property_feat = property_feat.to(self.device)
                    targets = targets.to(self.device)

                    predictions, _ = self.model(sequences, spatial, property_feat)
                    loss = self.criterion(predictions.squeeze(), targets)
                    val_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')

        return train_losses, val_losses


