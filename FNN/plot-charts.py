import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curves(vectorizer_types=None, hidden_layer_sizes=None):
    if vectorizer_types is None:
        vectorizer_types = ['tfidf', 'count']

    if hidden_layer_sizes is None:
        hidden_layer_sizes = [(50,), (100,), (200,), (50, 25), (100, 50), (200, 100), (100, 50, 25)]

    for vec_type in vectorizer_types:
        for hidden_layers in hidden_layer_sizes:
            # Construct directory path
            dir_path = os.path.join(vec_type, str(hidden_layers))
            loss_file = os.path.join(dir_path, "loss.csv")

            # Check if the loss file exists
            if os.path.exists(loss_file):
                # Read the loss data
                loss_data = pd.read_csv(loss_file)

                # Create a figure for this configuration
                plt.figure(figsize=(10, 6))

                # Plot the loss curve
                plt.plot(loss_data['epoch'], loss_data['loss'], 'b-')

                # Add labels and title
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{vec_type.upper()} Model with Hidden Layers: {hidden_layers}')
                plt.grid(True)

                # Save the figure in the same directory as the metrics
                output_file = os.path.join(dir_path, "loss_curve.png")
                plt.savefig(output_file)
                print(f"Saved loss curve to {output_file}")

                # Close the figure to free memory
                plt.close()
            else:
                print(f"Loss file not found: {loss_file}")

def plot_combined_loss_curves(vectorizer_types=None, hidden_layer_sizes=None):
    vectorizer_name = vectorizer_types
    if vectorizer_types is None:
        vectorizer_name = "tfidf and count"
        vectorizer_types = ['tfidf', 'count']

    if hidden_layer_sizes is None:
        hidden_layer_sizes = [(50,), (100,), (200,), (50, 25), (100, 50), (200, 100), (100, 50, 25)]

    plt.figure(figsize=(15, 10))

    for vec_type in vectorizer_types:
        for hidden_layers in hidden_layer_sizes:
            # Construct directory path
            dir_path = os.path.join(vec_type, str(hidden_layers))
            print("\nDIR PATH: ", dir_path)
            loss_file = os.path.join(dir_path, "loss.csv")

            # Check if the loss file exists
            if os.path.exists(loss_file):
                # Read the loss data
                loss_data = pd.read_csv(loss_file)

                # Plot the loss curve
                plt.plot(loss_data['epoch'], loss_data['loss'],
                         label=f"{vec_type.upper()} - {hidden_layers}")

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for {vectorizer_name}')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    output_file =f"{vectorizer_name}_models_loss_curves.png"
    plt.savefig(output_file)
    print(f"Saved combined loss curves to {output_file}")
    plt.close()


if __name__ == "__main__":
    print("Plot loss curves...")
    #plot_loss_curves()
    #plot_combined_loss_curves(["tfidf"])