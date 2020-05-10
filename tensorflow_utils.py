import numpy as np
import matplotlib.pyplot as plt

def create_timeseries_dataset(features, target, history_size, target_size, input_steps=1, convltsm_ts=2,
                              machanism='cnnlstm', multivariate_input=True, multistep_output=False):
    """The Funtion creates Time Series Dataset.
    
    Args:
        features (numpy.array): The Features set
        target (numpy.array): The Target set
        history_size (int): History Size of the fearures
        target_size (int): Target size for the targets
        input_steps (int): Steps between each feature
        convltsm_ts (int): Time Steps for ConvLSTM based models
        machanism (str): Model machenism, the dataset will feed into ['cnn', 'lstm', 'cnnlstm', 'convlstm']
        multivariate_input (bool): Features are univariate or not
        multistep_output (bool): Target is Single step or Multi step
        
    Returns:
        tuple: (Features, Labels) The Time serias dataset 
    
    """
    data = []
    labels = []
    
    start_index = history_size
    end_index = len(features) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, input_steps)
        
        if multivariate_input:
            f = features[indices]
        else:
            # If single variate input, reshape from (x, ) to (x, 1)
            f = features[indices]
            f = np.reshape(f, (f.shape[0], 1))

        if multistep_output:
            # If multistep output, reshape from (y, ) to (y, 1)
            t = target[i:i+target_size]
            labels.append(np.reshape(t, (t.shape[0],1)))
        else:
            labels.append(target[i+target_size])
        
        if machanism == 'convlstm':
            # reshape as (time_series, row=1, col=f.shape[0]/time_series, channels=f.shape[1])
            data.append(np.reshape(f, (convltsm_ts, 1, f.shape[0]//convltsm_ts, f.shape[1])))
        else:
            data.append(f)

    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                       label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def plot_train_history(history, title, previous_loss=None, previous_val=None):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    if previous_val:
        plt.plot(epochs, previous_val, 'r', label='Last Validation loss')
        
    if previous_loss:
        plt.plot(epochs, previous_loss, 'y', label='Last Training loss')
    
    plt.title(title)
    plt.legend()

    plt.show()
    
def multi_step_plot(history, true_future, STEP, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, -1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction),
             label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()