import torch
from pidataset import get_data
import warnings
import time
import tensorflow as tf

warnings.filterwarnings("ignore")


def test_model(dataloader):

    interpreter = tf.lite.Interpreter('pruneddb.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details= interpreter.get_output_details()
    t = 0
    i = 0
    with torch.no_grad():
        for X, _ in dataloader:
            print(i)
            i+=1
            
       
            X = X.numpy()
            X = tf.convert_to_tensor(X)
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], X)
            interpreter.invoke()
            out = [0,0]
            out[1] = torch.from_numpy(interpreter.get_tensor(output_details[0]['index'])).float()
            out[0] = torch.from_numpy(interpreter.get_tensor(output_details[1]['index'])).float()

            end = time.perf_counter()
            t += (end-start)
        
            

    print(100/(t))




if __name__ == "__main__":

    test_loader = get_data('transform.csv')

    test_model(test_loader, device=None)
