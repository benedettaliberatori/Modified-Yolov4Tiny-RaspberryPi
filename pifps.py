import torch
from yolo import Yolo
from loss import Loss
from utils import  AverageMeter, use_gpu_if_possible
from pidataset import get_data
import warnings
import time
import tensorflow as tf
from torchvision import transforms
warnings.filterwarnings("ignore")


def test_model(dataloader, device=None):
    start = time.perf_counter()
    #if loss_fn is not None:
    #    loss_meter = AverageMeter()
    
    if device is None:
        device = use_gpu_if_possible()

    interpreter = tf.lite.Interpreter('downblur.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details= interpreter.get_output_details()
    time = 0

    with torch.no_grad():
        for X, y in dataloader:
            
            #y0, y1= (
            #y[0].to(device),
            #y[1].to(device),
       # )
            X = X.numpy()
            X = tf.convert_to_tensor(X)
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], X)
            interpreter.invoke()
            out = [0,0]
            out[1] = torch.from_numpy(interpreter.get_tensor(output_details[0]['index'])).float()
            out[0] = torch.from_numpy(interpreter.get_tensor(output_details[1]['index'])).float()

            end = time.perf_counter()
            time += (end-start)
        
            #loss = (
            #loss_fn(out[0], y0, scaled_anchors[0])
            #+ loss_fn(out[1], y1, scaled_anchors[1])
        #)
        
            
            #if loss_fn is not None:
            #    loss_meter.update(loss.item(), X.shape[0])
            
    # get final performances
    #fin_loss = loss_meter.sum if loss_fn is not None else None


    print(1318/(time))

    #return fin_loss


if __name__ == "__main__":
    num_anchor = 6
    model = Yolo(3, num_anchor //2, 2)

    #model.load_state_dict(torch.load("downblur.pt")) 
    #loss_fn = Loss()
    S=[13, 26]
  

    ANCHORS = [[(0.276  , 0.320312), (0.068  ,  0.113281), (0.03   ,  0.056    )], [(0.017 ,   0.03  ), (0.01 ,  0.018  ), (0.006  , 0.01 )]]

    test_loader = get_data('transform.csv')

    #scaled_anchors = (
    #    torch.tensor(ANCHORS)
    #    * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    #).to("cuda:0")

    test_model(test_loader, device=None)
