import model_functions as modfunc
import other_functions as othfunc
import argparse

parser = argparse.ArgumentParser(description="Train a model on a given dataset and save the model as a checkpoint")
parser.add_argument("data_directory", help="Directory where image data is stored")
parser.add_argument("-sd", "--save_dir", help="Directory where the model checkpoint should be saved")
parser.add_argument("-cp", "--checkpoint", help="Checkpoint name for saved model, please use .pth as file type")
parser.add_argument("-a", "--arch", help="Architecture of pre-trained network", default='densenet169')
parser.add_argument("-lr", "--learning_rate", help="Setting the learning rate", type=float, default=0.001)
parser.add_argument("-hu", "--hidden_units", help="Setting the number hidden units in the classifier", type=int, default=800)
parser.add_argument("-do", "--dropout", help="Setting the dropout rate in the classifier", type=float, default=0.5)
parser.add_argument("-ep", "--epochs", help="Setting the dropout rate in the classifier", type=int, default=15)
parser.add_argument("--gpu", help="Use GPU for training", action="store_true")
args = parser.parse_args()

image_datasets, data_loaders, _ = modfunc.transform_load(args.data_directory)
model, criterion, optimizer = modfunc.create_model(arch=args.arch, dropout=args.dropout, hidden_units=args.hidden_units, learning_rate=args.learning_rate)

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
    
modfunc.train_model(image_datasets, data_loaders, model, criterion, optimizer, args.epochs, device)
modfunc.model_test(image_datasets, data_loaders, model, 'test', criterion, device)
    
if args.save_dir:
    othfunc.save_model(image_datasets, args.arch, model, args.dropout, args.hidden_units, args.learning_rate, args.epochs, optimizer, checkpoint=args.save_dir+'/'+args.checkpoint)
    print("Model saved as", args.save_dir+'/'+args.checkpoint)