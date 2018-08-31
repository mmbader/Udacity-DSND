import model_functions as modfunc
import other_functions as othfunc
import argparse
import os
import random

parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name. Pass in a single image /path/to/image and return the flower name and class probability.")
parser.add_argument("checkpoint", help="Checkpoint name for trained and saved model you want to use for prediction. Please include directory where checkpoint is saved.")
parser.add_argument("-rd", "--random_image_from_path", help="Predict flower name for a random image from this path. This will be done if no path_to_image is specified.", default='flowers/test/')
parser.add_argument("-pti", "--path_to_image", help="Path to the image file. Specify this if you want to predict the label for a specific flower and don't want to choose a random one.")
parser.add_argument("--top_k", help="Return top KK most likely classes", type=int, default=3)
parser.add_argument("-cn", "--category_names", help="Use a mapping of categories to real names, default is 'cat_to_name.json'", default='cat_to_name.json')
parser.add_argument("--gpu", help="Use GPU for inference", action="store_true")
args = parser.parse_args()

if args.gpu:
    device='cuda:0'
else:
    device='cpu'

model_loaded, optimizer, criterion = othfunc.load_checkpoint(args.checkpoint, device)
cat_to_name = othfunc.label_map(label_file=args.category_names)

if args.path_to_image:
    print(args.path_to_image)
    othfunc.predict(cat_to_name, args.path_to_image, model_loaded, device, args.top_k)
else:
    cat = args.random_image_from_path
    cats = os.listdir(cat)
    cat_index = str(random.randrange(0, len(cats)))

    file = args.random_image_from_path + cat_index + '/'
    files = os.listdir(file)
    file_index = random.randrange(0, len(files))

    random_test_image = file+files[file_index]
    print(random_test_image)

    #othfunc.plot_image_with_probs(cat_to_name, random_test_image, model_loaded, args.top_k, device)
    othfunc.predict(cat_to_name, random_test_image, model_loaded, device, args.top_k)
    