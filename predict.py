"""Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
    - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    - Use GPU for inference: python predict.py input checkpoint --gpu"""

import argparse
import util as util
import model as train_model

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict the most likely categories with a saved model')
    parser.add_argument("checkpoint",help='Saved model', default='/home/workspace/ImageClassifier/checkpoint.pth')
    parser.add_argument("input",help='Image', default='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg')
    parser.add_argument("--category_names", help='Category to Name Mapping',default='cat_to_name.json')
    parser.add_argument("--top_k", default=3, type = int, help='Set number of epochs')
    parser.add_argument("--gpu", action='store_true', help='Defaults to CPU. Change from CPU tp GPU')
    
    args = parser.parse_args()
    print(args)
    
    return args

if __name__ == '__main__':
    args = get_arguments()
    model = util.load_model(args.checkpoint,args.gpu)
    prob,labels,flowers,name = util.predict_cat(args.input,model,args.top_k,args.category_names,args.gpu)
    
    # Print predictions & probabilities
    print("Labeled Flower Name: "+name)
    print("Most Likely Classes & Probability: {} - {:.3f}%".format(flowers[0],prob[0]*100))
    
    print("Predicted Names & Probabilities:")
    for i in range(args.top_k):
        print("{} - {:.3f}%".format(flowers[i],prob[i]*100))
        
     
    